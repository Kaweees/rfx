"""
rfx.real.so101 - SO-101 arm hardware backend via Zenoh transport pipeline.

All hardware I/O goes through a Rust RobotNode that publishes state and
accepts commands over the Zenoh transport.  Python never touches serial
directly.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..observation import make_observation
from ..robot.config import SO101_CONFIG, RobotConfig
from ..runtime.otel import get_tracer, init_otel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _port_map_path() -> Path:
    """Path for persisted SO-101 leader/follower port identity mapping."""
    raw = os.getenv("RFX_SO101_PORT_MAP")
    if raw:
        return Path(raw).expanduser()
    config_home = os.getenv("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else (Path.home() / ".config")
    return base / "rfx" / "so101_port_map.json"


def _port_fingerprint(port_info: dict[str, Any]) -> str:
    """Stable-ish USB adapter fingerprint used to preserve arm role assignments."""
    serial = str(port_info.get("serial_number") or "").strip()
    if serial:
        return f"serial:{serial}"
    location = str(port_info.get("location") or "").strip()
    if location:
        return f"location:{location}"
    hwid = str(port_info.get("hwid") or "").strip()
    if hwid:
        return f"hwid:{hwid}"
    desc = str(port_info.get("description") or "").strip()
    if desc:
        return f"description:{desc}"
    return f"port:{port_info.get('port', '')}"


def _load_port_map() -> dict[str, Any]:
    path = _port_map_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_port_map(mapping: dict[str, Any]) -> None:
    path = _port_map_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def _remember_pair(leader: dict[str, Any], follower: dict[str, Any]) -> None:
    try:
        _save_port_map(
            {
                "leader_fingerprint": _port_fingerprint(leader),
                "follower_fingerprint": _port_fingerprint(follower),
                "leader_last_port": str(leader.get("port", "")),
                "follower_last_port": str(follower.get("port", "")),
            }
        )
    except Exception as exc:
        logger.debug("Failed to persist SO-101 port map: %s", exc)


def _so101_ports_with_meta() -> list[dict[str, Any]]:
    from .. import node as _node_mod

    return [
        p for p in _node_mod.discover_ports() if p.get("robot_type") == "so101" and p.get("port")
    ]


def _wait_for_port_count(count: int, timeout_s: float = 20.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ports = _so101_ports_with_meta()
        if len(ports) == count:
            return ports
        time.sleep(0.2)
    return _so101_ports_with_meta()


def _interactive_pair_setup() -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not sys.stdin.isatty():
        return None

    print("\nSO-101 first-time mapping wizard")
    print("This will map which physical arm is leader/follower across replug/reboot.")
    input("1) Unplug BOTH SO-101 USB adapters, then press Enter...")
    none_ports = _wait_for_port_count(0, timeout_s=30.0)
    if none_ports:
        print(f"Expected 0 SO-101 ports, found {len(none_ports)}. Skipping wizard.")
        return None

    input("2) Plug ONLY the LEADER arm USB adapter, then press Enter...")
    one_ports = _wait_for_port_count(1, timeout_s=30.0)
    if len(one_ports) != 1:
        print(f"Expected 1 SO-101 port after leader plug, found {len(one_ports)}. Skipping wizard.")
        return None
    leader = one_ports[0]
    print(f"Leader detected on {leader.get('port')}")

    input("3) Plug the FOLLOWER arm USB adapter, then press Enter...")
    two_ports = _wait_for_port_count(2, timeout_s=30.0)
    if len(two_ports) != 2:
        print(
            f"Expected 2 SO-101 ports after follower plug, found {len(two_ports)}. Skipping wizard."
        )
        return None

    leader_fp = _port_fingerprint(leader)
    follower_candidates = [p for p in two_ports if _port_fingerprint(p) != leader_fp]
    if len(follower_candidates) != 1:
        print("Could not uniquely identify follower port. Skipping wizard.")
        return None
    follower = follower_candidates[0]
    print(f"Follower detected on {follower.get('port')}")
    return leader, follower


def _auto_port() -> str:
    """Auto-detect a single SO-101 Feetech USB-serial port."""
    from .. import node as _node_mod

    ports = _node_mod.discover_so101_ports()
    if not ports:
        raise RuntimeError("No SO-101 arm detected. Plug in the Feetech USB adapter and try again.")
    if len(ports) > 1:
        logger.info("Multiple SO-101 ports found: %s — using %s", ports, ports[0])
    return ports[0]


def _auto_pair() -> tuple[str, str]:
    """Auto-detect a leader/follower SO-101 pair (two USB adapters)."""
    all_ports = _so101_ports_with_meta()
    if len(all_ports) < 2:
        raise RuntimeError(
            f"Need 2 SO-101 arms for leader-follower, found {len(all_ports)}. "
            f"Plug in both Feetech USB adapters."
        )
    if len(all_ports) > 2:
        logger.info(
            "Found %d SO-101 ports: %s — selecting by saved identity or deterministic fallback",
            len(all_ports),
            [p.get("port") for p in all_ports],
        )

    mapping = _load_port_map()
    by_fingerprint = {_port_fingerprint(p): p for p in all_ports}

    leader: dict[str, Any] | None = by_fingerprint.get(str(mapping.get("leader_fingerprint", "")))
    follower: dict[str, Any] | None = by_fingerprint.get(
        str(mapping.get("follower_fingerprint", ""))
    )
    if leader is not None and follower is not None and leader.get("port") != follower.get("port"):
        _remember_pair(leader, follower)
        logger.info(
            "Auto-detected from saved mapping: leader=%s follower=%s",
            leader.get("port"),
            follower.get("port"),
        )
        return str(leader["port"]), str(follower["port"])

    # Optional interactive first-time setup when no complete mapping exists.
    interactive_pref = os.getenv("RFX_SO101_INTERACTIVE_MAP", "1").strip().lower()
    if interactive_pref not in {"0", "false", "no", "off"}:
        interactive_pair = _interactive_pair_setup()
        if interactive_pair is not None:
            leader, follower = interactive_pair
            _remember_pair(leader, follower)
            return str(leader["port"]), str(follower["port"])

    # Partial match: keep whichever side was recognized and assign the other
    # role from remaining ports using stable sort by current device path.
    chosen: list[dict[str, Any]] = sorted(all_ports, key=lambda p: str(p.get("port")))
    if leader is not None:
        remaining = [p for p in chosen if p.get("port") != leader.get("port")]
        if not remaining:
            raise RuntimeError("Could not determine follower arm port from connected devices.")
        follower = remaining[0]
    elif follower is not None:
        remaining = [p for p in chosen if p.get("port") != follower.get("port")]
        if not remaining:
            raise RuntimeError("Could not determine leader arm port from connected devices.")
        leader = remaining[0]
    else:
        leader, follower = chosen[0], chosen[1]

    _remember_pair(leader, follower)
    logger.info(
        "Auto-detected leader=%s, follower=%s (mapping saved at %s)",
        leader.get("port"),
        follower.get("port"),
        _port_map_path(),
    )
    return str(leader["port"]), str(follower["port"])


class So101Backend:
    """SO-101 hardware backend using the universal Zenoh/transport pipeline.

    Internally creates a Rust ``RobotNode`` that owns the serial connection,
    publishes ``RobotState`` on ``rfx/{name}/state``, and listens for
    ``Command`` messages on ``rfx/{name}/cmd``.

    ``observe()`` reads state from the transport subscription.
    ``act()`` sends commands through the transport.
    """

    def __init__(
        self,
        config: RobotConfig,
        port: str = "/dev/ttyACM0",
        baudrate: int = 1_000_000,
        is_leader: bool = False,
        transport: Any | None = None,
        name: str | None = None,
        **kwargs: Any,
    ):
        self.config = config
        self.port = port
        self.is_leader = is_leader
        init_otel(service_name="rfx-so101")
        self._tracer = get_tracer("rfx.real.so101")
        self._state_miss_count = 0

        try:
            from rfx._rfx import RobotNode as _RustRobotNode
        except ImportError as err:
            raise ImportError(
                "rfx Rust bindings not available. Build with: maturin develop"
            ) from err

        # Shared or auto-created transport
        if transport is None:
            from .. import node as _node_mod

            transport = _node_mod.auto_transport()
        self._transport = transport

        if name is None:
            name = f"so101-{'leader' if is_leader else 'follower'}"

        # RobotNode: connects hardware, publishes state, accepts commands
        self._node = _RustRobotNode.so101(name, port, transport, is_leader, baudrate, 50.0)
        with self._tracer.start_as_current_span("so101.node_init") as span:
            span.set_attribute("name", str(name))
            span.set_attribute("port", str(port))
            span.set_attribute("is_leader", bool(is_leader))

        # Subscribe to this node's state for observe() / read_positions()
        self._state_sub = self._node.subscribe_state()

        # Camera support
        self._camera = None
        camera_id = kwargs.get("camera_id")
        if camera_id is not None:
            from .camera import Camera

            self._camera = Camera(device_id=camera_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_latest(self) -> Any | None:
        """Drain the subscription queue and return the most recent envelope."""
        latest = None
        while True:
            env = self._state_sub.try_recv()
            if env is None:
                break
            latest = env
        if latest is None:
            # Wait briefly for the first message
            latest = self._state_sub.recv_timeout(0.1)
        if latest is None:
            self._state_miss_count += 1
            if self._state_miss_count % 10 == 0:
                with self._tracer.start_as_current_span("so101.state_miss") as span:
                    span.set_attribute("port", str(self.port))
                    span.set_attribute("is_leader", bool(self.is_leader))
                    span.set_attribute("miss_count", int(self._state_miss_count))
        return latest

    def _parse_state(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Deserialize a transport envelope into (positions, velocities)."""
        state = json.loads(bytes(env.payload))
        positions = torch.tensor(state["joint_positions"], dtype=torch.float32)
        velocities = torch.tensor(state["joint_velocities"], dtype=torch.float32)
        return positions, velocities

    # ------------------------------------------------------------------
    # Public API (same interface as before)
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._node.is_running

    def observe(self) -> dict[str, torch.Tensor]:
        env = self._drain_latest()
        if env is not None:
            positions, velocities = self._parse_state(env)
        else:
            positions = torch.zeros(6, dtype=torch.float32)
            velocities = torch.zeros(6, dtype=torch.float32)

        raw_state = torch.cat([positions, velocities]).unsqueeze(0)

        obs = make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

        if self._camera is not None:
            image = self._camera.capture()
            obs["images"] = image.unsqueeze(0).unsqueeze(0)

        return obs

    def act(self, action: torch.Tensor) -> None:
        if self.is_leader:
            return
        action_6dof = action[0, : self.config.action_dim].cpu().tolist()
        with self._tracer.start_as_current_span("so101.send_command") as span:
            span.set_attribute("port", str(self.port))
            span.set_attribute("num_joints", len(action_6dof))
        self._node.send_command(action_6dof)

    def reset(self) -> dict[str, torch.Tensor]:
        # Send home position command through transport
        if not self.is_leader:
            home = [0.0] * 6
            self._node.send_command(home)
        return self.observe()

    def go_home(self) -> None:
        if not self.is_leader:
            home = [0.0] * 6
            self._node.send_command(home)

    def disconnect(self) -> None:
        if self._camera is not None:
            self._camera.release()
        self._node.stop()

    def read_positions(self) -> torch.Tensor:
        """Read current joint positions from the transport."""
        env = self._drain_latest()
        if env is None:
            return torch.zeros(6, dtype=torch.float32)
        positions, _ = self._parse_state(env)
        return positions

    @property
    def node(self) -> Any:
        """Access the underlying RobotNode."""
        return self._node

    @property
    def transport(self) -> Any:
        """Access the transport backend."""
        return self._transport


class So101Robot:
    """Convenience class for SO-101 robot.

    Auto-discovers the serial port if not provided.

        >>> robot = So101Robot()          # auto-discover
        >>> robot = So101Robot("/dev/ttyACM0")  # explicit
    """

    def __new__(cls, port: str | None = None, **kwargs):
        if port is None:
            port = _auto_port()
        from .base import RealRobot

        return RealRobot(config=SO101_CONFIG, robot_type="so101", port=port, **kwargs)


class So101LeaderFollower:
    """Leader-follower teleoperation over a shared Zenoh transport.

    Auto-discovers both serial ports if not provided.  Expects exactly
    two Feetech USB adapters plugged in.

        >>> teleop = So101LeaderFollower()   # auto-discover both
        >>> teleop.run()                     # Ctrl+C to stop

    Both arms are RobotNodes on the same transport.  The leader publishes
    state; the Python loop reads it and forwards as a command to the follower.
    """

    def __init__(
        self,
        leader_port: str | None = None,
        follower_port: str | None = None,
        transport: Any | None = None,
        **kwargs: Any,
    ):
        if leader_port is None or follower_port is None:
            leader_port, follower_port = _auto_pair()

        # Shared transport for both nodes
        if transport is None:
            from .. import node as _node_mod

            transport = _node_mod.auto_transport()
        self._transport = transport

        self._leader = So101Backend(
            config=SO101_CONFIG,
            port=leader_port,
            is_leader=True,
            transport=transport,
            name="so101-leader",
        )
        self._follower = So101Backend(
            config=SO101_CONFIG,
            port=follower_port,
            is_leader=False,
            transport=transport,
            name="so101-follower",
        )

    @property
    def leader(self) -> So101Backend:
        return self._leader

    @property
    def follower(self) -> So101Backend:
        return self._follower

    @property
    def transport(self) -> Any:
        return self._transport

    def step(self) -> torch.Tensor:
        """Read leader positions and mirror to follower via transport."""
        positions = self._leader.read_positions()
        self._follower.node.send_command(positions.tolist())
        return positions

    def run(self, callback=None):
        import time

        print("Starting teleoperation. Press Ctrl+C to stop.")
        try:
            while True:
                positions = self.step()
                if callback:
                    callback(positions)
                time.sleep(0.02)
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            self.disconnect()

    def disconnect(self):
        self._leader.disconnect()
        self._follower.disconnect()
