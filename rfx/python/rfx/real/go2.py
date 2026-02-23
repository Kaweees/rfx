"""
rfx.real.go2 - Unitree Go2 hardware backend via Zenoh transport pipeline.

When using the Rust backend (default), a RobotNode wraps the hardware and
all state/command traffic flows through the Zenoh transport.
Legacy backends (unitree_sdk2py, subprocess) are kept for backwards compat.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import warnings
from typing import TYPE_CHECKING, Any

import torch

from ..observation import make_observation
from ..robot.config import GO2_CONFIG, RobotConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Go2Backend:
    """Unitree Go2 hardware backend.

    Uses a Rust RobotNode that communicates with the Go2 hardware and
    publishes state/commands over the Zenoh transport pipeline.
    """

    _channel_initialized = False
    _system_python = "/usr/bin/python3"

    def __init__(
        self,
        config: RobotConfig,
        ip_address: str = "192.168.123.161",
        edu_mode: bool = False,
        hw_backend: str | None = None,
        zenoh_endpoint: str | None = None,
        transport: Any | None = None,
        name: str | None = None,
        # Deprecated alias — prefer hw_backend
        dds_backend: str | None = None,
        **kwargs,
    ):
        self.config = config
        self.ip_address = ip_address
        self.edu_mode = edu_mode
        self._backend_mode = "rust"
        self._robot = None
        self._node = None
        self._state_sub = None
        self._sport_client = None
        self._unitree_state_sub = None
        self._latest_lowstate = None
        self._state_lock = threading.Lock()

        # Deprecation warning for old parameter name
        if dds_backend is not None:
            warnings.warn(
                "dds_backend is deprecated, use hw_backend instead",
                FutureWarning,
                stacklevel=2,
            )

        # Resolve hw backend: explicit param > deprecated alias > env var > auto
        _backend_arg = hw_backend or dds_backend
        backend_pref = (_backend_arg or os.getenv("RFX_GO2_BACKEND", "auto")).strip().lower()
        if backend_pref not in {
            "auto",
            "rust",
            "unitree",
            "unitree_sdk2py",
            "zenoh",
            "dust",
            "cyclone",
        }:
            backend_pref = "auto"

        if not self.edu_mode and backend_pref in {"auto", "unitree", "unitree_sdk2py"}:
            if self._init_unitree_sdk_backend():
                self._backend_mode = "unitree_sdk2py"
                logger.warning(
                    "Using legacy unitree_sdk2py backend (not Zenoh). "
                    "Set hw_backend='rust' or RFX_GO2_BACKEND=rust to force Zenoh."
                )
                return
            if self._init_unitree_subprocess_backend():
                self._backend_mode = "unitree_subprocess"
                logger.warning(
                    "Using legacy unitree subprocess backend (not Zenoh). "
                    "Set hw_backend='rust' or RFX_GO2_BACKEND=rust to force Zenoh."
                )
                return

        # ---- Rust backend via RobotNode pipeline ----
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
            name = "go2"

        # Build RobotNode for Go2 (publishes state/commands via Zenoh transport)
        self._node = _RustRobotNode.go2(name, ip_address, transport, edu_mode, 50.0)
        self._state_sub = self._node.subscribe_state()
        self._backend_mode = "rust"

    # ------------------------------------------------------------------
    # Legacy unitree_sdk2py init (kept for backwards compat)
    # ------------------------------------------------------------------

    def _init_unitree_sdk_backend(self) -> bool:
        pet_go_path = "/unitree/module/pet_go"
        if pet_go_path not in sys.path:
            sys.path.insert(0, pet_go_path)

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
        except Exception:
            return False

        try:
            if not Go2Backend._channel_initialized:
                ChannelFactoryInitialize(0)
                Go2Backend._channel_initialized = True

            client = SportClient()
            client.SetTimeout(5.0)
            client.Init()

            def _on_state(msg):
                with self._state_lock:
                    self._latest_lowstate = msg

            sub = ChannelSubscriber("rt/lowstate", LowState_)
            sub.Init(_on_state, 10)

            self._sport_client = client
            self._unitree_state_sub = sub
            return True
        except Exception:
            return False

    def _check_rc(self, rc: int, command: str) -> None:
        if rc != 0:
            raise RuntimeError(f"Go2 command '{command}' failed with code {rc}")

    def _init_unitree_subprocess_backend(self) -> bool:
        check = (
            "import sys; "
            'sys.path.insert(0, "/unitree/module/pet_go"); '
            "from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelFactory; "
            "from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_"
        )
        try:
            p = subprocess.run(
                [self._system_python, "-c", check],
                capture_output=True,
                text=True,
                timeout=3.0,
                check=False,
            )
            return p.returncode == 0
        except Exception:
            return False

    def _run_unitree_cmd(self, command: str, *args: float) -> int:
        api_id_map = {
            "Move": 1008,
            "Sit": 1009,
            "RiseSit": 1010,
            "RecoveryStand": 1006,
            "StopMove": 1003,
            "GetServerApiVersion": 0,
        }
        if command not in api_id_map:
            raise RuntimeError(f"Unsupported command for unitree subprocess backend: {command}")

        if command == "GetServerApiVersion":
            py = (
                "import sys; "
                'sys.path.insert(0, "/unitree/module/pet_go"); '
                "from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelFactory; "
                "from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_; "
                "ChannelFactoryInitialize(0); "
                'f=ChannelFactory(); ch=f.CreateSendChannel("rt/api/sport/request", Request_); '
                "print(0)"
            )
        else:
            api_id = api_id_map[command]
            if command == "Move":
                parameter = f'{{"x":{args[0]},"y":{args[1]},"z":{args[2]}}}'
                noreply = True
            else:
                parameter = "{}"
                noreply = False

            py = (
                "import sys,time; "
                'sys.path.insert(0, "/unitree/module/pet_go"); '
                "from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelFactory; "
                "from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_, RequestHeader_, RequestIdentity_, RequestLease_, RequestPolicy_; "
                "ChannelFactoryInitialize(0); "
                'f=ChannelFactory(); ch=f.CreateSendChannel("rt/api/sport/request", Request_); '
                f"req=Request_(RequestHeader_(RequestIdentity_(time.monotonic_ns(), {api_id}), RequestLease_(0), RequestPolicy_(0, {str(noreply)})), {parameter!r}, []); "
                "ok=ch.Write(req, 1.0); "
                "print(0 if ok else 1)"
            )

        p = subprocess.run(
            [self._system_python, "-c", py],
            capture_output=True,
            text=True,
            timeout=6.0,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"unitree subprocess command failed (rc={p.returncode}): {p.stderr.strip()}"
            )
        out = p.stdout.strip().splitlines()
        if not out:
            return -1
        return int(out[-1])

    # ------------------------------------------------------------------
    # Internal helpers for Rust/transport path
    # ------------------------------------------------------------------

    def _drain_latest(self) -> Any | None:
        """Drain subscription queue and return most recent envelope."""
        if self._state_sub is None:
            return None
        latest = None
        while True:
            env = self._state_sub.try_recv()
            if env is None:
                break
            latest = env
        if latest is None:
            latest = self._state_sub.recv_timeout(0.1)
        return latest

    def _parse_state(self, env: Any) -> dict:
        """Deserialize transport envelope payload."""
        return json.loads(bytes(env.payload))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        if self._backend_mode == "unitree_sdk2py":
            if self._sport_client is None:
                return False
            code, _ = self._sport_client.GetServerApiVersion()
            return code == 0
        if self._backend_mode == "unitree_subprocess":
            return self._run_unitree_cmd("GetServerApiVersion") == 0
        return self._node is not None and self._node.is_running

    def observe(self) -> dict[str, torch.Tensor]:
        if self._backend_mode in {"unitree_sdk2py", "unitree_subprocess"}:
            return self._observe_unitree()

        # Rust transport path
        env = self._drain_latest()
        if env is not None:
            state = self._parse_state(env)
            positions = torch.tensor(state["joint_positions"], dtype=torch.float32)
            velocities = torch.tensor(state["joint_velocities"], dtype=torch.float32)
        else:
            positions = torch.zeros(12, dtype=torch.float32)
            velocities = torch.zeros(12, dtype=torch.float32)

        # Go2 state also has IMU data in the pose field — extract from state
        if env is not None:
            state = self._parse_state(env)
            # RobotState.pose has orientation info
            pose = state.get("pose", {})
            rot = pose.get("rotation", {})
            orientation = torch.tensor(
                [rot.get("w", 1.0), rot.get("x", 0.0), rot.get("y", 0.0), rot.get("z", 0.0)],
                dtype=torch.float32,
            )
            # Joint torques can serve as proxy for angular velocity (reserved for future use)
            _ = state.get("joint_torques", [])
            angular_vel = torch.zeros(3, dtype=torch.float32)
            linear_acc = torch.zeros(3, dtype=torch.float32)
        else:
            orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            angular_vel = torch.zeros(3, dtype=torch.float32)
            linear_acc = torch.zeros(3, dtype=torch.float32)

        raw_state = torch.cat(
            [positions, velocities, orientation, angular_vel, linear_acc]
        ).unsqueeze(0)

        return make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

    def _observe_unitree(self) -> dict[str, torch.Tensor]:
        """Observe via legacy unitree_sdk2py path."""
        with self._state_lock:
            low_state = self._latest_lowstate

        if low_state is None:
            positions = torch.zeros(12, dtype=torch.float32)
            velocities = torch.zeros(12, dtype=torch.float32)
            orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            angular_vel = torch.zeros(3, dtype=torch.float32)
            linear_acc = torch.zeros(3, dtype=torch.float32)
        else:
            positions = torch.tensor([m.q for m in low_state.motor_state[:12]], dtype=torch.float32)
            velocities = torch.tensor(
                [m.dq for m in low_state.motor_state[:12]], dtype=torch.float32
            )
            imu = low_state.imu_state
            orientation = torch.tensor(list(imu.quaternion), dtype=torch.float32)
            angular_vel = torch.tensor(list(imu.gyroscope), dtype=torch.float32)
            linear_acc = torch.tensor(list(imu.accelerometer), dtype=torch.float32)

        raw_state = torch.cat(
            [positions, velocities, orientation, angular_vel, linear_acc]
        ).unsqueeze(0)

        return make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

    def act(self, action: torch.Tensor) -> None:
        if self._backend_mode == "unitree_sdk2py":
            if self.edu_mode:
                raise RuntimeError("EDU mode is not supported with unitree_sdk2py backend")
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._check_rc(self._sport_client.Move(vx, vy, vyaw), "Move")
            return
        if self._backend_mode == "unitree_subprocess":
            if self.edu_mode:
                raise RuntimeError("EDU mode is not supported with unitree subprocess backend")
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._check_rc(self._run_unitree_cmd("Move", vx, vy, vyaw), "Move")
            return

        # Rust transport path — send command via node
        if not self.edu_mode:
            # Sport-mode velocity command as 3-element position command
            vx = action[0, 0].item()
            vy = action[0, 1].item()
            vyaw = action[0, 2].item()
            self._node.send_command([vx, vy, vyaw])
        else:
            action_12dof = action[0, :12].cpu().tolist()
            self._node.send_command(action_12dof)

    def reset(self) -> dict[str, torch.Tensor]:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.RecoveryStand(), "RecoveryStand")
            return self.observe()
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("RecoveryStand"), "RecoveryStand")
            return self.observe()
        # Rust transport path — send stand-up command
        self._node.send_command([0.0] * 12)
        return self.observe()

    def go_home(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.RecoveryStand(), "RecoveryStand")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("RecoveryStand"), "RecoveryStand")
            return
        self._node.send_command([0.0] * 12)

    def disconnect(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            if self._sport_client is not None:
                try:
                    self._sport_client.StopMove()
                except Exception:
                    pass
            if self._unitree_state_sub is not None:
                try:
                    self._unitree_state_sub.Close()
                except Exception:
                    pass
            return
        if self._backend_mode == "unitree_subprocess":
            try:
                self._run_unitree_cmd("StopMove")
            except Exception:
                pass
            return
        if self._node is not None:
            self._node.stop()

    def stand(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.RecoveryStand(), "RecoveryStand")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("RecoveryStand"), "RecoveryStand")
            return
        self._node.send_command([0.0] * 12)

    def sit(self) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.Sit(), "Sit")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("Sit"), "Sit")
            return
        # No direct "sit" via generic Command — approximate with zero torques
        self._node.send_command([0.0] * 12)

    def walk(self, vx: float, vy: float, vyaw: float) -> None:
        if self._backend_mode == "unitree_sdk2py":
            self._check_rc(self._sport_client.Move(vx, vy, vyaw), "Move")
            return
        if self._backend_mode == "unitree_subprocess":
            self._check_rc(self._run_unitree_cmd("Move", vx, vy, vyaw), "Move")
            return
        self._node.send_command([vx, vy, vyaw])

    @property
    def node(self) -> Any:
        """Access the underlying RobotNode (Rust transport path only)."""
        return self._node

    @property
    def transport(self) -> Any:
        """Access the transport backend."""
        return getattr(self, "_transport", None)


class Go2Robot:
    """Convenience class for Go2 robot.

    Args:
        ip_address: Robot IP address.
        hw_backend: Hardware communication backend ("rust" for Zenoh transport,
            "unitree" for legacy unitree_sdk2py, or None for auto).
            Can also be set via RFX_GO2_BACKEND env var.
        zenoh_endpoint: Zenoh router endpoint (e.g. "tcp/192.168.123.161:7447").
        edu_mode: Enable low-level motor control.
        **kwargs: Additional kwargs forwarded to RealRobot.

    Examples:
        >>> go2 = Go2Robot()                                    # auto-detect (Rust/Zenoh)
        >>> go2 = Go2Robot(zenoh_endpoint="tcp/10.0.0.1:7447")  # Zenoh with explicit router
        >>> go2 = Go2Robot(edu_mode=True)                       # low-level motor control
    """

    def __new__(
        cls,
        ip_address: str = "192.168.123.161",
        hw_backend: str | None = None,
        zenoh_endpoint: str | None = None,
        edu_mode: bool = False,
        **kwargs,
    ):
        from .base import RealRobot

        return RealRobot(
            config=GO2_CONFIG,
            robot_type="go2",
            ip_address=ip_address,
            hw_backend=hw_backend,
            zenoh_endpoint=zenoh_endpoint,
            edu_mode=edu_mode,
            **kwargs,
        )
