"""
rfx.node - Universal robot node with Zenoh transport.

Every robot in rfx is a node on the Zenoh bus. This module provides:
- create(): creates a RobotNode wrapping any hardware driver
- auto_transport(): creates the Zenoh transport backend
- discover_ports(): scans serial ports for known robot hardware
- discover_nodes(): lists live robot nodes on the network

One pipeline for all robots. Hardware I/O goes through RobotNode → Zenoh.

Example:
    >>> import rfx
    >>> node = rfx.node.create("so101", port="/dev/ttyACM0")
    >>> # State is now publishing on rfx/so101/state
    >>> sub = node.subscribe_state()
    >>> env = sub.recv_timeout(1.0)
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Topic key conventions
TOPIC_PREFIX = "rfx"


def state_key(name: str) -> str:
    """State topic key for a robot node."""
    return f"{TOPIC_PREFIX}/{name}/state"


def cmd_key(name: str) -> str:
    """Command topic key for a robot node."""
    return f"{TOPIC_PREFIX}/{name}/cmd"


def auto_transport(
    *,
    connect: tuple[str, ...] | list[str] = (),
    listen: tuple[str, ...] | list[str] = (),
    shared_memory: bool = True,
    key_prefix: str = "",
):
    """Create the Zenoh transport backend.

    Zenoh is the only supported distributed transport in rfx. If Zenoh is not
    compiled in or the Rust bindings are unavailable, this raises RuntimeError
    with a clear message about how to fix the build.

    Environment variables (used when ``connect``/``listen`` are empty):
      - ``RFX_ZENOH_CONNECT``: comma-separated Zenoh endpoints to connect to
        (e.g. ``tcp/192.168.1.100:7447,tcp/10.0.0.1:7447``)
      - ``RFX_ZENOH_LISTEN``: comma-separated Zenoh endpoints to listen on
      - ``RFX_ZENOH_SHARED_MEMORY``: ``0`` to disable shared-memory transport

    For in-process testing only, use ``rfx.teleop.transport.InprocTransport``
    directly.
    """
    try:
        from rfx._rfx import Transport as _RustTransport
    except ImportError as err:
        raise RuntimeError(
            "rfx Rust bindings are not installed. "
            "Build with: maturin develop --features 'extension-module,zenoh'"
        ) from err

    if not (hasattr(_RustTransport, "zenoh_available") and _RustTransport.zenoh_available()):
        raise RuntimeError(
            "Zenoh transport is not compiled into the rfx native extension. "
            "Rebuild with: maturin develop --features 'extension-module,zenoh'"
        )

    # Resolve endpoints: explicit args take priority over env vars.
    resolved_connect = list(connect)
    if not resolved_connect:
        env_connect = os.environ.get("RFX_ZENOH_CONNECT", "")
        if env_connect.strip():
            resolved_connect = [e.strip() for e in env_connect.split(",") if e.strip()]

    resolved_listen = list(listen)
    if not resolved_listen:
        env_listen = os.environ.get("RFX_ZENOH_LISTEN", "")
        if env_listen.strip():
            resolved_listen = [e.strip() for e in env_listen.split(",") if e.strip()]

    # Shared memory can be disabled via env var.
    env_shm = os.environ.get("RFX_ZENOH_SHARED_MEMORY")
    if env_shm is not None:
        shared_memory = env_shm not in ("0", "false", "no", "off")

    if resolved_connect:
        logger.info("Zenoh connecting to: %s", resolved_connect)
    if resolved_listen:
        logger.info("Zenoh listening on: %s", resolved_listen)

    return _RustTransport.zenoh(resolved_connect, resolved_listen, shared_memory, key_prefix)


def discover_ports() -> list[dict[str, Any]]:
    """Scan serial ports for known robot hardware.

    Returns a list of dicts with keys:
    - port: serial port path (e.g. /dev/ttyACM0)
    - description: device description
    - robot_type: detected type ("so101", "unknown")
    - hwid: hardware ID string
    """
    try:
        import serial.tools.list_ports
    except ImportError:
        logger.warning("pyserial not installed — cannot scan serial ports")
        return []

    results = []
    for port_info in serial.tools.list_ports.comports():
        robot_type = "unknown"
        desc = (port_info.description or "").lower()
        hwid = (port_info.hwid or "").lower()

        # Feetech STS3215 USB-serial adapters (CH340, CP210x, FTDI)
        if any(tag in desc for tag in ("ch340", "cp210", "ft232", "usb serial", "usb-serial")):
            robot_type = "so101"
        elif any(tag in hwid for tag in ("1a86:", "10c4:", "0403:")):
            # CH340 (1a86), CP210x (10c4), FTDI (0403) vendor IDs
            robot_type = "so101"

        results.append(
            {
                "port": port_info.device,
                "description": port_info.description,
                "robot_type": robot_type,
                "hwid": port_info.hwid,
                "serial_number": getattr(port_info, "serial_number", None),
                "location": getattr(port_info, "location", None),
                "manufacturer": getattr(port_info, "manufacturer", None),
            }
        )

    return results


def discover_so101_ports() -> list[str]:
    """Return serial port paths that look like SO-101 arms."""
    return [p["port"] for p in discover_ports() if p["robot_type"] == "so101"]


def discover_nodes(timeout_s: float = 2.0) -> list[str]:
    """List live robot nodes on the Zenoh network.

    Uses liveliness queries to find nodes that have declared
    ``rfx/alive/node/{name}`` tokens.

    .. note::
        Not yet implemented. Returns an empty list. Will be wired when
        Zenoh liveliness queries are exposed through the transport layer.

    Raises:
        RuntimeError: If Zenoh transport is unavailable.
    """
    try:
        from rfx._rfx import Transport as _RustTransport
    except ImportError as err:
        raise RuntimeError(
            "rfx Rust bindings required for node discovery. "
            "Build with: maturin develop --features 'extension-module,zenoh'"
        ) from err

    if not (hasattr(_RustTransport, "zenoh_available") and _RustTransport.zenoh_available()):
        raise RuntimeError(
            "Zenoh transport required for node discovery. "
            "Rebuild with: maturin develop --features 'extension-module,zenoh'"
        )

    logger.warning("discover_nodes() not yet implemented — returning empty list")
    return []


def create(
    robot_type: str,
    *,
    name: str | None = None,
    port: str | None = None,
    ip_address: str | None = None,
    is_leader: bool = False,
    edu_mode: bool = False,
    transport: Any | None = None,
    publish_rate_hz: float = 50.0,
    **kwargs: Any,
) -> Any:
    """Create a robot node on the Zenoh bus.

    This is THE entry point for connecting a robot. It:
    1. Connects to hardware (serial, network, etc.)
    2. Wraps it in a RobotNode
    3. Publishes state to Zenoh at publish_rate_hz
    4. Accepts commands via Zenoh

    Args:
        robot_type: "so101", "go2", etc.
        name: Node name (default: robot_type)
        port: Serial port (for SO-101)
        ip_address: Robot IP (for Go2)
        is_leader: Leader mode (torque disabled, for teleop input)
        edu_mode: Enable low-level motor control (Go2)
        transport: Transport backend (auto-created if None)
        publish_rate_hz: State publish rate
        **kwargs: Forwarded to the hardware driver

    Returns:
        RobotNode instance

    Example:
        >>> node = rfx.node.create("so101", port="/dev/ttyACM0")
        >>> node = rfx.node.create("so101", port="auto")  # auto-detect port
        >>> node = rfx.node.create("go2")                  # Go2 over Zenoh
    """
    if publish_rate_hz <= 0:
        raise ValueError(f"publish_rate_hz must be > 0, got {publish_rate_hz}")

    # Validate robot type before any transport initialization so callers get
    # deterministic argument errors even when transport infra is unavailable.
    supported_types = {"so101", "go2"}
    if robot_type not in supported_types:
        raise ValueError(f"Unsupported robot_type: {robot_type!r}. Supported: 'so101', 'go2'")

    try:
        from rfx._rfx import RobotNode as _RustRobotNode
    except ImportError as err:
        raise ImportError(
            "rfx Rust bindings required. Build with: maturin develop --features 'extension-module,zenoh'"
        ) from err

    if name is None:
        name = robot_type

    if transport is None:
        transport = auto_transport()

    if robot_type == "so101":
        if port == "auto" or port is None:
            ports = discover_so101_ports()
            if not ports:
                raise RuntimeError(
                    "No SO-101 arms detected. Plug in USB and try again, "
                    "or specify port= explicitly."
                )
            port = ports[0]
            logger.info("Auto-detected SO-101 on %s", port)

        baudrate = kwargs.pop("baudrate", 1_000_000)
        return _RustRobotNode.so101(
            name,
            port,
            transport,
            is_leader,
            baudrate,
            publish_rate_hz,
        )

    elif robot_type == "go2":
        ip = ip_address or kwargs.pop("ip_address", "192.168.123.161")
        return _RustRobotNode.go2(
            name,
            ip,
            transport,
            edu_mode,
            publish_rate_hz,
        )

    raise RuntimeError(f"Internal error: unexpected robot_type {robot_type!r}")
