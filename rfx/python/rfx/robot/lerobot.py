"""
rfx.robot.lerobot - Factory functions for creating robots.

Beginner-friendly entry point:

    >>> from rfx.robot import lerobot
    >>> arm = lerobot.so101()
    >>> go2 = lerobot.go2()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..real.base import RealRobot


def so101(
    config: Any = None,
    *,
    port: str | None = None,
    scale: int = 1,
    urdf: str | None = None,
    **kw: Any,
) -> RealRobot:
    """Create an SO-101 arm ready for teleop or inference.

    Args:
        config: RobotConfig, path to YAML, or None for defaults.
        port: Serial port (auto-detected if None).
        scale: Number of arms (1 for single, 2 for bimanual). Reserved for future use.
        urdf: Path to URDF file (overrides config).
        **kw: Forwarded to RealRobot constructor.

    Returns:
        A connected RealRobot instance.
    """
    from ..real.base import RealRobot
    from .config import SO101_CONFIG

    if config is None:
        config = SO101_CONFIG

    extra: dict[str, Any] = {}
    if port is not None:
        extra["port"] = port
    extra.update(kw)

    return RealRobot(config=config, robot_type="so101", **extra)


def go2(
    config: Any = None,
    *,
    ip_address: str | None = None,
    hw_backend: str | None = None,
    edu_mode: bool = False,
    **kw: Any,
) -> RealRobot:
    """Create a Go2 quadruped.

    Args:
        config: RobotConfig, path to YAML, or None for defaults.
        ip_address: Robot IP address (default: 192.168.123.161).
        hw_backend: Hardware backend ("rust" for Zenoh, "unitree" for legacy).
        edu_mode: Enable low-level motor control.
        **kw: Forwarded to RealRobot constructor.

    Returns:
        A connected RealRobot instance.
    """
    from ..real.base import RealRobot
    from .config import GO2_CONFIG

    if config is None:
        config = GO2_CONFIG

    extra: dict[str, Any] = {}
    if ip_address is not None:
        extra["ip_address"] = ip_address
    if hw_backend is not None:
        extra["hw_backend"] = hw_backend
    extra["edu_mode"] = edu_mode
    extra.update(kw)

    return RealRobot(config=config, robot_type="go2", **extra)


def g1(
    config: Any = None,
    *,
    ip_address: str | None = None,
    kp: list[float] | None = None,
    kd: list[float] | None = None,
    **kw: Any,
) -> RealRobot:
    """Create a G1 humanoid.

    Args:
        config: RobotConfig, path to YAML, or None for defaults.
        ip_address: Robot IP address (default: 192.168.123.161).
        kp: Per-joint proportional gains (29 values).
        kd: Per-joint derivative gains (29 values).
        **kw: Forwarded to RealRobot constructor.

    Returns:
        A connected RealRobot instance.
    """
    from ..real.base import RealRobot
    from .config import G1_CONFIG

    if config is None:
        config = G1_CONFIG

    extra: dict[str, Any] = {}
    if ip_address is not None:
        extra["ip_address"] = ip_address
    if kp is not None:
        extra["kp"] = kp
    if kd is not None:
        extra["kd"] = kd
    extra.update(kw)

    return RealRobot(config=config, robot_type="g1", **extra)


__all__ = ["so101", "go2", "g1"]
