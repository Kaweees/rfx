"""Backward compatibility — config moved to rfx.robot.config."""

from .robot.config import (  # noqa: F401, I001
    CameraConfig,
    G1_CONFIG,
    GO2_CONFIG,
    JointConfig,
    RobotConfig,
    SO101_CONFIG,
    load_config,
)

__all__ = [
    "CameraConfig",
    "JointConfig",
    "RobotConfig",
    "SO101_CONFIG",
    "GO2_CONFIG",
    "G1_CONFIG",
    "load_config",
]
