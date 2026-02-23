"""Backward compatibility — urdf moved to rfx.robot.urdf."""

from .robot.urdf import *  # noqa: F401, F403
from .robot.urdf import URDF  # noqa: F401

__all__ = ["URDF"]
