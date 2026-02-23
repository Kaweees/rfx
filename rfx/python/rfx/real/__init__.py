"""
rfx.real - Real hardware robot backends

Same interface as simulation - no ROS, no middleware.
"""

from .base import RealRobot
from .g1 import G1Robot
from .go2 import Go2Robot
from .so101 import So101Robot

try:
    from .camera import Camera, RealSenseCamera
except ModuleNotFoundError:
    # Keep non-camera backends importable when optional camera deps (e.g. torch) are absent.
    Camera = None
    RealSenseCamera = None

__all__ = ["RealRobot", "So101Robot", "Go2Robot", "G1Robot", "Camera", "RealSenseCamera"]
