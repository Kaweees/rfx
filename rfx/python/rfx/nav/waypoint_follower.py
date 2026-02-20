"""
rfx.nav.waypoint_follower - Minimal navigation implementations.
"""

from __future__ import annotations

import math
from typing import Any


class ProportionalController:
    """Simple proportional controller for waypoint tracking."""

    def __init__(self, linear_gain: float = 1.0, angular_gain: float = 2.0) -> None:
        self.linear_gain = linear_gain
        self.angular_gain = angular_gain

    def compute_velocity(
        self,
        current_pose: tuple[float, float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        x, y, yaw = current_pose
        tx, ty = target

        dx = tx - x
        dy = ty - y
        distance = math.sqrt(dx * dx + dy * dy)

        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - yaw
        # Normalize to [-pi, pi]
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        linear_vel = self.linear_gain * distance
        angular_vel = self.angular_gain * yaw_error

        return (linear_vel, angular_vel)


class SimpleWaypointFollower:
    """Follow a sequence of waypoints using a controller."""

    def __init__(
        self,
        controller: Any | None = None,
        goal_tolerance: float = 0.1,
    ) -> None:
        self.controller = controller or ProportionalController()
        self.goal_tolerance = goal_tolerance
        self._waypoints: list[tuple[float, float]] = []
        self._current_idx = 0

    def set_waypoints(self, waypoints: list[tuple[float, float]]) -> None:
        self._waypoints = list(waypoints)
        self._current_idx = 0

    @property
    def is_complete(self) -> bool:
        return self._current_idx >= len(self._waypoints)

    @property
    def current_target(self) -> tuple[float, float] | None:
        if self.is_complete:
            return None
        return self._waypoints[self._current_idx]

    def step(self, current_pose: tuple[float, float, float]) -> tuple[float, float]:
        """Compute velocity command for current step. Returns (0, 0) if complete."""
        if self.is_complete:
            return (0.0, 0.0)

        target = self._waypoints[self._current_idx]
        x, y, _ = current_pose
        dx = target[0] - x
        dy = target[1] - y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < self.goal_tolerance:
            self._current_idx += 1
            if self.is_complete:
                return (0.0, 0.0)
            target = self._waypoints[self._current_idx]

        return self.controller.compute_velocity(current_pose, target)
