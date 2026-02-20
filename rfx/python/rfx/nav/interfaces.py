"""
rfx.nav.interfaces - Navigation protocol definitions.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Planner(Protocol):
    """Path planning protocol."""

    def plan(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        costmap: Costmap | None = None,
    ) -> list[tuple[float, float]]:
        """Return a list of waypoints from start to goal."""
        ...


@runtime_checkable
class Controller(Protocol):
    """Local trajectory-tracking controller protocol."""

    def compute_velocity(
        self,
        current_pose: tuple[float, float, float],  # (x, y, yaw)
        target: tuple[float, float],
    ) -> tuple[float, float]:
        """Return (linear_vel, angular_vel) to reach target."""
        ...


@runtime_checkable
class Costmap(Protocol):
    """2D costmap protocol for obstacle avoidance."""

    def cost_at(self, x: float, y: float) -> float:
        """Return cost [0.0=free, 1.0=obstacle] at world coordinates."""
        ...

    @property
    def width(self) -> float: ...

    @property
    def height(self) -> float: ...

    @property
    def resolution(self) -> float: ...
