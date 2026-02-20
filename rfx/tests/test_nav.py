"""Tests for rfx.nav - ProportionalController and SimpleWaypointFollower."""

from __future__ import annotations

import math

import pytest

from rfx.nav import ProportionalController, SimpleWaypointFollower


# ---------------------------------------------------------------------------
# ProportionalController
# ---------------------------------------------------------------------------


def test_controller_compute_velocity_toward_target() -> None:
    """Controller produces positive linear velocity toward a target ahead."""
    ctrl = ProportionalController(linear_gain=1.0, angular_gain=2.0)
    # Robot at origin facing +x, target at (5, 0)
    linear, angular = ctrl.compute_velocity((0.0, 0.0, 0.0), (5.0, 0.0))

    assert linear > 0.0, "Should move forward"
    assert abs(angular) < 1e-9, "No angular correction needed when aligned"


def test_controller_angular_correction() -> None:
    """Controller produces angular velocity when target is to the left."""
    ctrl = ProportionalController(linear_gain=1.0, angular_gain=2.0)
    # Robot at origin facing +x (yaw=0), target at (0, 5) -> 90 degrees left
    linear, angular = ctrl.compute_velocity((0.0, 0.0, 0.0), (0.0, 5.0))

    assert linear > 0.0
    assert angular > 0.0, "Should turn left (positive angular)"


def test_controller_behind_target_turns() -> None:
    """Target behind the robot produces a large angular correction."""
    ctrl = ProportionalController(linear_gain=1.0, angular_gain=2.0)
    # Robot at origin facing +x, target at (-5, 0) -> behind
    linear, angular = ctrl.compute_velocity((0.0, 0.0, 0.0), (-5.0, 0.0))

    assert linear > 0.0
    # Should want to turn pi radians
    assert abs(angular) > 3.0


def test_controller_at_target_zero_velocity() -> None:
    """When already at the target, velocity is zero."""
    ctrl = ProportionalController(linear_gain=1.0, angular_gain=2.0)
    linear, angular = ctrl.compute_velocity((3.0, 4.0, 0.0), (3.0, 4.0))

    assert linear == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SimpleWaypointFollower
# ---------------------------------------------------------------------------


def test_waypoint_follower_steps_through_waypoints() -> None:
    """Follower advances to the next waypoint when goal_tolerance is met."""
    follower = SimpleWaypointFollower(goal_tolerance=0.5)
    follower.set_waypoints([(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)])

    assert not follower.is_complete
    assert follower.current_target == (1.0, 0.0)

    # Step from very close to first waypoint -> should advance
    follower.step((0.9, 0.0, 0.0))
    # Within tolerance of (1.0, 0.0), so target advances
    assert follower.current_target == (2.0, 0.0)


def test_waypoint_follower_completion() -> None:
    """Follower reports complete after passing all waypoints."""
    follower = SimpleWaypointFollower(goal_tolerance=1.0)
    follower.set_waypoints([(0.0, 0.0)])

    assert not follower.is_complete
    vel = follower.step((0.0, 0.0, 0.0))

    assert follower.is_complete
    assert vel == (0.0, 0.0)


def test_waypoint_follower_returns_zero_when_complete() -> None:
    """step() returns (0, 0) after all waypoints are reached."""
    follower = SimpleWaypointFollower(goal_tolerance=1.0)
    follower.set_waypoints([(0.0, 0.0)])
    follower.step((0.0, 0.0, 0.0))  # completes

    vel = follower.step((0.0, 0.0, 0.0))
    assert vel == (0.0, 0.0)


def test_waypoint_follower_empty_waypoints() -> None:
    """Follower with no waypoints is immediately complete."""
    follower = SimpleWaypointFollower()
    follower.set_waypoints([])

    assert follower.is_complete
    assert follower.current_target is None
    assert follower.step((0.0, 0.0, 0.0)) == (0.0, 0.0)
