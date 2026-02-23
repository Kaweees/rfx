"""Pytest configuration and shared fixtures."""

from collections.abc import Callable

import pytest

# ---------------------------------------------------------------------------
# Environment isolation (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_tinygrad_cache(tmp_path, monkeypatch):
    """Redirect tinygrad cache to a per-test temp directory.

    Prevents sqlite3.OperationalError when the system-wide tinygrad cache DB
    is locked or readonly (e.g. concurrent CI runners, readonly filesystems).
    """
    cache_dir = tmp_path / "tinygrad_cache"
    cache_dir.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
    monkeypatch.setenv("CACHELEVEL", "0")


@pytest.fixture(autouse=True)
def _disable_zenoh_shm(monkeypatch):
    """Default Zenoh shared memory OFF in tests.

    SHM requires OS-level support (POSIX shm_open, /dev/shm) that is not
    always available in CI or sandboxed environments.  Individual tests that
    specifically need SHM can override this by setting the env var back.
    """
    monkeypatch.setenv("RFX_ZENOH_SHARED_MEMORY", "0")


# ---------------------------------------------------------------------------
# Shared skill/robot fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_skill_func() -> Callable[..., str]:
    """A sample function to use as a skill."""

    def walk_forward(distance: float = 1.0) -> str:
        """Walk forward by the specified distance in meters."""
        return f"Walked {distance}m"

    return walk_forward


@pytest.fixture
def sample_skill_with_types() -> Callable[..., dict]:
    """A sample function with full type annotations."""

    def move_robot(
        x: float,
        y: float,
        speed: float = 0.5,
        blocking: bool = True,
    ) -> dict:
        """
        Move the robot to a position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            speed: Movement speed in m/s
            blocking: Whether to wait for completion

        Returns:
            Movement result dictionary
        """
        return {"x": x, "y": y, "speed": speed, "blocking": blocking}

    return move_robot


@pytest.fixture
def mock_robot_state() -> dict:
    """Mock robot state for testing."""
    return {
        "tick": 12345,
        "timestamp": 1.234,
        "imu": {
            "roll": 0.01,
            "pitch": 0.02,
            "yaw": 0.0,
            "gyroscope": [0.0, 0.0, 0.0],
            "accelerometer": [0.0, 0.0, 9.81],
        },
        "joint_positions": [0.0] * 12,
        "joint_velocities": [0.0] * 12,
        "foot_contact": [True, True, True, True],
    }
