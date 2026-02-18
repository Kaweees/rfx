"""Tests for rfx.Session inference runtime."""

import importlib.util
import time

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")

if TORCH_AVAILABLE:
    import torch

    from rfx import MockRobot, Session, SessionStats, run


def _zero_policy(obs):
    """Policy that returns zeros matching the observation state shape."""
    return torch.zeros_like(obs["state"])


class TestSessionRunsSteps:
    def test_session_runs_steps(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        with Session(robot, _zero_policy, rate_hz=100, warmup_s=0.0) as s:
            time.sleep(0.5)
            assert s.step_count > 0
            assert s.is_running


class TestSessionContextManager:
    def test_start_stop_lifecycle(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        session = Session(robot, _zero_policy, rate_hz=100, warmup_s=0.0)
        assert not session.is_running

        with session as s:
            assert s.is_running
            time.sleep(0.1)

        assert not session.is_running


class TestSessionStats:
    def test_stats_populated(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        with Session(robot, _zero_policy, rate_hz=100, warmup_s=0.0) as s:
            time.sleep(0.3)
            stats = s.stats

        assert isinstance(stats, SessionStats)
        assert stats.iterations > 0
        assert stats.target_period_s == pytest.approx(0.01, abs=1e-6)
        assert stats.avg_period_s > 0
        assert stats.p50_jitter_s >= 0
        assert stats.p95_jitter_s >= 0
        assert stats.p99_jitter_s >= 0
        assert stats.max_jitter_s >= 0

        d = stats.to_dict()
        assert set(d.keys()) == {
            "iterations",
            "overruns",
            "target_period_s",
            "avg_period_s",
            "p50_jitter_s",
            "p95_jitter_s",
            "p99_jitter_s",
            "max_jitter_s",
        }


class TestSessionErrorPropagation:
    def test_policy_error_propagates(self):
        robot = MockRobot(state_dim=12, action_dim=6)

        def bad_policy(obs):
            raise ValueError("policy exploded")

        session = Session(robot, bad_policy, rate_hz=100, warmup_s=0.0)
        session.start()
        time.sleep(0.2)

        with pytest.raises(RuntimeError, match="Control loop failed"):
            session.check_health()

        session.stop()


class TestSessionRunDuration:
    def test_run_completes_in_expected_time(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        session = Session(robot, _zero_policy, rate_hz=100, warmup_s=0.0)

        t0 = time.perf_counter()
        session.run(duration=0.5)
        elapsed = time.perf_counter() - t0

        assert elapsed == pytest.approx(0.5, abs=0.2)
        assert not session.is_running


class TestRunConvenience:
    def test_returns_stats(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        stats = run(robot, _zero_policy, rate_hz=100, duration=0.3, warmup_s=0.0)
        assert isinstance(stats, SessionStats)
        assert stats.iterations > 0


class TestSessionStopIdempotent:
    def test_stop_twice(self):
        robot = MockRobot(state_dim=12, action_dim=6)
        session = Session(robot, _zero_policy, rate_hz=100, warmup_s=0.0)
        session.start()
        time.sleep(0.1)
        session.stop()
        session.stop()  # should not raise
        assert not session.is_running
