"""Tests for teleop jitter benchmark helpers."""

from __future__ import annotations

import pytest

from rfx.teleop.benchmark import assert_jitter_budget, run_jitter_benchmark


def test_run_jitter_benchmark_returns_stats() -> None:
    result = run_jitter_benchmark(rate_hz=120.0, duration_s=0.05, warmup_s=0.01)
    payload = result.to_dict()
    assert payload["rate_hz"] == 120.0
    assert payload["iterations"] > 0
    assert payload["p99_jitter_s"] >= 0.0


def test_assert_jitter_budget_raises_when_exceeded() -> None:
    result = run_jitter_benchmark(rate_hz=120.0, duration_s=0.03, warmup_s=0.01)
    with pytest.raises(RuntimeError):
        assert_jitter_budget(result, p99_budget_s=0.0)
