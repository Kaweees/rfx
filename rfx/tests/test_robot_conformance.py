"""Contract conformance tests for supported robot adapters."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rfx import MockRobot, SimRobot

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")

if TORCH_AVAILABLE:
    import torch
else:
    torch = None


def _assert_robot_contract(robot) -> None:
    obs = robot.observe()
    assert isinstance(obs, dict)
    assert "state" in obs
    assert obs["state"].shape[0] == robot.num_envs
    assert obs["state"].shape[1] == robot.max_state_dim

    action = torch.zeros(robot.num_envs, robot.max_action_dim)
    robot.act(action)
    reset_obs = robot.reset()
    assert reset_obs["state"].shape == obs["state"].shape


def test_mock_robot_conformance() -> None:
    robot = MockRobot(state_dim=12, action_dim=6, num_envs=2)
    _assert_robot_contract(robot)


def test_sim_robot_mock_backend_conformance() -> None:
    config_path = Path(__file__).parent.parent / "configs" / "so101.yaml"
    robot = SimRobot.from_config(config_path, backend="mock", num_envs=2, device="cpu")
    _assert_robot_contract(robot)
