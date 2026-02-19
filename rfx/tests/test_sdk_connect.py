"""Tests for rfx.connect_robot default config behavior."""

import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is required")

from rfx import connect_robot


class TestConnectRobotDefaults:
    def test_go2_mock_without_config(self):
        bot = connect_robot("go2", backend="mock", num_envs=2, device="cpu")
        obs = bot.reset()
        assert obs["state"].shape == (2, 64)
        bot.close()

    def test_so101_mock_without_config(self):
        bot = connect_robot("so101", backend="mock", num_envs=1, device="cpu")
        obs = bot.reset()
        assert obs["state"].shape == (1, 64)
        bot.close()
