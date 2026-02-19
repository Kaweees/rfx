"""Tests for self-describing policy save/load/inspect."""

from __future__ import annotations

import json

import pytest

# Skip entire module if tinygrad not available
tinygrad = pytest.importorskip("tinygrad")
torch = pytest.importorskip("torch")

from tinygrad import Tensor
from tinygrad.nn.state import get_state_dict, safe_save

from rfx.config import RobotConfig
from rfx.hub import LoadedPolicy, inspect_policy, load_policy
from rfx.nn import MLP, ActorCritic, Policy, _POLICY_REGISTRY
from rfx.utils.transforms import ObservationNormalizer


# ---------------------------------------------------------------------------
# 1. Save / load round-trip
# ---------------------------------------------------------------------------


def test_mlp_save_load_roundtrip(tmp_path):
    """Save an MLP, load it back, verify forward pass produces same output."""
    policy = MLP(obs_dim=8, act_dim=3, hidden=[16, 16])
    obs = Tensor.randn(1, 8)
    expected = policy(obs).numpy()

    policy.save(tmp_path / "test-model")
    loaded = MLP.load(tmp_path / "test-model")

    actual = loaded(obs).numpy()
    assert actual.shape == expected.shape
    assert abs(actual - expected).max() < 1e-5


def test_actor_critic_save_load_roundtrip(tmp_path):
    """Save an ActorCritic, load it back, verify forward pass."""
    policy = ActorCritic(obs_dim=8, act_dim=3, hidden=[16, 16])
    obs = Tensor.randn(1, 8)
    expected = policy(obs).numpy()

    policy.save(tmp_path / "test-ac")
    loaded = ActorCritic.load(tmp_path / "test-ac")

    actual = loaded(obs).numpy()
    assert actual.shape == expected.shape
    assert abs(actual - expected).max() < 1e-5


# ---------------------------------------------------------------------------
# 2. Config preservation (robot_config + normalizer)
# ---------------------------------------------------------------------------


def test_config_preservation(tmp_path):
    """Save with robot_config + normalizer, load back, verify metadata."""
    policy = MLP(obs_dim=8, act_dim=3)
    robot_config = RobotConfig(name="TestBot", state_dim=8, action_dim=3, control_freq_hz=100)

    norm = ObservationNormalizer(state_dim=8)
    norm.update(torch.randn(10, 8))

    policy.save(
        tmp_path / "full-model",
        robot_config=robot_config,
        normalizer=norm,
        training_info={"total_steps": 1000, "best_reward": 42.0},
    )

    loaded = load_policy(tmp_path / "full-model")

    assert isinstance(loaded, LoadedPolicy)
    assert loaded.robot_config is not None
    assert loaded.robot_config.name == "TestBot"
    assert loaded.robot_config.control_freq_hz == 100
    assert loaded.normalizer is not None
    assert loaded.normalizer._count > 0
    assert loaded.policy_type == "MLP"
    assert loaded.training_info["total_steps"] == 1000
    assert loaded.training_info["best_reward"] == 42.0


# ---------------------------------------------------------------------------
# 3. Policy registry
# ---------------------------------------------------------------------------


def test_registry_has_builtins():
    """MLP and ActorCritic should be registered."""
    assert "MLP" in _POLICY_REGISTRY
    assert "ActorCritic" in _POLICY_REGISTRY
    assert _POLICY_REGISTRY["MLP"] is MLP
    assert _POLICY_REGISTRY["ActorCritic"] is ActorCritic


def test_auto_detect_type_from_registry(tmp_path):
    """Policy.load() on base class auto-detects type from rfx_config.json."""
    policy = MLP(obs_dim=4, act_dim=2, hidden=[8])
    policy.save(tmp_path / "auto-detect")

    loaded = Policy.load(tmp_path / "auto-detect")
    assert isinstance(loaded, MLP)
    assert loaded.obs_dim == 4
    assert loaded.act_dim == 2


# ---------------------------------------------------------------------------
# 4. Legacy fallback
# ---------------------------------------------------------------------------


def test_legacy_safetensors_fallback(tmp_path):
    """A bare .safetensors file should still load on a subclass with matching shape."""
    policy = MLP(obs_dim=4, act_dim=2, hidden=[8])
    obs = Tensor.randn(1, 4)
    expected = policy(obs).numpy()

    # Save as a bare safetensors file (legacy format)
    state = get_state_dict(policy)
    legacy_path = tmp_path / "legacy.safetensors"
    safe_save(state, str(legacy_path))

    # Load via subclass (must create matching architecture manually)
    loaded = MLP(obs_dim=4, act_dim=2, hidden=[8])
    from tinygrad.nn.state import load_state_dict, safe_load

    load_state_dict(loaded, safe_load(str(legacy_path)))
    actual = loaded(obs).numpy()
    assert abs(actual - expected).max() < 1e-5


# ---------------------------------------------------------------------------
# 5. inspect_policy
# ---------------------------------------------------------------------------


def test_inspect_policy(tmp_path):
    """inspect_policy returns config dict without loading weights."""
    policy = MLP(obs_dim=4, act_dim=2, hidden=[8])
    policy.save(tmp_path / "inspect-test")

    config = inspect_policy(tmp_path / "inspect-test")
    assert config["policy_type"] == "MLP"
    assert config["policy_config"]["obs_dim"] == 4
    assert config["policy_config"]["act_dim"] == 2
    assert "rfx_version" in config


# ---------------------------------------------------------------------------
# 6. Normalizer round-trip
# ---------------------------------------------------------------------------


def test_normalizer_roundtrip():
    """to_dict -> from_dict preserves mean/var/count."""
    norm = ObservationNormalizer(state_dim=4, clip=5.0, eps=1e-6)
    norm.update(torch.randn(20, 4))
    norm.update(torch.randn(30, 4))

    d = norm.to_dict()
    restored = ObservationNormalizer.from_dict(d)

    assert restored.state_dim == 4
    assert restored.clip == 5.0
    assert restored.eps == 1e-6
    assert restored._count == norm._count
    assert torch.allclose(restored._mean, norm._mean, atol=1e-6)
    assert torch.allclose(restored._var, norm._var, atol=1e-6)


def test_normalizer_json_serializable():
    """to_dict output must be JSON-serializable."""
    norm = ObservationNormalizer(state_dim=4)
    norm.update(torch.randn(10, 4))
    d = norm.to_dict()
    # Should not raise
    json.dumps(d)


# ---------------------------------------------------------------------------
# 7. LoadedPolicy is callable
# ---------------------------------------------------------------------------


def test_loaded_policy_callable(tmp_path):
    """LoadedPolicy.__call__ passes through to policy."""
    policy = MLP(obs_dim=4, act_dim=2, hidden=[8])
    policy.save(tmp_path / "callable-test")

    loaded = load_policy(tmp_path / "callable-test")
    obs = Tensor.randn(1, 4)
    result = loaded(obs)
    assert result.numpy().shape == (1, 2)


# ---------------------------------------------------------------------------
# 8. Directory format
# ---------------------------------------------------------------------------


def test_save_creates_expected_files(tmp_path):
    """save() creates rfx_config.json, model.safetensors, and optionally normalizer.json."""
    policy = MLP(obs_dim=4, act_dim=2)
    norm = ObservationNormalizer(state_dim=4)

    out = policy.save(tmp_path / "dir-test", normalizer=norm)
    assert (out / "rfx_config.json").exists()
    assert (out / "model.safetensors").exists()
    assert (out / "normalizer.json").exists()


def test_save_without_normalizer(tmp_path):
    """save() without normalizer should not create normalizer.json."""
    policy = MLP(obs_dim=4, act_dim=2)
    out = policy.save(tmp_path / "no-norm")
    assert (out / "rfx_config.json").exists()
    assert (out / "model.safetensors").exists()
    assert not (out / "normalizer.json").exists()
