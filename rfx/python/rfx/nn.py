"""
Neural network primitives using tinygrad

A simple, transparent implementation of policies using tinygrad tensors.
Follows the tinygrad philosophy: simple to start, powerful enough for production.

Example:
    >>> from rfx.nn import MLP, go2_mlp
    >>> policy = go2_mlp()
    >>> obs = Tensor.randn(1, 48)
    >>> actions = policy(obs)  # JIT compiled on second call
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import rfx

from .jit import PolicyJitRuntime

try:
    from tinygrad import Tensor
    from tinygrad.engine.jit import TinyJit
    from tinygrad.nn import Linear
    from tinygrad.nn.state import (
        get_parameters,
        get_state_dict,
        load_state_dict,
        safe_load,
        safe_save,
    )

    TINYGRAD_AVAILABLE = True
except ImportError:
    TINYGRAD_AVAILABLE = False
    Tensor = Any

    def TinyJit(x):
        return x  # no-op decorator


def _check_tinygrad():
    if not TINYGRAD_AVAILABLE:
        raise ImportError(
            "tinygrad is required for neural network support. Install with: pip install tinygrad"
        )


_POLICY_REGISTRY: dict[str, type[Policy]] = {}


def register_policy(cls: type[Policy]) -> type[Policy]:
    """Register a policy class for auto-detection during load."""
    _POLICY_REGISTRY[cls.__name__] = cls
    return cls


class Policy:
    """
    Base policy class for neural network policies.

    Users can subclass this to create custom architectures.
    The forward method should take observations and return actions.

    Example:
        >>> class CustomPolicy(Policy):
        ...     def __init__(self):
        ...         self.l1 = Linear(48, 256)
        ...         self.l2 = Linear(256, 12)
        ...
        ...     def forward(self, obs: Tensor) -> Tensor:
        ...         x = self.l1(obs).tanh()
        ...         return self.l2(x).tanh()
    """

    def forward(self, obs: Tensor) -> Tensor:
        """
        Forward pass: observations -> actions.

        Args:
            obs: Observation tensor of shape (batch, obs_dim)

        Returns:
            Action tensor of shape (batch, act_dim)
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, obs: Tensor) -> Tensor:
        """Run inference (JIT compiled after first call)."""
        return self.forward(obs)

    def config_dict(self) -> dict[str, Any]:
        """Return constructor arguments needed to recreate this policy."""
        raise NotImplementedError(f"{type(self).__name__} must implement config_dict() for saving")

    @classmethod
    def from_config_dict(cls, d: dict[str, Any]) -> Policy:
        """Reconstruct a policy from its config dict."""
        return cls(**d)

    def parameters(self) -> list:
        """Get all trainable parameters."""
        _check_tinygrad()
        return get_parameters(self)

    def save(
        self,
        path: str | Path,
        *,
        robot_config: Any | None = None,
        normalizer: Any | None = None,
        training_info: dict[str, Any] | None = None,
    ) -> Path:
        """Save policy as a self-describing directory.

        Args:
            path: Directory path to save into
            robot_config: Optional RobotConfig to bundle
            normalizer: Optional ObservationNormalizer to bundle
            training_info: Optional training metadata dict

        Returns:
            The directory path
        """
        _check_tinygrad()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Weights
        state = get_state_dict(self)
        safe_save(state, str(path / "model.safetensors"))

        # Config
        config: dict[str, Any] = {
            "rfx_version": rfx.__version__,
            "policy_type": type(self).__name__,
            "policy_config": self.config_dict(),
        }
        if robot_config is not None:
            config["robot_config"] = robot_config.to_dict()
        if training_info is not None:
            config["training"] = training_info

        (path / "rfx_config.json").write_text(json.dumps(config, indent=2))

        # Normalizer
        if normalizer is not None:
            (path / "normalizer.json").write_text(json.dumps(normalizer.to_dict(), indent=2))

        return path

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        """Load a self-describing policy from a directory.

        If called on the base Policy class, auto-detects the type from rfx_config.json.
        If called on a subclass (e.g. MLP.load()), uses that subclass.
        Falls back to legacy single-file safetensors if no directory found.

        Args:
            path: Directory path or legacy .safetensors file

        Returns:
            Policy instance with loaded weights
        """
        _check_tinygrad()
        path = Path(path)

        # Legacy: bare safetensors file
        if path.is_file() and path.suffix == ".safetensors":
            policy = cls()
            load_state_dict(policy, safe_load(str(path)))
            return policy

        config_path = path / "rfx_config.json"
        config = json.loads(config_path.read_text())

        # Resolve class
        if cls is Policy:
            policy_cls = _POLICY_REGISTRY.get(config["policy_type"])
            if policy_cls is None:
                raise ValueError(f"Unknown policy type: {config['policy_type']}")
        else:
            policy_cls = cls

        policy = policy_cls.from_config_dict(config["policy_config"])
        load_state_dict(policy, safe_load(str(path / "model.safetensors")))
        return policy

    def to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert a tinygrad tensor to numpy array."""
        return tensor.numpy()


@register_policy
class MLP(Policy):
    """
    Multi-layer perceptron policy.

    A simple feedforward network with tanh activations.
    Suitable for most locomotion tasks.

    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden: List of hidden layer sizes (default: [256, 256])

    Example:
        >>> policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])
        >>> obs = Tensor.randn(1, 48)
        >>> actions = policy(obs)
        >>> print(actions.shape)  # (1, 12)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int] | None = None,
    ):
        _check_tinygrad()

        if hidden is None:
            hidden = [256, 256]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        # Build layers
        dims = [obs_dim] + hidden + [act_dim]
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))

    def config_dict(self) -> dict[str, Any]:
        return {"obs_dim": self.obs_dim, "act_dim": self.act_dim, "hidden": list(self.hidden)}

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass with tanh activations."""
        x = obs
        for layer in self.layers[:-1]:
            x = layer(x).tanh()
        # Final layer also uses tanh to bound actions to [-1, 1]
        return self.layers[-1](x).tanh()

    def __repr__(self) -> str:
        return f"MLP(obs_dim={self.obs_dim}, act_dim={self.act_dim}, hidden={self.hidden})"


class JitPolicy(Policy):
    """
    A policy wrapper that enables TinyJit compilation.

    Wraps any policy and JIT compiles its forward pass for faster inference.
    The first call traces the computation graph, subsequent calls are fast.

    Save/load delegates to the inner policy -- JIT compilation state is
    re-created on load automatically.

    Args:
        policy: The policy to wrap

    Example:
        >>> mlp = MLP(48, 12)
        >>> jit_policy = JitPolicy(mlp)
        >>> # First call: traces graph
        >>> actions = jit_policy(obs)
        >>> # Second call: runs compiled kernel
        >>> actions = jit_policy(obs)
    """

    def __init__(self, policy: Policy):
        _check_tinygrad()
        self._policy = policy
        fallback = TinyJit(policy.forward)
        self._jit_runtime = PolicyJitRuntime(
            policy.forward,
            fallback=fallback,
            name=f"{policy.__class__.__name__}_forward",
        )
        self._jit_forward = self._jit_runtime
        self._rfx_jit_backend = self._jit_runtime.backend

    def config_dict(self) -> dict[str, Any]:
        """Delegate to the inner policy's config_dict."""
        return self._policy.config_dict()

    def save(self, path, **kwargs) -> Path:
        """Save the inner policy (JIT state is re-created on load)."""
        return self._policy.save(path, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> Policy:
        """Load the inner policy and wrap it with JIT compilation."""
        inner = Policy.load(path)
        return cls(inner)

    def forward(self, obs: Tensor) -> Tensor:
        return self._jit_forward(obs)

    def __repr__(self) -> str:
        return f"JitPolicy({self._policy!r})"


@register_policy
class ActorCritic(Policy):
    """
    Actor-critic network for PPO training.

    Shares a backbone between actor (policy) and critic (value function).

    Args:
        obs_dim: Observation dimension
        act_dim: Action dimension
        hidden: Hidden layer sizes for shared backbone

    Example:
        >>> ac = ActorCritic(48, 12)
        >>> obs = Tensor.randn(32, 48)
        >>> actions, values = ac.forward_actor_critic(obs)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int] | None = None,
    ):
        _check_tinygrad()

        if hidden is None:
            hidden = [256, 256]

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        # Shared backbone
        self.backbone = []
        dims = [obs_dim] + hidden
        for i in range(len(dims) - 1):
            self.backbone.append(Linear(dims[i], dims[i + 1]))

        # Actor head (outputs action mean)
        self.actor_head = Linear(hidden[-1], act_dim)

        # Critic head (outputs value)
        self.critic_head = Linear(hidden[-1], 1)

        # Learnable log std for action distribution
        self.log_std = Tensor.zeros(act_dim)

    def config_dict(self) -> dict[str, Any]:
        return {"obs_dim": self.obs_dim, "act_dim": self.act_dim, "hidden": list(self.hidden)}

    def _backbone_forward(self, obs: Tensor) -> Tensor:
        """Forward through shared backbone."""
        x = obs
        for layer in self.backbone:
            x = layer(x).tanh()
        return x

    def forward(self, obs: Tensor) -> Tensor:
        """Get action mean (for inference)."""
        features = self._backbone_forward(obs)
        return self.actor_head(features).tanh()

    def forward_actor_critic(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Get both actions and values (for training).

        Returns:
            Tuple of (action_mean, value)
        """
        features = self._backbone_forward(obs)
        action_mean = self.actor_head(features).tanh()
        value = self.critic_head(features)
        return action_mean, value

    def get_action_and_value(
        self, obs: Tensor, action: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample action and compute log prob + entropy (for PPO update).

        Args:
            obs: Observations
            action: Optional pre-sampled action (for computing log prob)

        Returns:
            Tuple of (action, log_prob, entropy, value)
        """
        features = self._backbone_forward(obs)
        action_mean = self.actor_head(features).tanh()
        value = self.critic_head(features)

        # Gaussian action distribution
        std = self.log_std.exp()

        if action is None:
            # Sample from Gaussian
            noise = Tensor.randn(*action_mean.shape)
            action = (action_mean + noise * std).tanh()

        # Log probability (simplified, ignoring tanh correction)
        log_prob = (
            -0.5 * ((action - action_mean) / std).pow(2) - self.log_std - 0.5 * np.log(2 * np.pi)
        ).sum(axis=-1)

        # Entropy
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + self.log_std).sum()

        return action, log_prob, entropy, value.squeeze(-1)


# Convenience constructors for Go2 robot
def go2_mlp(hidden: list[int] | None = None) -> MLP:
    """
    Create an MLP policy sized for the Go2 robot.

    Go2 observation space: 48 dimensions
    Go2 action space: 12 dimensions (joint positions)

    Args:
        hidden: Hidden layer sizes (default: [256, 256])

    Returns:
        MLP policy for Go2
    """
    if hidden is None:
        hidden = [256, 256]
    return MLP(obs_dim=48, act_dim=12, hidden=hidden)


def go2_actor_critic(hidden: list[int] | None = None) -> ActorCritic:
    """
    Create an ActorCritic network sized for the Go2 robot.

    Args:
        hidden: Hidden layer sizes (default: [256, 256])

    Returns:
        ActorCritic network for Go2
    """
    if hidden is None:
        hidden = [256, 256]
    return ActorCritic(obs_dim=48, act_dim=12, hidden=hidden)


__all__ = [
    "Policy",
    "MLP",
    "JitPolicy",
    "ActorCritic",
    "go2_mlp",
    "go2_actor_critic",
    "register_policy",
    "TINYGRAD_AVAILABLE",
]
