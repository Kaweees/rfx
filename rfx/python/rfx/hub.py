"""
rfx.hub - Load, push, and inspect self-describing policy models.

Provides the main entry points for working with saved policies:

    >>> loaded = rfx.load_policy("runs/go2-walk-v1")
    >>> loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")
    >>> rfx.push_policy("runs/go2-walk-v1", "rfx-community/go2-walk-v1")
    >>> rfx.inspect_policy("runs/go2-walk-v1")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_policy(source: str | Path) -> LoadedPolicy:
    """Load a self-describing policy from a local path or HuggingFace Hub.

    Args:
        source: Local directory path, or "hf://org/repo" for Hub models.

    Returns:
        LoadedPolicy with .policy, .robot_config, .normalizer, .config
    """
    from .config import RobotConfig
    from .nn import Policy
    from .utils.transforms import ObservationNormalizer

    path = _resolve_source(source)
    config = json.loads((path / "rfx_config.json").read_text())

    policy = Policy.load(path)

    robot_config = None
    if "robot_config" in config:
        robot_config = RobotConfig.from_dict(dict(config["robot_config"]))

    normalizer = None
    norm_path = path / "normalizer.json"
    if norm_path.exists():
        normalizer = ObservationNormalizer.from_dict(json.loads(norm_path.read_text()))

    return LoadedPolicy(
        policy=policy,
        robot_config=robot_config,
        normalizer=normalizer,
        config=config,
    )


def push_policy(path: str | Path, repo_id: str, *, private: bool = False) -> str:
    """Push a saved policy directory to HuggingFace Hub.

    Args:
        path: Local directory containing the saved policy.
        repo_id: HuggingFace repo ID (e.g. "rfx-community/go2-walk-v1").
        private: Whether to create a private repo.

    Returns:
        The URL of the uploaded model.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, private=private)
    api.upload_folder(folder_path=str(path), repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"


def inspect_policy(source: str | Path) -> dict[str, Any]:
    """Read rfx_config.json without loading weights. Useful for quick inspection.

    Args:
        source: Local directory path, or "hf://org/repo" for Hub models.

    Returns:
        The parsed rfx_config.json dict.
    """
    path = _resolve_source(source)
    return json.loads((path / "rfx_config.json").read_text())


def _resolve_source(source: str | Path) -> Path:
    """Resolve a source string to a local path, downloading from HF if needed."""
    source = str(source)
    if source.startswith("hf://"):
        from huggingface_hub import snapshot_download

        repo_id = source[5:]  # strip "hf://"
        return Path(snapshot_download(repo_id))
    return Path(source)


@dataclass
class LoadedPolicy:
    """A loaded policy with all its metadata.

    Callable — accepts the obs dict that ``robot.observe()`` returns and
    handles torch↔tinygrad conversion automatically so it plugs straight
    into ``rfx.run(robot, loaded_policy)``.
    """

    policy: Any  # Policy
    robot_config: Any  # RobotConfig | None
    normalizer: Any  # ObservationNormalizer | None
    config: dict[str, Any]

    def __call__(self, obs):
        # When called from rfx.run() / Session, obs is dict[str, torch.Tensor].
        # Normalizer works on torch dicts, so run it before conversion.
        if isinstance(obs, dict):
            if self.normalizer is not None:
                obs = self.normalizer.normalize(obs)
            obs = self._dict_to_tinygrad(obs)
            action = self.policy(obs)
            return self._tinygrad_to_torch(action)

        # Raw tinygrad tensor passed directly — no conversion needed.
        return self.policy(obs)

    # ------------------------------------------------------------------
    # torch ↔ tinygrad helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_tinygrad(obs: dict) -> Any:
        """Extract 'state' tensor from obs dict, convert torch → tinygrad."""
        from tinygrad import Tensor as TinyTensor

        state = obs["state"]
        # torch.Tensor → numpy → tinygrad.Tensor
        return TinyTensor(state.numpy())

    @staticmethod
    def _tinygrad_to_torch(t: Any) -> Any:
        """Convert tinygrad.Tensor → torch.Tensor."""
        import torch

        return torch.from_numpy(t.numpy())

    @property
    def policy_type(self) -> str:
        return self.config.get("policy_type", "unknown")

    @property
    def training_info(self) -> dict[str, Any]:
        return self.config.get("training", {})
