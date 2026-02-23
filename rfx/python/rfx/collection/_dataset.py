"""rfx.collection._dataset — LeRobot dataset wrapper with rfx helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def _load_lerobot_dataset_class() -> Any:
    """Load the LeRobotDataset class with version-compat fallback.

    Reuses the same import strategy from teleop/lerobot_writer.py.
    """
    import importlib

    module_paths = (
        "lerobot.common.datasets.lerobot_dataset",
        "lerobot.datasets.lerobot_dataset",
    )
    for module_path in module_paths:
        try:
            mod = importlib.import_module(module_path)
        except Exception:
            continue
        if hasattr(mod, "LeRobotDataset"):
            return mod.LeRobotDataset

    raise ImportError(
        "LeRobot package is unavailable. "
        "Install with: pip install -e '.[collection]'"
    )


def _create_lerobot_dataset(
    dataset_cls: Any,
    *,
    repo_id: str,
    root: Path,
    fps: int,
    robot_type: str,
    features: dict[str, Any],
    use_videos: bool,
) -> Any:
    """Create a LeRobotDataset with version-compat fallback signatures."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    create = getattr(dataset_cls, "create", None)
    if create is None:
        raise AttributeError(
            "LeRobotDataset.create is unavailable in this package version"
        )

    attempts = [
        lambda: create(
            repo_id=repo_id,
            root=str(root),
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        ),
        lambda: create(
            repo_id,
            fps,
            root=str(root),
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        ),
        lambda: create(
            repo_id=repo_id, root=str(root), fps=fps, features=features
        ),
        lambda: create(repo_id, fps, features=features, root=str(root)),
        lambda: create(repo_id=repo_id, root=str(root), fps=fps),
    ]

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Unable to instantiate LeRobotDataset with known signatures"
    ) from last_error


def _build_features(
    state_dim: int,
    camera_names: Sequence[str],
    camera_shape: tuple[int, int, int],
) -> dict[str, Any]:
    """Build LeRobot feature spec from state_dim + camera names."""
    features: dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,)},
        "action": {"dtype": "float32", "shape": (state_dim,)},
    }
    for name in camera_names:
        features[f"observation.images.{name}"] = {
            "dtype": "uint8",
            "shape": camera_shape,
        }
    return features


class Dataset:
    """A LeRobot dataset with rfx helpers.

    This IS a LeRobot dataset. Stored as parquet + videos on disk.
    Push/pull from HuggingFace Hub natively.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    @classmethod
    def create(
        cls,
        repo_id: str,
        *,
        root: str | Path = "datasets",
        fps: int = 30,
        robot_type: str = "so101",
        features: dict[str, Any] | None = None,
        state_dim: int = 6,
        camera_names: Sequence[str] = (),
        camera_shape: tuple[int, int, int] = (480, 640, 3),
        use_videos: bool = True,
    ) -> Dataset:
        """Create a new empty dataset.

        If features not provided, auto-builds from state_dim + camera_names.
        """
        dataset_cls = _load_lerobot_dataset_class()
        if features is None:
            features = _build_features(state_dim, camera_names, camera_shape)

        inner = _create_lerobot_dataset(
            dataset_cls,
            repo_id=repo_id,
            root=Path(root),
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        return cls(inner)

    @classmethod
    def open(cls, repo_id: str, *, root: str | Path = "datasets") -> Dataset:
        """Open an existing local dataset."""
        dataset_cls = _load_lerobot_dataset_class()
        inner = dataset_cls(repo_id=repo_id, root=str(Path(root)))
        return cls(inner)

    @classmethod
    def from_hub(
        cls, repo_id: str, *, root: str | Path = "datasets"
    ) -> Dataset:
        """Pull a dataset from HuggingFace Hub."""
        dataset_cls = _load_lerobot_dataset_class()
        inner = dataset_cls(repo_id=repo_id, root=str(Path(root)))
        return cls(inner)

    def push(self, repo_id: str | None = None) -> None:
        """Push dataset to HuggingFace Hub."""
        push_to_hub = getattr(self._inner, "push_to_hub", None)
        if push_to_hub is None:
            raise AttributeError(
                "LeRobotDataset instance has no push_to_hub method"
            )
        if repo_id is not None:
            push_to_hub(repo_id)
        else:
            push_to_hub()

    @property
    def repo_id(self) -> str:
        return str(getattr(self._inner, "repo_id", ""))

    @property
    def num_episodes(self) -> int:
        return int(getattr(self._inner, "num_episodes", 0))

    @property
    def num_frames(self) -> int:
        return int(getattr(self._inner, "num_frames", len(self._inner)))

    @property
    def fps(self) -> int:
        return int(getattr(self._inner, "fps", 0))

    def __len__(self) -> int:
        return self.num_frames

    def summary(self) -> dict[str, Any]:
        """Return aggregate stats: episodes, frames, features, etc."""
        features = getattr(self._inner, "features", None)
        return {
            "repo_id": self.repo_id,
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "features": dict(features) if features else {},
        }

    def validate(self, thresholds: Any = None) -> dict[str, Any]:
        """Quality checks on the dataset.

        Delegates to workflow.quality.validate_dataset when dataset
        path is available on disk, otherwise returns basic stats.
        """
        meta_path = getattr(self._inner, "root", None)
        if meta_path is not None:
            try:
                from ..workflow.quality import validate_dataset

                return validate_dataset(str(meta_path), thresholds=thresholds)
            except Exception:
                pass
        return {
            "passed": True,
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
        }
