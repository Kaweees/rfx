"""rfx.collection._recorder — Real-time frame recorder to LeRobot Dataset."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._mcap import McapSidecar

from ._dataset import Dataset


def _dataset_add_frame(dataset: Any, frame: dict[str, Any], task_name: str) -> None:
    """Add a frame to a LeRobotDataset with version-compat fallback."""
    add_frame = getattr(dataset, "add_frame", None)
    if add_frame is None:
        raise AttributeError("Dataset instance has no add_frame method")

    attempts = [
        lambda: add_frame(frame, task=task_name),
        lambda: add_frame(frame),
        lambda: add_frame({**frame, "task": task_name}),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            attempt()
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError("Unable to add frame to LeRobot dataset") from last_error


class Recorder:
    """Real-time frame recorder that writes directly to a LeRobot Dataset.

    Thread-safe. Designed to be called from a teleop control loop.

    Usage:
        recorder = Recorder.create("my-org/demos", robot_type="so101", state_dim=6)
        recorder.start_episode(task="pick-place")
        for frame in control_loop:
            recorder.add_frame(state=positions, action=positions, images={"cam0": img})
        recorder.save_episode()
        recorder.push()
    """

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._lock = threading.Lock()
        self._episode_active = False
        self._current_task = "default"
        self._frame_count = 0
        self._mcap_writer: McapSidecar | None = None

    @classmethod
    def create(
        cls,
        repo_id: str,
        *,
        root: str | Path = "datasets",
        fps: int = 30,
        robot_type: str = "so101",
        state_dim: int = 6,
        camera_names: Sequence[str] = (),
        camera_shape: tuple[int, int, int] = (480, 640, 3),
        use_videos: bool = True,
        mcap: bool = False,
    ) -> Recorder:
        """Create a recorder with a new dataset."""
        dataset = Dataset.create(
            repo_id,
            root=root,
            fps=fps,
            robot_type=robot_type,
            state_dim=state_dim,
            camera_names=camera_names,
            camera_shape=camera_shape,
            use_videos=use_videos,
        )
        recorder = cls(dataset)
        if mcap:
            from ._mcap import McapSidecar

            recorder._mcap_writer = McapSidecar(Path(root) / repo_id)
        return recorder

    def start_episode(self, *, task: str = "default") -> None:
        """Begin a new episode."""
        with self._lock:
            if self._episode_active:
                raise RuntimeError("Episode already active")
            self._episode_active = True
            self._current_task = task
            self._frame_count = 0
            if self._mcap_writer:
                episode_id = f"episode_{self._dataset.num_episodes}"
                self._mcap_writer.start_episode(episode_id)

    def add_frame(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray | None = None,
        images: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Add a single frame to the active episode.

        Thread-safe. Called from the teleop control loop at rate_hz.
        Writes directly to LeRobotDataset.add_frame().
        """
        with self._lock:
            if not self._episode_active:
                raise RuntimeError("No active episode. Call start_episode() first.")

            frame: dict[str, Any] = {
                "observation.state": np.asarray(state, dtype=np.float32),
            }
            frame["action"] = (
                np.asarray(action, dtype=np.float32)
                if action is not None
                else frame["observation.state"].copy()
            )
            if images:
                for name, img in images.items():
                    frame[f"observation.images.{name}"] = np.asarray(img, dtype=np.uint8)

            _dataset_add_frame(self._dataset._inner, frame, self._current_task)
            self._frame_count += 1

            if self._mcap_writer:
                self._mcap_writer.write_frame(frame)

    def save_episode(self) -> int:
        """Finalize the current episode. Returns frame count."""
        with self._lock:
            if not self._episode_active:
                raise RuntimeError("No active episode to save.")

            save_episode = getattr(self._dataset._inner, "save_episode", None)
            if save_episode is not None:
                save_episode()

            count = self._frame_count
            self._frame_count = 0
            self._episode_active = False

            if self._mcap_writer:
                self._mcap_writer.save_episode()

            return count

    def push(self, repo_id: str | None = None) -> None:
        """Push the dataset to HuggingFace Hub."""
        self._dataset.push(repo_id)

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def is_recording(self) -> bool:
        return self._episode_active
