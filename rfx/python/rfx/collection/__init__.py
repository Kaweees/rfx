"""rfx.collection — LeRobot-native data collection for robots."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from ._dataset import Dataset
from ._hub import from_hub, pull, push
from ._recorder import Recorder


def collect(
    robot_type: str,
    repo_id: str,
    *,
    output: str | Path = "datasets",
    episodes: int = 1,
    duration_s: float | None = None,
    task: str = "default",
    fps: int = 30,
    state_dim: int = 6,
    camera_names: Sequence[str] = (),
    push_to_hub: bool = False,
    mcap: bool = False,
) -> Dataset:
    """Collect episodes from a robot into a LeRobot dataset.

    The simplest path from hardware to HuggingFace Hub.

    Example:
        dataset = rfx.collection.collect("so101", "my-org/demos", episodes=10)
        dataset.push()
    """
    recorder = Recorder.create(
        repo_id,
        root=output,
        fps=fps,
        robot_type=robot_type,
        state_dim=state_dim,
        camera_names=camera_names,
        mcap=mcap,
    )

    for _ep in range(episodes):
        recorder.start_episode(task=task)

        if duration_s is not None:
            deadline = time.perf_counter() + duration_s
            while time.perf_counter() < deadline:
                time.sleep(0.01)
        else:
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

        recorder.save_episode()

    if push_to_hub:
        recorder.push()

    return recorder.dataset


def open_dataset(repo_id: str, *, root: str | Path = "datasets") -> Dataset:
    """Open an existing local LeRobot dataset."""
    return Dataset.open(repo_id, root=root)


__all__ = [
    "Dataset",
    "Recorder",
    "collect",
    "from_hub",
    "open_dataset",
    "pull",
    "push",
]
