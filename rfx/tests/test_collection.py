"""Tests for rfx.collection — LeRobot-native data collection SDK."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake LeRobotDataset for unit tests (no real lerobot dependency needed)
# ---------------------------------------------------------------------------
class _FakeLeRobotDataset:
    """Minimal mock of LeRobotDataset for testing collection module."""

    _instances: list[_FakeLeRobotDataset] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.repo_id = kwargs.get("repo_id", "")
        self.fps = kwargs.get("fps", 30)
        self.frames: list[tuple[dict, str | None]] = []
        self.episodes_saved = 0
        self.pushed = False
        self.pushed_repo_id: str | None = None
        self.features = kwargs.get("features", {})
        self._instances.append(self)

    @classmethod
    def create(cls, **kwargs):
        instance = cls(**kwargs)
        return instance

    def add_frame(self, frame, task=None):
        self.frames.append((frame, task))

    def save_episode(self):
        self.episodes_saved += 1

    def push_to_hub(self, repo_id=None):
        self.pushed = True
        self.pushed_repo_id = repo_id

    @property
    def num_episodes(self):
        return self.episodes_saved

    @property
    def num_frames(self):
        return len(self.frames)

    def __len__(self):
        return len(self.frames)


def _install_fake_lerobot(monkeypatch) -> None:
    """Install fake lerobot modules into sys.modules."""
    package = types.ModuleType("lerobot")
    common = types.ModuleType("lerobot.common")
    datasets = types.ModuleType("lerobot.common.datasets")
    dataset_mod = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeLeRobotDataset

    package.common = common
    common.datasets = datasets
    datasets.lerobot_dataset = dataset_mod

    monkeypatch.setitem(sys.modules, "lerobot", package)
    monkeypatch.setitem(sys.modules, "lerobot.common", common)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.common.datasets.lerobot_dataset", dataset_mod)


@pytest.fixture(autouse=True)
def _reset_fake_instances():
    _FakeLeRobotDataset._instances.clear()
    yield
    _FakeLeRobotDataset._instances.clear()


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------
class TestDataset:
    def test_create_default_features(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        ds = Dataset.create("test/demo", state_dim=6, fps=30)
        assert ds.repo_id == "test/demo"
        assert ds.fps == 30

    def test_create_with_cameras(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        ds = Dataset.create(
            "test/demo",
            state_dim=6,
            camera_names=["cam0", "cam1"],
            camera_shape=(480, 640, 3),
        )
        inner = ds._inner
        assert "observation.images.cam0" in inner.features
        assert "observation.images.cam1" in inner.features

    def test_create_with_explicit_features(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        features = {
            "observation.state": {"dtype": "float32", "shape": (3,)},
            "action": {"dtype": "float32", "shape": (3,)},
        }
        ds = Dataset.create("test/demo", features=features)
        assert ds._inner.features == features

    def test_summary(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        ds = Dataset.create("test/demo", state_dim=4)
        summary = ds.summary()
        assert summary["repo_id"] == "test/demo"
        assert summary["fps"] == 30
        assert "observation.state" in summary["features"]

    def test_len(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        ds = Dataset.create("test/demo")
        assert len(ds) == 0

    def test_push(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        ds = Dataset.create("test/demo")
        ds.push()
        assert ds._inner.pushed is True

    def test_push_with_repo_id(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset

        ds = Dataset.create("test/demo")
        ds.push("other/repo")
        assert ds._inner.pushed_repo_id == "other/repo"


# ---------------------------------------------------------------------------
# Recorder tests
# ---------------------------------------------------------------------------
class TestRecorder:
    def test_create_and_record_episode(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create("test/demo", state_dim=6)
        assert not recorder.is_recording

        recorder.start_episode(task="pick-place")
        assert recorder.is_recording

        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        recorder.add_frame(state=state)
        recorder.add_frame(state=state, action=state * 2)

        count = recorder.save_episode()
        assert count == 2
        assert not recorder.is_recording

        inner = recorder.dataset._inner
        assert inner.episodes_saved == 1
        assert len(inner.frames) == 2
        assert inner.frames[0][1] == "pick-place"

    def test_add_frame_with_images(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create(
            "test/demo",
            state_dim=6,
            camera_names=["cam0"],
        )
        recorder.start_episode()

        state = np.zeros(6, dtype=np.float32)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        recorder.add_frame(state=state, images={"cam0": img})

        count = recorder.save_episode()
        assert count == 1

        frame, _ = recorder.dataset._inner.frames[0]
        assert "observation.images.cam0" in frame

    def test_action_defaults_to_state(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create("test/demo", state_dim=3)
        recorder.start_episode()

        state = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        recorder.add_frame(state=state)

        frame, _ = recorder.dataset._inner.frames[0]
        np.testing.assert_array_equal(
            frame["observation.state"], frame["action"]
        )

    def test_multiple_episodes(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create("test/demo", state_dim=3)

        recorder.start_episode(task="task-a")
        recorder.add_frame(state=np.zeros(3, dtype=np.float32))
        c1 = recorder.save_episode()

        recorder.start_episode(task="task-b")
        recorder.add_frame(state=np.ones(3, dtype=np.float32))
        recorder.add_frame(state=np.ones(3, dtype=np.float32))
        c2 = recorder.save_episode()

        assert c1 == 1
        assert c2 == 2
        assert recorder.dataset._inner.episodes_saved == 2

    def test_error_add_frame_without_episode(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create("test/demo", state_dim=3)
        with pytest.raises(RuntimeError, match="No active episode"):
            recorder.add_frame(state=np.zeros(3, dtype=np.float32))

    def test_error_double_start_episode(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create("test/demo", state_dim=3)
        recorder.start_episode()
        with pytest.raises(RuntimeError, match="already active"):
            recorder.start_episode()

    def test_error_save_without_episode(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Recorder

        recorder = Recorder.create("test/demo", state_dim=3)
        with pytest.raises(RuntimeError, match="No active episode"):
            recorder.save_episode()


# ---------------------------------------------------------------------------
# Hub operations tests
# ---------------------------------------------------------------------------
class TestHub:
    def test_push_returns_url(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset
        from rfx.collection._hub import push

        ds = Dataset.create("test/demo")
        url = push(ds)
        assert "huggingface.co/datasets/test/demo" in url

    def test_push_with_custom_repo_id(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import Dataset
        from rfx.collection._hub import push

        ds = Dataset.create("test/demo")
        url = push(ds, "other/repo")
        assert "huggingface.co/datasets/other/repo" in url


# ---------------------------------------------------------------------------
# __init__ facade tests
# ---------------------------------------------------------------------------
class TestFacade:
    def test_public_api_imports(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        from rfx.collection import (
            Dataset,
            Recorder,
            collect,
            from_hub,
            open_dataset,
            pull,
            push,
        )

        assert Dataset is not None
        assert Recorder is not None
        assert callable(collect)
        assert callable(from_hub)
        assert callable(open_dataset)
        assert callable(pull)
        assert callable(push)

    def test_rfx_collection_import(self, monkeypatch):
        _install_fake_lerobot(monkeypatch)
        import rfx

        assert rfx.collection is not None


# ---------------------------------------------------------------------------
# CLI args tests
# ---------------------------------------------------------------------------
class TestCli:
    def test_add_collect_args(self):
        import argparse

        from rfx.collection._cli import add_collect_args

        parser = argparse.ArgumentParser()
        add_collect_args(parser)
        args = parser.parse_args(
            ["--robot", "so101", "--repo-id", "test/demo", "--episodes", "5", "--fps", "60"]
        )
        assert args.robot == "so101"
        assert args.repo_id == "test/demo"
        assert args.episodes == 5
        assert args.fps == 60
        assert args.push is False
        assert args.mcap is False
