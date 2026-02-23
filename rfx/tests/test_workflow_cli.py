from __future__ import annotations

import json
from pathlib import Path

from rfx.runtime.cli import build_parser
from rfx.workflow.registry import list_runs


def _invoke(argv: list[str]) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.fn(ns))


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "timestamp_ns": 1_000,
            "camera_timestamp_ns": 1_010,
            "control_timestamp_ns": 1_000,
            "observation": {"state": [0.0]},
            "action": {"joint": [0.1]},
        },
        {
            "timestamp_ns": 2_000,
            "camera_timestamp_ns": 2_010,
            "control_timestamp_ns": 2_000,
            "observation": {"state": [0.0]},
            "action": {"joint": [0.1]},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_golden_path_stages_write_run_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    config = tmp_path / "robot.json"
    config.write_text(json.dumps({"name": "SO-101"}))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"success_rate": 0.9}))

    assert _invoke(["collect", "--config", str(config), "--input", str(dataset)]) == 0
    assert _invoke(["validate", "--dataset", str(dataset), "--input", str(dataset)]) == 0
    assert (
        _invoke(
            [
                "train",
                "--config",
                str(config),
                "--input",
                str(dataset),
                "--safety-profile",
                "safe-default",
            ]
        )
        == 0
    )

    train_run = list_runs(stage="train", limit=1)[0]
    artifact_ref = train_run["metadata"]["artifact_ref"]
    config_hash = train_run["artifacts"][0]["digest"]

    assert (
        _invoke(
            [
                "eval",
                "--artifact-ref",
                artifact_ref,
                "--metrics-json",
                str(metrics),
                "--min-success-rate",
                "0.5",
            ]
        )
        == 0
    )
    assert (
        _invoke(
            [
                "deploy",
                "--artifact-ref",
                artifact_ref,
                "--robot-config-hash",
                train_run["config_snapshot"]["digest"],
                "--safety-profile",
                "safe-default",
            ]
        )
        == 0
    )
    # Artifact digest and config hash are both tracked in run metadata.
    assert isinstance(config_hash, str) and len(config_hash) > 8

    runs = list_runs(limit=20)
    assert len(runs) >= 5
    run_ids = [run["run_id"] for run in runs]
    assert len(run_ids) == len(set(run_ids))


def test_deploy_gate_blocks_without_eval(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    assert _invoke(["train", "--input", str(dataset)]) == 0
    train_run = list_runs(stage="train", limit=1)[0]
    artifact_ref = train_run["metadata"]["artifact_ref"]

    rc = _invoke(["deploy", "--artifact-ref", artifact_ref])
    assert rc == 2
    deploy_run = list_runs(stage="deploy", limit=1)[0]
    assert deploy_run["status"] == "blocked"


def test_reproduce_reports_missing_inputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = tmp_path / "input.bin"
    data_path.write_bytes(b"abc")

    assert _invoke(["collect", "--input", str(data_path)]) == 0
    run_id = list_runs(stage="collect", limit=1)[0]["run_id"]
    data_path.unlink()

    rc = _invoke(["reproduce", run_id])
    assert rc == 2


def test_collect_cli_maps_collection_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    assert (
        _invoke(
            [
                "collect",
                "--repo-id",
                "local/demo",
                "--episodes",
                "2",
                "--duration",
                "0.0",
                "--task",
                "pick-place",
                "--fps",
                "60",
                "--state-dim",
                "8",
                "--collection-root",
                "datasets_out",
            ]
        )
        == 0
    )

    collect_run = list_runs(stage="collect", limit=1)[0]
    metadata = collect_run["metadata"]
    assert metadata["repo_id"] == "local/demo"
    assert metadata["episodes"] == 2
    assert metadata["duration"] == 0.0
    assert metadata["task"] == "pick-place"
    assert metadata["fps"] == 60
    assert metadata["state_dim"] == 8
    assert metadata["output"] == "datasets_out"
