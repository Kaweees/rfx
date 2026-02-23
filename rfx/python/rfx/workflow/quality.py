"""
Dataset quality gates for the collect -> validate stage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetValidationThresholds:
    max_gap_ns: int = 200_000_000
    max_alignment_error_ns: int = 50_000_000
    max_missing_frame_ratio: float = 0.02
    max_missing_step_ratio: float = 0.02


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _load_records(dataset_path: Path) -> list[dict[str, Any]]:
    if dataset_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in dataset_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if isinstance(value, dict):
                records.append(value)
        return records

    data = json.loads(dataset_path.read_text())
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        episodes = data.get("steps")
        if isinstance(episodes, list):
            return [item for item in episodes if isinstance(item, dict)]
    raise ValueError(f"Unsupported dataset format in {dataset_path}")


def validate_dataset(
    dataset_path: str | Path,
    *,
    thresholds: DatasetValidationThresholds | None = None,
) -> dict[str, Any]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    limits = thresholds or DatasetValidationThresholds()
    records = _load_records(path)

    monotonic_errors = 0
    gap_errors = 0
    alignment_errors = 0
    missing_frame = 0
    missing_obs_or_action = 0
    observed_timestamps: list[int] = []

    previous_ts: int | None = None
    for step in records:
        timestamp_ns = _to_int(step.get("timestamp_ns"))
        if timestamp_ns is None:
            missing_obs_or_action += 1
        else:
            observed_timestamps.append(timestamp_ns)
            if previous_ts is not None:
                if timestamp_ns <= previous_ts:
                    monotonic_errors += 1
                if timestamp_ns - previous_ts > limits.max_gap_ns:
                    gap_errors += 1
            previous_ts = timestamp_ns

        if "observation" not in step or "action" not in step:
            missing_obs_or_action += 1

        camera_ts = _to_int(step.get("camera_timestamp_ns"))
        control_ts = _to_int(step.get("control_timestamp_ns"))
        if camera_ts is None:
            missing_frame += 1
        if camera_ts is not None and control_ts is not None:
            if abs(camera_ts - control_ts) > limits.max_alignment_error_ns:
                alignment_errors += 1

    step_count = max(len(records), 1)
    missing_frame_ratio = missing_frame / step_count
    missing_step_ratio = missing_obs_or_action / step_count

    checks = {
        "timestamp_monotonicity": monotonic_errors == 0,
        "timestamp_gap_budget": gap_errors == 0,
        "camera_control_alignment": alignment_errors == 0,
        "missing_frame_ratio": missing_frame_ratio <= limits.max_missing_frame_ratio,
        "missing_step_ratio": missing_step_ratio <= limits.max_missing_step_ratio,
        "observation_action_schema": missing_obs_or_action == 0,
    }
    passed = all(checks.values())
    defect_counts = {
        "monotonicity_violations": monotonic_errors,
        "gap_violations": gap_errors,
        "alignment_violations": alignment_errors,
        "missing_frame_steps": missing_frame,
        "missing_obs_action_steps": missing_obs_or_action,
    }
    return {
        "schema_version": "1.0",
        "dataset_path": str(path),
        "step_count": len(records),
        "checks": checks,
        "defect_counts": defect_counts,
        "thresholds": {
            "max_gap_ns": limits.max_gap_ns,
            "max_alignment_error_ns": limits.max_alignment_error_ns,
            "max_missing_frame_ratio": limits.max_missing_frame_ratio,
            "max_missing_step_ratio": limits.max_missing_step_ratio,
        },
        "stats": {
            "min_timestamp_ns": min(observed_timestamps) if observed_timestamps else None,
            "max_timestamp_ns": max(observed_timestamps) if observed_timestamps else None,
            "missing_frame_ratio": missing_frame_ratio,
            "missing_step_ratio": missing_step_ratio,
        },
        "passed": passed,
    }
