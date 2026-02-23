"""
Stage execution helpers for the workflow CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .quality import validate_dataset
from .registry import (
    deterministic_artifact_ref,
    list_runs,
    load_run,
    write_artifact_manifest,
)


@dataclass(frozen=True)
class StageResult:
    status: str
    metadata: dict[str, Any]
    reports: list[str]
    artifacts: list[dict[str, Any]]
    generated_outputs: list[str]
    message: str = ""


def _now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _reports_dir(root: Path, stage: str) -> Path:
    path = root / ".rfx" / "reports" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected object JSON in {path}")
    return value


def _find_passed_eval_run(root: Path, artifact_ref: str) -> dict[str, Any] | None:
    for run in list_runs(root=root, stage="eval", status="succeeded"):
        metadata = run.get("metadata", {})
        if metadata.get("artifact_ref") != artifact_ref:
            continue
        report_path = metadata.get("eval_report_path")
        if not isinstance(report_path, str):
            continue
        if Path(report_path).exists():
            report = json.loads(Path(report_path).read_text())
            if bool(report.get("passed")):
                return run
    return None


def _find_passed_shadow_run(root: Path, artifact_ref: str) -> dict[str, Any] | None:
    for run in list_runs(root=root, stage="shadow", status="succeeded"):
        metadata = run.get("metadata", {})
        if metadata.get("artifact_ref") != artifact_ref:
            continue
        report_path = metadata.get("shadow_report_path")
        if not isinstance(report_path, str):
            continue
        if Path(report_path).exists():
            report = json.loads(Path(report_path).read_text())
            if bool(report.get("passed")):
                return run
    return None


def _load_artifact_manifest(root: Path, artifact_ref: str) -> dict[str, Any] | None:
    if not artifact_ref.startswith("artifact://"):
        return None
    marker = "sha256:"
    if marker not in artifact_ref:
        return None
    digest = artifact_ref.split(marker, 1)[1]
    path = root / ".rfx" / "artifacts" / f"{digest}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return None
    return data


def _resolve_artifact_ref(metadata: dict[str, Any], root: Path) -> str | None:
    value = metadata.get("artifact_ref")
    if isinstance(value, str) and value:
        return value
    artifact_run = metadata.get("artifact_run_id")
    if isinstance(artifact_run, str) and artifact_run:
        run = load_run(artifact_run, root=root)
        candidate = run.get("metadata", {}).get("artifact_ref")
        if isinstance(candidate, str):
            return candidate
    return None


def _run_collect_stage(metadata: dict[str, Any]) -> dict[str, Any]:
    """Run collection through the public rfx.collection API when configured."""
    try:
        from ..collection import collect

        repo_id = metadata.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            return {"collection_skipped": True, "reason": "repo_id not provided"}

        robot_type = str(metadata.get("robot_type", "so101"))
        output = str(metadata.get("output", "datasets"))
        episodes = int(metadata.get("episodes", 1))
        duration_s_raw = metadata.get("duration", None)
        duration_s = float(duration_s_raw) if duration_s_raw is not None else None
        task = str(metadata.get("task", "default"))
        fps = int(metadata.get("fps", 30))
        state_dim = int(metadata.get("state_dim", 6))
        push_to_hub = bool(metadata.get("push_to_hub", False))
        mcap = bool(metadata.get("mcap", False))

        dataset = collect(
            robot_type=robot_type,
            repo_id=repo_id,
            output=output,
            episodes=episodes,
            duration_s=duration_s,
            task=task,
            fps=fps,
            state_dim=state_dim,
            push_to_hub=push_to_hub,
            mcap=mcap,
        )
        return {
            "repo_id": repo_id,
            "root": output,
            "episodes_collected": int(episodes),
            "num_episodes": int(dataset.num_episodes),
            "num_frames": int(dataset.num_frames),
        }
    except Exception as exc:
        return {"collection_skipped": True, "error": str(exc)}


def execute_stage(
    *,
    stage: str,
    run_id: str,
    root: Path,
    config_snapshot_data: dict[str, Any] | None,
    input_refs: list[dict[str, Any]],
    output_refs: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> StageResult:
    if stage == "collect":
        collect_result = _run_collect_stage(metadata)
        return StageResult(
            status="succeeded",
            metadata={**metadata, **collect_result},
            reports=[],
            artifacts=[],
            generated_outputs=[],
            message="collect completed",
        )

    if stage == "validate":
        dataset = metadata.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            for entry in input_refs:
                if entry.get("kind") == "file" and isinstance(entry.get("path"), str):
                    dataset = entry["path"]
                    break
        if not isinstance(dataset, str) or not dataset:
            return StageResult(
                status="failed",
                metadata={**metadata, "error": "dataset input is required"},
                reports=[],
                artifacts=[],
                generated_outputs=[],
                message="dataset input is required for validate",
            )
        report = validate_dataset(dataset)
        report["run_id"] = run_id
        report["generated_at"] = _now_iso()
        report_path = _reports_dir(root, "validate") / f"{run_id}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        status = "succeeded" if report.get("passed") else "failed"
        return StageResult(
            status=status,
            metadata={
                **metadata,
                "validate_report_path": str(report_path),
                "validate_passed": bool(report.get("passed")),
            },
            reports=[str(report_path)],
            artifacts=[],
            generated_outputs=[str(report_path)],
            message="validate completed",
        )

    if stage == "train":
        config_digest = config_snapshot_data.get("digest") if config_snapshot_data else None
        input_digests = [str(entry.get("digest", "")) for entry in input_refs]
        output_digests = [str(entry.get("digest", "")) for entry in output_refs]
        artifact_ref, artifact_digest = deterministic_artifact_ref(
            "policy",
            config_digest=config_digest,
            input_digests=input_digests,
            output_digests=output_digests,
        )
        manifest = {
            "schema_version": "1.0",
            "artifact_ref": artifact_ref,
            "artifact_type": "policy_package",
            "created_at": _now_iso(),
            "lineage": {
                "run_id": run_id,
                "input_digests": sorted(input_digests),
                "output_digests": sorted(output_digests),
            },
            "compatibility": {
                "config_hash": config_digest,
                "safety_profile": metadata.get("safety_profile"),
                "robot_type": metadata.get("robot_type"),
            },
        }
        artifact_path = write_artifact_manifest(
            digest=artifact_digest,
            manifest=manifest,
            root=root,
        )
        artifact_entry = {
            "ref": artifact_ref,
            "digest": artifact_digest,
            "path": artifact_path,
        }
        return StageResult(
            status="succeeded",
            metadata={**metadata, "artifact_ref": artifact_ref, "artifact_path": artifact_path},
            reports=[],
            artifacts=[artifact_entry],
            generated_outputs=[artifact_path],
            message="train completed",
        )

    if stage == "eval":
        artifact_ref = _resolve_artifact_ref(metadata, root)
        if artifact_ref is None:
            return StageResult(
                status="failed",
                metadata={**metadata, "error": "artifact_ref is required for eval"},
                reports=[],
                artifacts=[],
                generated_outputs=[],
                message="artifact_ref is required for eval",
            )
        metrics = _read_json_file(metadata.get("metrics_json"))
        min_success_rate = float(metadata.get("min_success_rate", 0.0))
        success_rate = float(metrics.get("success_rate", 1.0))
        passed = success_rate >= min_success_rate
        report = {
            "schema_version": "1.0",
            "run_id": run_id,
            "artifact_ref": artifact_ref,
            "generated_at": _now_iso(),
            "metrics": metrics,
            "thresholds": {"min_success_rate": min_success_rate},
            "passed": passed,
        }
        report_path = _reports_dir(root, "eval") / f"{run_id}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return StageResult(
            status="succeeded" if passed else "failed",
            metadata={
                **metadata,
                "artifact_ref": artifact_ref,
                "eval_report_path": str(report_path),
                "eval_passed": passed,
            },
            reports=[str(report_path)],
            artifacts=[],
            generated_outputs=[str(report_path)],
            message="eval completed",
        )

    if stage == "shadow":
        artifact_ref = _resolve_artifact_ref(metadata, root)
        if artifact_ref is None:
            return StageResult(
                status="failed",
                metadata={**metadata, "error": "artifact_ref is required for shadow"},
                reports=[],
                artifacts=[],
                generated_outputs=[],
                message="artifact_ref is required for shadow",
            )
        data = _read_json_file(metadata.get("shadow_json"))
        max_policy_delta = float(metadata.get("max_policy_delta", 1.0))
        observed_delta = float(data.get("mean_policy_delta", 0.0))
        passed = observed_delta <= max_policy_delta
        report = {
            "schema_version": "1.0",
            "run_id": run_id,
            "artifact_ref": artifact_ref,
            "generated_at": _now_iso(),
            "policy_vs_executed": data,
            "thresholds": {"max_policy_delta": max_policy_delta},
            "passed": passed,
        }
        report_path = _reports_dir(root, "shadow") / f"{run_id}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return StageResult(
            status="succeeded" if passed else "failed",
            metadata={
                **metadata,
                "artifact_ref": artifact_ref,
                "shadow_report_path": str(report_path),
                "shadow_passed": passed,
            },
            reports=[str(report_path)],
            artifacts=[],
            generated_outputs=[str(report_path)],
            message="shadow completed",
        )

    if stage == "deploy":
        artifact_ref = _resolve_artifact_ref(metadata, root)
        errors: list[str] = []
        if artifact_ref is None:
            errors.append("artifact_ref is required")
        artifact = _load_artifact_manifest(root, artifact_ref or "")
        if artifact_ref is not None and artifact is None:
            errors.append(f"artifact manifest not found for {artifact_ref}")
        if artifact is not None and artifact.get("schema_version") != "1.0":
            errors.append("artifact schema_version is unsupported")

        if artifact_ref is not None:
            eval_run = _find_passed_eval_run(root, artifact_ref)
            if eval_run is None:
                errors.append("no successful eval report found for artifact")

        require_shadow = bool(metadata.get("require_shadow", False))
        if require_shadow and artifact_ref is not None:
            shadow_run = _find_passed_shadow_run(root, artifact_ref)
            if shadow_run is None:
                errors.append("shadow gate requested but no passing shadow report found")

        expected_config_hash = metadata.get("robot_config_hash")
        if isinstance(expected_config_hash, str) and artifact is not None:
            actual_hash = artifact.get("compatibility", {}).get("config_hash")
            if actual_hash != expected_config_hash:
                errors.append(
                    "artifact config hash mismatch: "
                    f"expected={expected_config_hash} actual={actual_hash}"
                )

        expected_profile = metadata.get("safety_profile")
        if isinstance(expected_profile, str) and artifact is not None:
            actual_profile = artifact.get("compatibility", {}).get("safety_profile")
            if actual_profile != expected_profile:
                errors.append(
                    "artifact safety profile mismatch: "
                    f"expected={expected_profile} actual={actual_profile}"
                )

        report = {
            "schema_version": "1.0",
            "run_id": run_id,
            "artifact_ref": artifact_ref,
            "generated_at": _now_iso(),
            "allowed": len(errors) == 0,
            "errors": errors,
        }
        report_path = _reports_dir(root, "deploy") / f"{run_id}.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return StageResult(
            status="succeeded" if not errors else "blocked",
            metadata={
                **metadata,
                "artifact_ref": artifact_ref,
                "deploy_report_path": str(report_path),
                "deploy_allowed": len(errors) == 0,
            },
            reports=[str(report_path)],
            artifacts=[],
            generated_outputs=[str(report_path)],
            message="deploy preflight completed",
        )

    raise ValueError(f"Unsupported stage: {stage}")
