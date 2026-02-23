"""
Immutable run registry and lineage utilities for the rfx workflow CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

STAGES = ("collect", "validate", "train", "eval", "shadow", "deploy")
RunStatus = Literal["succeeded", "failed", "blocked"]


def resolve_workflow_root(root: Path | None = None) -> Path:
    base = root or Path.cwd()
    return base / ".rfx"


def _runs_dir(root: Path | None = None) -> Path:
    return resolve_workflow_root(root) / "runs"


def _snapshots_dir(root: Path | None = None) -> Path:
    return resolve_workflow_root(root) / "snapshots"


def _artifacts_dir(root: Path | None = None) -> Path:
    return resolve_workflow_root(root) / "artifacts"


def _ensure_layout(root: Path | None = None) -> None:
    for path in (_runs_dir(root), _snapshots_dir(root), _artifacts_dir(root)):
        path.mkdir(parents=True, exist_ok=True)


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(data: str) -> str:
    return _sha256_bytes(data.encode("utf-8"))


def generate_run_id(stage: str) -> str:
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    nonce = _sha256_bytes(os.urandom(16))[:8]
    return f"{timestamp}-{stage}-{nonce}"


def _safe_git(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_git_metadata(root: Path | None = None) -> dict[str, Any]:
    cwd = root or Path.cwd()
    commit = _safe_git(["git", "rev-parse", "HEAD"], cwd)
    branch = _safe_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status = _safe_git(["git", "status", "--porcelain"], cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def collect_environment_summary() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "zenoh_connect": os.environ.get("RFX_ZENOH_CONNECT", ""),
        "zenoh_listen": os.environ.get("RFX_ZENOH_LISTEN", ""),
        "zenoh_shared_memory": os.environ.get("RFX_ZENOH_SHARED_MEMORY", ""),
    }


def _resolve_path(value: str, root: Path | None = None) -> Path | None:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    cwd = root or Path.cwd()
    joined = cwd / value
    if joined.exists():
        return joined
    return None


def snapshot_config(config: str | None, root: Path | None = None) -> dict[str, Any] | None:
    if config is None:
        return None
    _ensure_layout(root)
    resolved = _resolve_path(config, root=root)
    if resolved is None:
        raw = config.encode("utf-8")
        source = "inline"
        source_value = config
        suffix = ".txt"
    else:
        raw = resolved.read_bytes()
        source = "file"
        source_value = str(resolved)
        suffix = resolved.suffix if resolved.suffix else ".txt"
    digest = _sha256_bytes(raw)
    snap_path = _snapshots_dir(root) / f"{digest}{suffix}"
    if not snap_path.exists():
        snap_path.write_bytes(raw)
    return {
        "source": source,
        "source_value": source_value,
        "digest": digest,
        "snapshot_path": str(snap_path),
    }


def materialize_refs(refs: list[str], root: Path | None = None) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for ref in refs:
        if ref.startswith("run:"):
            run_id = ref.split(":", 1)[1]
            run_path = _runs_dir(root) / f"{run_id}.json"
            materialized.append(
                {
                    "ref": ref,
                    "kind": "run",
                    "exists": run_path.exists(),
                    "digest": _sha256_text(ref),
                    "path": str(run_path),
                }
            )
            continue

        resolved = _resolve_path(ref, root=root)
        if resolved is not None:
            content = resolved.read_bytes()
            materialized.append(
                {
                    "ref": ref,
                    "kind": "file",
                    "exists": True,
                    "digest": _sha256_bytes(content),
                    "path": str(resolved),
                    "size_bytes": len(content),
                }
            )
            continue

        materialized.append(
            {
                "ref": ref,
                "kind": "literal",
                "exists": False,
                "digest": _sha256_text(ref),
                "path": None,
            }
        )
    return materialized


def deterministic_artifact_ref(
    stage: str,
    config_digest: str | None,
    input_digests: list[str],
    output_digests: list[str],
) -> tuple[str, str]:
    seed = {
        "stage": stage,
        "config_digest": config_digest,
        "inputs": sorted(input_digests),
        "outputs": sorted(output_digests),
    }
    digest = _sha256_text(json.dumps(seed, sort_keys=True))
    return f"artifact://{stage}/sha256:{digest}", digest


def write_artifact_manifest(
    *,
    digest: str,
    manifest: dict[str, Any],
    root: Path | None = None,
) -> str:
    _ensure_layout(root)
    path = _artifacts_dir(root) / f"{digest}.json"
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return str(path)


def create_run_record(
    *,
    run_id: str,
    stage: str,
    status: RunStatus,
    invocation_argv: list[str],
    config_snapshot_data: dict[str, Any] | None,
    input_refs: list[dict[str, Any]],
    output_refs: list[dict[str, Any]],
    metadata: dict[str, Any],
    reports: list[str],
    artifacts: list[dict[str, Any]],
    root: Path | None = None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported workflow stage {stage!r}")
    _ensure_layout(root)
    started = _utc_now_iso()
    upstream_run_ids = [
        entry["ref"].split(":", 1)[1]
        for entry in input_refs
        if entry.get("kind") == "run" and isinstance(entry.get("ref"), str)
    ]
    record = {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "started_at": started,
        "finished_at": _utc_now_iso(),
        "invocation": {
            "argv": invocation_argv,
            "command": " ".join(shlex.quote(part) for part in invocation_argv),
            "cwd": str(root or Path.cwd()),
        },
        "git": collect_git_metadata(root=root),
        "environment": collect_environment_summary(),
        "config_snapshot": config_snapshot_data,
        "input_refs": input_refs,
        "output_refs": output_refs,
        "upstream_run_ids": upstream_run_ids,
        "reports": reports,
        "artifacts": artifacts,
        "metadata": metadata,
    }
    path = _runs_dir(root) / f"{run_id}.json"
    if path.exists():
        raise FileExistsError(f"Run record already exists: {run_id}")
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return record


def _run_path(run_id: str, root: Path | None = None) -> Path:
    return _runs_dir(root) / f"{run_id}.json"


def load_run(run_id: str, root: Path | None = None) -> dict[str, Any]:
    path = _run_path(run_id, root=root)
    if not path.exists():
        raise FileNotFoundError(f"Unknown run_id: {run_id}")
    return json.loads(path.read_text())


def list_runs(
    *,
    root: Path | None = None,
    stage: str | None = None,
    status: RunStatus | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = _runs_dir(root)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for run_file in sorted(path.glob("*.json"), reverse=True):
        data = json.loads(run_file.read_text())
        if stage is not None and data.get("stage") != stage:
            continue
        if status is not None and data.get("status") != status:
            continue
        records.append(data)
        if limit is not None and len(records) >= limit:
            break
    return records


def build_lineage(run_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    lineage: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack: list[str] = [run_id]
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.add(current)
        record = load_run(current, root=root)
        lineage.append(record)
        for parent in record.get("upstream_run_ids", []):
            if isinstance(parent, str) and parent not in seen:
                stack.append(parent)
    return lineage


def build_reproduce_context(run_id: str, root: Path | None = None) -> dict[str, Any]:
    record = load_run(run_id, root=root)
    missing: list[str] = []
    for ref in record.get("input_refs", []):
        if ref.get("kind") == "file":
            path = ref.get("path")
            if not isinstance(path, str) or not Path(path).exists():
                missing.append(f"missing input file: {path or ref.get('ref')}")
        if ref.get("kind") == "run":
            dep = ref.get("ref", "")
            if isinstance(dep, str):
                dep_id = dep.split(":", 1)[1] if dep.startswith("run:") else dep
                if not _run_path(dep_id, root=root).exists():
                    missing.append(f"missing upstream run record: {dep_id}")
    snapshot = record.get("config_snapshot")
    if isinstance(snapshot, dict):
        snap_path = snapshot.get("snapshot_path")
        if isinstance(snap_path, str) and not Path(snap_path).exists():
            missing.append(f"missing config snapshot: {snap_path}")
    return {
        "run_id": run_id,
        "stage": record.get("stage"),
        "command": record.get("invocation", {}).get("command", ""),
        "argv": record.get("invocation", {}).get("argv", []),
        "missing_dependencies": missing,
    }
