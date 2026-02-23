from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path


class DoraCliError(RuntimeError):
    """Raised when Dora CLI execution fails or is unavailable."""


def _dora_bin() -> str:
    dora_bin = shutil.which("dora")
    if dora_bin is None:
        raise DoraCliError("Dora CLI not found in PATH. Install with: pip install dora-rs-cli")
    return dora_bin


def build_dataflow(dataflow: str | Path, *, uv: bool = True) -> int:
    """
    Build a Dora dataflow file.

    Equivalent to: `dora build <dataflow> [--uv]`.
    """
    args = [_dora_bin(), "build", str(dataflow)]
    if uv:
        args.append("--uv")
    result = subprocess.run(args, check=False)
    if result.returncode != 0:
        raise DoraCliError(f"Dora build failed with exit code {result.returncode}")
    return int(result.returncode)


def run_dataflow(
    dataflow: str | Path,
    *,
    uv: bool = True,
    env: Mapping[str, str] | None = None,
) -> int:
    """
    Run a Dora dataflow file.

    Equivalent to: `dora run <dataflow> [--uv]`.
    """
    args = [_dora_bin(), "run", str(dataflow)]
    if uv:
        args.append("--uv")
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    result = subprocess.run(args, check=False, env=merged_env)
    if result.returncode != 0:
        raise DoraCliError(f"Dora run failed with exit code {result.returncode}")
    return int(result.returncode)
