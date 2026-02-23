from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class LaunchNode:
    package: str
    node: str
    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    rate_hz: float = 50.0
    max_steps: int | None = None
    remap: dict[str, str] = field(default_factory=dict)
    condition: str | None = None
    managed: bool = False


@dataclass
class LaunchInclude:
    file: str
    condition: str | None = None


@dataclass
class LaunchSpec:
    name: str
    backend: str = "mock"
    profile: str = "default"
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes: list[LaunchNode] = field(default_factory=list)
    includes: list[LaunchInclude] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _substitute_env(value: str) -> str:
    """Substitute ${VAR} or ${VAR:default} patterns with environment variables."""

    def _replace(m: re.Match) -> str:
        var_name = m.group(1)
        default = m.group(2)
        return os.environ.get(var_name, default if default is not None else m.group(0))

    return _ENV_PATTERN.sub(_replace, value)


def evaluate_condition(condition: str | None) -> bool:
    """Evaluate a launch condition string. Supports ${VAR} substitution."""
    if condition is None:
        return True
    resolved = _substitute_env(condition)
    resolved = resolved.strip().lower()
    if resolved in ("true", "1", "yes"):
        return True
    if resolved in ("false", "0", "no", ""):
        return False
    return bool(resolved)


def load_launch_file(path: str | Path) -> LaunchSpec:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Launch file must be a mapping")

    # Process includes
    includes = []
    for item in data.get("includes", []):
        if isinstance(item, str):
            includes.append(LaunchInclude(file=item))
        elif isinstance(item, dict):
            includes.append(
                LaunchInclude(
                    file=item["file"],
                    condition=item.get("condition"),
                )
            )

    nodes = []
    for item in data.get("nodes", []):
        nodes.append(
            LaunchNode(
                package=item["package"],
                node=item["node"],
                name=item.get("name"),
                params=item.get("params", {}) or {},
                rate_hz=float(item.get("rate_hz", 50.0)),
                max_steps=int(item["max_steps"]) if "max_steps" in item else None,
                remap=item.get("remap", {}) or {},
                condition=item.get("condition"),
                managed=bool(item.get("managed", False)),
            )
        )
    return LaunchSpec(
        name=str(data.get("name", path.stem)),
        backend=str(data.get("backend", "mock")),
        profile=str(data.get("profile", "default")),
        profiles=data.get("profiles", {}) or {},
        nodes=nodes,
        includes=includes,
        env=data.get("env", {}) or {},
    )
