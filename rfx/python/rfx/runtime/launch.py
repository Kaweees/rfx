from __future__ import annotations

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


@dataclass
class LaunchSpec:
    name: str
    backend: str = "mock"
    profile: str = "default"
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    nodes: list[LaunchNode] = field(default_factory=list)


def load_launch_file(path: str | Path) -> LaunchSpec:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Launch file must be a mapping")

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
            )
        )
    return LaunchSpec(
        name=str(data.get("name", path.stem)),
        backend=str(data.get("backend", "mock")),
        profile=str(data.get("profile", "default")),
        profiles=data.get("profiles", {}) or {},
        nodes=nodes,
    )
