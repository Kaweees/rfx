from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _registry_path(root: Path | None = None) -> Path:
    root = root or Path.cwd()
    return root / ".rfx" / "runtime" / "graph.json"


def load_registry(root: Path | None = None) -> dict[str, Any]:
    path = _registry_path(root)
    if not path.exists():
        return {"nodes": [], "topics": {"publish": [], "subscribe": []}, "launch": None}
    return json.loads(path.read_text())


def write_registry(data: dict[str, Any], root: Path | None = None) -> None:
    path = _registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
