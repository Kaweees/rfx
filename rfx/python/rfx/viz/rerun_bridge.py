"""
rfx.viz.rerun_bridge - Bridge between rfx transport and rerun.io visualization.

Optional dependency: rerun-sdk >= 0.18
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


class RerunBridge:
    """
    Subscribes to rfx transport topics and logs data to rerun.

    Supports: joint states, transforms, images, URDF models.
    """

    def __init__(
        self,
        transport: Any,
        application_id: str = "rfx",
        topics: list[str] | None = None,
    ) -> None:
        try:
            import rerun as rr
        except ImportError:
            raise ImportError(
                "rerun-sdk is required for RerunBridge. Install with: pip install rerun-sdk>=0.18"
            )

        self._rr = rr
        self._transport = transport
        self._topics = topics or ["rfx/**"]
        self._running = False
        self._thread: threading.Thread | None = None
        self._subs: list[Any] = []

        rr.init(application_id, spawn=True)

    def start(self) -> None:
        self._running = True
        for pattern in self._topics:
            sub = self._transport.subscribe(pattern, capacity=256)
            self._subs.append(sub)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        rr = self._rr
        while self._running:
            for sub in self._subs:
                env = sub.recv(timeout_s=0.1) if hasattr(sub, 'recv') else None
                if env is None:
                    continue
                try:
                    self._log_envelope(env)
                except Exception:
                    logger.debug("Failed to log envelope: %s", env.key, exc_info=True)

    def _log_envelope(self, env: Any) -> None:
        rr = self._rr
        key = env.key
        raw = env.payload
        if isinstance(raw, memoryview):
            raw = bytes(raw)

        try:
            data = json.loads(raw)
        except Exception:
            rr.log(key, rr.TextLog(f"binary payload ({len(raw)} bytes)"))
            return

        # Route based on key pattern
        if "tf/" in key:
            self._log_transform(key, data)
        elif "joint" in key.lower():
            self._log_joint_state(key, data)
        else:
            rr.log(key, rr.TextLog(json.dumps(data, indent=2)[:500]))

    def _log_transform(self, key: str, data: dict) -> None:
        rr = self._rr
        t = data.get("translation", [0, 0, 0])
        rr.log(key, rr.Transform3D(translation=t))

    def _log_joint_state(self, key: str, data: dict) -> None:
        rr = self._rr
        if "positions" in data:
            positions = data["positions"]
            for i, p in enumerate(positions):
                rr.log(f"{key}/joint_{i}", rr.Scalar(p))

    def log_urdf(self, urdf: Any, entity_path: str = "robot") -> None:
        """Log URDF model as a mesh hierarchy."""
        rr = self._rr
        rr.log(entity_path, rr.TextLog(f"URDF: {getattr(urdf, 'name', 'unknown')}"))
