"""
rfx.teleop.vr - VR motion publisher for SteamVR via vuer.

Uses the actual vuer async API with ``@app.add_handler`` for HAND_MOVE
and CAMERA_MOVE events. Runs the vuer event loop in a daemon thread.

Coordinate transform: SteamVR Y-up -> robotics Z-up using the
OpenTeleVision ground-plane transform.

``vuer`` is an optional dependency -- guarded at import time.

Example:
    >>> from rfx.teleop.vr import VRMotionPublisher
    >>> vr = VRMotionPublisher()
    >>> vr.start()
    >>> poses = vr.latest_poses  # {"head": 4x4, "left_hand": 4x4, "right_hand": 4x4}
    >>> vr.stop()
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from vuer import Vuer
    from vuer.events import ClientEvent

    VUER_AVAILABLE = True
except ImportError:
    VUER_AVAILABLE = False

_POSE_DTYPE = np.float64
_IDENTITY_4x4 = np.eye(4, dtype=_POSE_DTYPE)

# OpenTeleVision Y-up to Z-up ground-plane transform
# Maps: x_zup = -z_yup, y_zup = -x_yup, z_zup = y_yup
_GRD_YUP_TO_ZUP = np.array(
    [
        [0, 0, -1, 0],
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=_POSE_DTYPE,
)


@dataclass
class VRConfig:
    """Configuration for VR motion publisher."""

    rate_hz: float = 60.0
    host: str = "0.0.0.0"
    port: int = 8012
    cert: str | None = None
    key: str | None = None
    coord_transform: bool = True


def _matrix_from_vuer(raw: list | np.ndarray) -> np.ndarray:
    """Parse a 16-element column-major matrix from vuer into a 4x4 row-major array."""
    arr = np.array(raw, dtype=_POSE_DTYPE).reshape(4, 4, order="F")
    return arr


class VRMotionPublisher:
    """Reads VR head/hand poses from SteamVR via vuer.

    The vuer server runs in a daemon thread with its own asyncio event loop.
    The control loop reads poses via ``latest_poses`` (thread-safe).

    Args:
        config: VR configuration.
        transport: Optional rfx transport to publish poses on.
    """

    def __init__(
        self,
        config: VRConfig | None = None,
        transport: Any | None = None,
    ):
        self.config = config or VRConfig()
        self._transport = transport

        self._lock = threading.Lock()
        self._poses: dict[str, np.ndarray] = {
            "head": _IDENTITY_4x4.copy(),
            "left_hand": _IDENTITY_4x4.copy(),
            "right_hand": _IDENTITY_4x4.copy(),
        }
        self._head_mat: np.ndarray = _IDENTITY_4x4.copy()

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    @property
    def latest_poses(self) -> dict[str, np.ndarray]:
        """Most recent VR poses (thread-safe copy)."""
        with self._lock:
            return {k: v.copy() for k, v in self._poses.items()}

    @property
    def head_matrix(self) -> np.ndarray:
        """Latest head pose (for yaw computation)."""
        with self._lock:
            return self._head_mat.copy()

    def start(self) -> None:
        """Start the VR publisher thread."""
        if self._thread is not None:
            return

        if not VUER_AVAILABLE:
            self._running.set()
            return

        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="vr-motion-publisher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the VR publisher thread."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run_loop(self) -> None:
        """Thread entry: create asyncio loop and run vuer server."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_main())
        except Exception as exc:
            self._error = exc
        finally:
            loop.close()

    async def _async_main(self) -> None:
        """Async entry: set up vuer app with handlers and serve."""
        vuer_kwargs = {
            "host": self.config.host,
            "port": self.config.port,
        }
        if self.config.cert:
            vuer_kwargs["cert"] = self.config.cert
        if self.config.key:
            vuer_kwargs["key"] = self.config.key

        app = Vuer(**vuer_kwargs)

        @app.add_handler("HAND_MOVE")
        async def on_hand_move(event: ClientEvent, _session):
            self._handle_hands(event)

        @app.add_handler("CAMERA_MOVE")
        async def on_camera_move(event: ClientEvent, _session):
            self._handle_head(event)

        @app.spawn(start=True)
        async def main_loop(_session):
            while self._running.is_set():
                await asyncio.sleep(1.0 / max(self.config.rate_hz, 1))

        # app.run() blocks until stopped
        await app.arun()

    def _handle_hands(self, event: Any) -> None:
        """Process HAND_MOVE event."""
        val = event.value if hasattr(event, "value") else event
        if not isinstance(val, dict):
            return

        poses: dict[str, np.ndarray] = {}

        for vuer_key, our_key in [("leftHand", "left_hand"), ("rightHand", "right_hand")]:
            raw = val.get(vuer_key)
            if raw is not None:
                mat = _matrix_from_vuer(raw)
                if self.config.coord_transform:
                    mat = _GRD_YUP_TO_ZUP @ mat
                poses[our_key] = mat

        if poses:
            with self._lock:
                self._poses.update(poses)
            self._maybe_publish(poses)

    def _handle_head(self, event: Any) -> None:
        """Process CAMERA_MOVE event."""
        val = event.value if hasattr(event, "value") else event
        if not isinstance(val, dict):
            return

        cam = val.get("camera")
        if cam is None:
            return
        raw = cam.get("matrix") if isinstance(cam, dict) else None
        if raw is None:
            return

        mat = _matrix_from_vuer(raw)

        with self._lock:
            self._head_mat = mat.copy()

        if self.config.coord_transform:
            mat = _GRD_YUP_TO_ZUP @ mat

        with self._lock:
            self._poses["head"] = mat

        self._maybe_publish({"head": mat})

    def _maybe_publish(self, poses: dict[str, np.ndarray]) -> None:
        if self._transport is None:
            return
        ts = time.time_ns()
        for name, pose in poses.items():
            self._transport.publish(
                key=f"vr/{name}/pose",
                payload=memoryview(pose.astype(np.float32)),
                timestamp_ns=ts,
                metadata={"dtype": "float32", "shape": [4, 4]},
            )

    @property
    def error(self) -> Exception | None:
        return self._error
