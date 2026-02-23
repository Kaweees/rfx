"""
rfx.tf - Transform tree (tf2-equivalent).

Thread-safe transform buffer with BFS frame graph, timestamp interpolation,
broadcaster, and listener. All via pub/sub on rfx/tf/{parent}/{child}.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformStamped:
    parent_frame: str
    child_frame: str
    timestamp_ns: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # (x, y, z, w) quaternion

    def to_dict(self) -> dict:
        return {
            "parent_frame": self.parent_frame,
            "child_frame": self.child_frame,
            "timestamp_ns": self.timestamp_ns,
            "translation": list(self.translation),
            "rotation": list(self.rotation),
            "_schema_version": "1.0",
        }

    @staticmethod
    def from_dict(d: dict) -> TransformStamped:
        return TransformStamped(
            parent_frame=d["parent_frame"],
            child_frame=d["child_frame"],
            timestamp_ns=d["timestamp_ns"],
            translation=tuple(d["translation"]),
            rotation=tuple(d["rotation"]),
        )

    @staticmethod
    def identity(parent: str, child: str) -> TransformStamped:
        return TransformStamped(
            parent_frame=parent,
            child_frame=child,
            timestamp_ns=time.time_ns(),
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )


def _quat_multiply(q1: tuple, q2: tuple) -> tuple:
    """Hamilton product: q1 * q2 where q = (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _quat_inverse(q: tuple) -> tuple:
    """Inverse of unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return (-x, -y, -z, w)


def _quat_rotate(q: tuple, v: tuple) -> tuple:
    """Rotate vector v by quaternion q."""
    x, y, z, w = q
    vx, vy, vz = v
    # q * v * q_inv
    t = (
        2.0 * (y * vz - z * vy),
        2.0 * (z * vx - x * vz),
        2.0 * (x * vy - y * vx),
    )
    return (
        vx + w * t[0] + y * t[2] - z * t[1],
        vy + w * t[1] + z * t[0] - x * t[2],
        vz + w * t[2] + x * t[1] - y * t[0],
    )


def _compose_transforms(parent: TransformStamped, child: TransformStamped) -> TransformStamped:
    """Compose parent * child transforms."""
    rotated = _quat_rotate(parent.rotation, child.translation)
    translation = (
        parent.translation[0] + rotated[0],
        parent.translation[1] + rotated[1],
        parent.translation[2] + rotated[2],
    )
    rotation = _quat_multiply(parent.rotation, child.rotation)
    return TransformStamped(
        parent_frame=parent.parent_frame,
        child_frame=child.child_frame,
        timestamp_ns=max(parent.timestamp_ns, child.timestamp_ns),
        translation=translation,
        rotation=rotation,
    )


class TransformBuffer:
    """Thread-safe buffer of transforms with frame graph and BFS lookup."""

    def __init__(self, max_history: int = 100) -> None:
        self._lock = threading.Lock()
        self._transforms: dict[tuple[str, str], deque[TransformStamped]] = {}
        self._max_history = max_history

    def set_transform(self, tf: TransformStamped) -> None:
        key = (tf.parent_frame, tf.child_frame)
        with self._lock:
            if key not in self._transforms:
                self._transforms[key] = deque(maxlen=self._max_history)
            self._transforms[key].append(tf)

    def _get_latest(self, parent: str, child: str) -> TransformStamped | None:
        """Get latest transform for a direct edge (caller holds lock)."""
        key = (parent, child)
        if key in self._transforms and self._transforms[key]:
            return self._transforms[key][-1]
        # Try inverse
        inv_key = (child, parent)
        if inv_key in self._transforms and self._transforms[inv_key]:
            tf = self._transforms[inv_key][-1]
            inv_rot = _quat_inverse(tf.rotation)
            inv_trans = _quat_rotate(inv_rot, tuple(-x for x in tf.translation))
            return TransformStamped(
                parent_frame=parent,
                child_frame=child,
                timestamp_ns=tf.timestamp_ns,
                translation=inv_trans,
                rotation=inv_rot,
            )
        return None

    def lookup(self, target_frame: str, source_frame: str) -> TransformStamped | None:
        """BFS lookup from source_frame to target_frame."""
        if target_frame == source_frame:
            return TransformStamped.identity(target_frame, source_frame)

        with self._lock:
            # Build adjacency from known edges
            adj: dict[str, set[str]] = {}
            for p, c in self._transforms:
                adj.setdefault(p, set()).add(c)
                adj.setdefault(c, set()).add(p)

            # BFS from source to target
            visited = {source_frame}
            queue = deque([(source_frame, [])])

            while queue:
                current, path = queue.popleft()
                for neighbor in adj.get(current, set()):
                    if neighbor in visited:
                        continue
                    new_path = path + [(current, neighbor)]
                    if neighbor == target_frame:
                        # Compose chain
                        result = TransformStamped.identity(source_frame, source_frame)
                        for p, c in new_path:
                            tf = self._get_latest(p, c)
                            if tf is None:
                                return None
                            result = _compose_transforms(result, tf)
                        return TransformStamped(
                            parent_frame=target_frame,
                            child_frame=source_frame,
                            timestamp_ns=result.timestamp_ns,
                            translation=result.translation,
                            rotation=result.rotation,
                        )
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def all_frames(self) -> list[str]:
        with self._lock:
            frames = set()
            for p, c in self._transforms:
                frames.add(p)
                frames.add(c)
            return sorted(frames)


class TransformBroadcaster:
    """Publishes transforms to transport on rfx/tf/{parent}/{child}."""

    def __init__(self, transport: Any, buffer: TransformBuffer | None = None) -> None:
        self._transport = transport
        self._buffer = buffer

    def send_transform(self, tf: TransformStamped) -> None:
        import json

        key = f"rfx/tf/{tf.parent_frame}/{tf.child_frame}"
        self._transport.publish(key, json.dumps(tf.to_dict()))
        if self._buffer is not None:
            self._buffer.set_transform(tf)


class StaticTransformBroadcaster(TransformBroadcaster):
    """Broadcasts latched/static transforms with periodic republish."""

    def __init__(
        self,
        transport: Any,
        buffer: TransformBuffer | None = None,
        republish_interval_s: float = 10.0,
    ) -> None:
        super().__init__(transport, buffer)
        self._static_transforms: list[TransformStamped] = []
        self._interval = republish_interval_s
        self._running = True
        self._thread = threading.Thread(target=self._republish_loop, daemon=True)
        self._thread.start()

    def send_transform(self, tf: TransformStamped) -> None:
        self._static_transforms.append(tf)
        super().send_transform(tf)

    def _republish_loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            for tf in list(self._static_transforms):
                super().send_transform(tf)

    def stop(self) -> None:
        self._running = False


class TransformListener:
    """Subscribes to rfx/tf/** and populates a TransformBuffer."""

    def __init__(self, transport: Any, buffer: TransformBuffer) -> None:
        self._transport = transport
        self._buffer = buffer
        self._sub = transport.subscribe("rfx/tf/**", capacity=256)
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self) -> None:
        import json

        while self._running:
            env = self._sub.recv(timeout_s=0.5)
            if env is None:
                continue
            try:
                raw = env.payload
                if isinstance(raw, memoryview):
                    raw = bytes(raw)
                data = json.loads(raw)
                tf = TransformStamped.from_dict(data)
                self._buffer.set_transform(tf)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False


def broadcast_urdf_transforms(
    urdf: Any,
    joint_positions: dict[str, float] | list[float],
    broadcaster: TransformBroadcaster,
) -> None:
    """Broadcast all transforms from a URDF model given joint positions."""
    fk = urdf.forward_kinematics(
        joint_positions if isinstance(joint_positions, list) else list(joint_positions.values())
    )
    for joint in urdf.joints:
        if not joint.is_actuated and joint.joint_type != "fixed":
            continue
        parent = joint.parent
        child = joint.child
        if child in fk:
            m = fk[child]
            # Extract translation from 4x4 matrix
            tx, ty, tz = m[0][3], m[1][3], m[2][3]
            # Extract rotation as quaternion from rotation matrix
            # Simple extraction (not optimized)
            trace = m[0][0] + m[1][1] + m[2][2]
            if trace > 0:
                s = 0.5 / math.sqrt(trace + 1.0)
                w = 0.25 / s
                x = (m[2][1] - m[1][2]) * s
                y = (m[0][2] - m[2][0]) * s
                z = (m[1][0] - m[0][1]) * s
            else:
                w, x, y, z = 0.0, 0.0, 0.0, 1.0

            tf = TransformStamped(
                parent_frame=parent,
                child_frame=child,
                timestamp_ns=time.time_ns(),
                translation=(tx, ty, tz),
                rotation=(x, y, z, w),
            )
            broadcaster.send_transform(tf)
