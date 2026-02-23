"""rfx.collection._mcap — Optional MCAP sidecar logging alongside LeRobot recording."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _frame_schema() -> dict[str, Any]:
    """JSON schema for collection frame messages."""
    return {
        "type": "object",
        "properties": {
            "observation.state": {"type": "array"},
            "action": {"type": "array"},
        },
    }


class McapSidecar:
    """Optional MCAP logging alongside LeRobot recording.

    Writes a .mcap file per episode for ROS/Foxglove compatibility.
    Enabled via Recorder.create(..., mcap=True).
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None
        self._fh: Any = None
        self._episode_id: str = ""
        self._control_channel: int | None = None
        self._frame_count = 0

    def start_episode(self, episode_id: str) -> None:
        """Begin logging a new episode."""
        try:
            from mcap.writer import Writer
        except ImportError as exc:
            raise ImportError("MCAP sidecar requires: pip install mcap") from exc

        self._episode_id = episode_id
        self._frame_count = 0
        mcap_path = self._output_dir / f"{episode_id}.mcap"
        self._fh = open(mcap_path, "wb")  # noqa: SIM115
        self._writer = Writer(self._fh)
        self._writer.start()

        control_schema = self._writer.register_schema(
            name="rfx.collection.frame",
            encoding="jsonschema",
            data=json.dumps(_frame_schema()).encode("utf-8"),
        )
        self._control_channel = self._writer.register_channel(
            topic="/rfx/collection/frame",
            message_encoding="json",
            schema_id=control_schema,
            metadata={"source": "rfx.collection"},
        )

    def write_frame(self, frame: dict[str, Any]) -> None:
        """Write a single frame to the MCAP file."""
        if self._writer is None or self._control_channel is None:
            return

        ts = time.time_ns()
        serializable = {}
        for key, value in frame.items():
            if hasattr(value, "tolist"):
                serializable[key] = value.tolist()
            else:
                serializable[key] = value

        payload = json.dumps(serializable, sort_keys=True).encode("utf-8")
        self._writer.add_message(
            channel_id=self._control_channel,
            log_time=ts,
            publish_time=ts,
            data=payload,
        )
        self._frame_count += 1

    def save_episode(self) -> Path | None:
        """Finalize the current episode MCAP file."""
        if self._writer is None:
            return None

        self._writer.finish()
        mcap_path = self._output_dir / f"{self._episode_id}.mcap"

        self._writer = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        self._control_channel = None

        return mcap_path

    def close(self) -> None:
        """Close any open resources."""
        if self._writer is not None:
            self._writer.finish()
            self._writer = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None
