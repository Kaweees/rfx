"""
rfx.teleop.mcap_writer - Export recorded teleop episodes to MCAP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class McapExportConfig:
    output_dir: Path | str = Path("mcap_exports")
    include_camera_frames: bool = True

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)


class McapEpisodeWriter:
    def __init__(self, config: McapExportConfig):
        self.config = config

    def write_episode(self, episode: Any) -> dict[str, Any]:
        try:
            from mcap.writer import Writer
        except Exception as exc:
            raise ImportError("MCAP export requires: uv pip install mcap") from exc

        output_dir = self.config.resolved_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        mcap_path = output_dir / f"{episode.episode_id}.mcap"

        control_rows = _load_jsonl(episode.episode_dir / "control.jsonl")
        if not control_rows:
            raise ValueError("Episode has no control rows to export")

        with open(mcap_path, "wb") as fh:
            writer = Writer(fh)
            writer.start()

            control_schema = writer.register_schema(
                name="rfx.teleop.control",
                encoding="jsonschema",
                data=json.dumps(_control_schema()).encode("utf-8"),
            )
            control_channel = writer.register_channel(
                topic="/rfx/teleop/control",
                message_encoding="json",
                schema_id=control_schema,
                metadata={"source": "rfx.teleop"},
            )

            cam_schema = writer.register_schema(
                name="rfx.teleop.camera_frame",
                encoding="jsonschema",
                data=json.dumps(_camera_schema()).encode("utf-8"),
            )
            cam_channel = writer.register_channel(
                topic="/rfx/teleop/camera",
                message_encoding="json",
                schema_id=cam_schema,
                metadata={"source": "rfx.teleop"},
            )

            control_count = 0
            camera_count = 0
            for row in control_rows:
                ts = int(row.get("timestamp_ns", 0))
                payload = json.dumps(row, sort_keys=True).encode("utf-8")
                writer.add_message(
                    channel_id=control_channel,
                    log_time=ts,
                    publish_time=ts,
                    data=payload,
                )
                control_count += 1

                if not self.config.include_camera_frames:
                    continue

                for camera_name, frame_idx in (row.get("camera_frame_indices") or {}).items():
                    idx = int(frame_idx)
                    if idx < 0:
                        continue
                    frame_path = (
                        episode.episode_dir / "cameras" / str(camera_name) / f"{idx:08d}.npy"
                    )
                    if not frame_path.exists():
                        continue
                    cam_msg = {
                        "camera": str(camera_name),
                        "frame_index": idx,
                        "timestamp_ns": ts,
                        "npy_path": str(frame_path.relative_to(episode.episode_dir)),
                    }
                    writer.add_message(
                        channel_id=cam_channel,
                        log_time=ts,
                        publish_time=ts,
                        data=json.dumps(cam_msg, sort_keys=True).encode("utf-8"),
                    )
                    camera_count += 1

            writer.finish()

        return {
            "episode_id": episode.episode_id,
            "mcap_path": str(mcap_path),
            "control_messages": control_count,
            "camera_messages": camera_count,
        }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _control_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "timestamp_ns": {"type": "integer"},
            "dt_s": {"type": "number"},
            "pairs": {"type": "object"},
            "camera_frame_indices": {"type": "object"},
        },
        "required": ["timestamp_ns", "dt_s", "pairs"],
    }


def _camera_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "camera": {"type": "string"},
            "frame_index": {"type": "integer"},
            "timestamp_ns": {"type": "integer"},
            "npy_path": {"type": "string"},
        },
        "required": ["camera", "frame_index", "timestamp_ns", "npy_path"],
    }
