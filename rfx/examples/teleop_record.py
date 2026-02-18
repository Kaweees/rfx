#!/usr/bin/env python3
"""
Record teleoperation episodes with the SO-101 arm.

Uses rfx.teleop to run a high-rate control loop between leader and
follower arms, optionally capturing camera frames and exporting to
LeRobot or MCAP format.

Key concepts:
    rfx.BimanualSo101Session — high-rate teleop session with async cameras
    rfx.TeleopSessionConfig  — rate, output dir, arm pairs, cameras
    rfx.ArmPairConfig        — leader/follower port pair
    rfx.CameraStreamConfig   — camera device, resolution, fps
    session.start_recording() / stop_recording() — episode boundaries

Configs (rfx/configs/):
    so101_bimanual.yaml  — bimanual setup with camera list

Export formats (--export-format):
    none     — raw episode only (default)
    lerobot  — LeRobot HuggingFace dataset
    mcap     — MCAP log format
    both     — both formats

Usage:
    # Single arm pair
    uv run rfx/examples/teleop_record.py --leader /dev/ttyACM0 --follower /dev/ttyACM1

    # Bimanual from config
    uv run rfx/examples/teleop_record.py --config rfx/configs/so101_bimanual.yaml

    # With LeRobot export
    uv run rfx/examples/teleop_record.py --config rfx/configs/so101_bimanual.yaml \
        --export-format lerobot --lerobot-repo-id your/repo
"""

import argparse
import threading
import time
from pathlib import Path

import yaml

from rfx.teleop import ArmPairConfig, BimanualSo101Session, CameraStreamConfig, TeleopSessionConfig


def _config_from_yaml(path: str | Path) -> TeleopSessionConfig:
    """Load a TeleopSessionConfig from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    pairs = tuple(
        ArmPairConfig(
            name=str(p["name"]),
            leader_port=str(p["leader_port"]),
            follower_port=str(p["follower_port"]),
        )
        for p in (data.get("arm_pairs") or [])
    )
    cameras = tuple(
        CameraStreamConfig(
            name=str(c.get("name", f"cam{i}")),
            device_id=c.get("device_id", i),
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            fps=int(c.get("fps", 30)),
        )
        for i, c in enumerate(data.get("cameras") or [])
    )
    if not pairs:
        raise ValueError(f"No arm_pairs defined in config: {path}")

    return TeleopSessionConfig(
        rate_hz=float(data.get("rate_hz", 350.0)),
        output_dir=Path(data.get("output_dir", "demos")),
        arm_pairs=pairs,
        cameras=cameras,
    )


def main():
    parser = argparse.ArgumentParser(description="Record teleop episodes")
    parser.add_argument("--config", default=None, help="YAML config file (overrides port args)")
    parser.add_argument("--leader", default="/dev/ttyACM0", help="Leader arm serial port")
    parser.add_argument("--follower", default="/dev/ttyACM1", help="Follower arm serial port")
    parser.add_argument("--right-leader", default=None, help="Right leader port (bimanual)")
    parser.add_argument("--right-follower", default=None, help="Right follower port (bimanual)")
    parser.add_argument("--rate-hz", type=float, default=350.0, help="Control loop Hz")
    parser.add_argument("--camera-ids", default="0,1,2", help="Comma-separated camera device IDs")
    parser.add_argument("--camera-fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--label", default="teleop", help="Episode label")
    parser.add_argument("--export-format", default="none", choices=["none", "lerobot", "mcap", "both"])
    parser.add_argument("--lerobot-repo-id", default=None, help="HuggingFace repo ID for LeRobot export")
    parser.add_argument("--lerobot-root", default="lerobot_datasets")
    parser.add_argument("--lerobot-task", default=None)
    parser.add_argument("--mcap-root", default="mcap_exports")
    parser.add_argument("--mcap-no-camera-frames", action="store_true")
    parser.add_argument("--output", default="demos", help="Output directory for episodes")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 1. Build session config
    # -------------------------------------------------------------------
    if args.config:
        config = _config_from_yaml(args.config)
        session = BimanualSo101Session(config=config)
    else:
        camera_ids = [c.strip() for c in args.camera_ids.split(",") if c.strip()]
        cameras = tuple(
            CameraStreamConfig(name=f"cam{i}", device_id=int(cid), fps=args.camera_fps)
            for i, cid in enumerate(camera_ids)
        )

        if args.right_leader and args.right_follower:
            config = TeleopSessionConfig.bimanual(
                left_leader_port=args.leader,
                left_follower_port=args.follower,
                right_leader_port=args.right_leader,
                right_follower_port=args.right_follower,
                rate_hz=args.rate_hz,
                output_dir=args.output,
                cameras=cameras,
            )
            session = BimanualSo101Session(config=config)
        else:
            session = BimanualSo101Session.from_single_pair(
                leader_port=args.leader,
                follower_port=args.follower,
                rate_hz=args.rate_hz,
                output_dir=args.output,
                cameras=cameras,
            )

    camera_fps = args.camera_fps
    if args.config and session.config.cameras:
        camera_fps = int(session.config.cameras[0].fps)

    # -------------------------------------------------------------------
    # 2. Start session
    # -------------------------------------------------------------------
    print("Starting teleop session...")
    session.start()

    recording = False

    def maybe_export(result):
        fmt = args.export_format
        if fmt in {"lerobot", "both"} and args.lerobot_repo_id:
            try:
                summary = session.recorder.export_episode_to_lerobot(
                    result,
                    repo_id=args.lerobot_repo_id,
                    root=args.lerobot_root,
                    fps=camera_fps,
                    task=args.lerobot_task or args.label,
                )
                print(f"LeRobot export: repo={summary['repo_id']} frames={summary['frames_added']}")
            except Exception as exc:
                print(f"LeRobot export failed: {exc}")
        if fmt in {"mcap", "both"}:
            try:
                summary = session.recorder.export_episode_to_mcap(
                    result,
                    output_dir=args.mcap_root,
                    include_camera_frames=not args.mcap_no_camera_frames,
                )
                print(f"MCAP export: path={summary['mcap_path']} msgs={summary['control_messages']}")
            except Exception as exc:
                print(f"MCAP export failed: {exc}")

    # -------------------------------------------------------------------
    # 3. Interactive loop
    #
    # Enter = start/stop recording
    # h     = go home
    # q     = quit
    # -------------------------------------------------------------------
    print("Controls: Enter=start/stop recording, h=home, q=quit")
    quit_flag = [False]
    toggle_record = [False]
    go_home = [False]

    def input_thread():
        while not quit_flag[0]:
            try:
                cmd = input()
                if cmd == "q":
                    quit_flag[0] = True
                elif cmd == "h":
                    go_home[0] = True
                else:
                    toggle_record[0] = True
            except EOFError:
                break

    threading.Thread(target=input_thread, daemon=True).start()

    step = 0
    try:
        while not quit_flag[0]:
            if go_home[0]:
                go_home[0] = False
                session.go_home()
                time.sleep(0.5)
                continue

            if toggle_record[0]:
                toggle_record[0] = False
                if recording:
                    result = session.stop_recording()
                    print(f"Saved: {result.episode_id} → {result.manifest_path}")
                    maybe_export(result)
                    recording = False
                else:
                    episode_id = session.start_recording(label=args.label)
                    recording = True
                    print(f"Recording: {episode_id}")

            if step % 25 == 0:
                stats = session.timing_stats()
                positions = session.latest_positions()
                pair = next(iter(positions.keys()), None)
                preview = positions.get(pair, ())[:3] if pair else ()
                tag = "REC" if recording else "   "
                print(
                    f"[{tag}] step={step} pair={pair} pos={preview} "
                    f"p99={stats.p99_jitter_s * 1e3:.3f}ms"
                )

            step += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if recording:
            result = session.stop_recording()
            print(f"Saved: {result.episode_id} → {result.manifest_path}")
            maybe_export(result)
        session.stop()

    stats = session.timing_stats()
    print(f"\nDone — {stats.iterations} iterations, {stats.overruns} overruns, p99={stats.p99_jitter_s * 1e3:.3f}ms")


if __name__ == "__main__":
    main()
