#!/usr/bin/env python3
"""
Teleoperation recording using the new rfx.teleop session API.

Usage:
    python teleop_record.py --leader /dev/ttyACM0 --follower /dev/ttyACM1 --output demos/
"""

import argparse
import threading
import time

from rfx.teleop import BimanualSo101Session, CameraStreamConfig, TeleopSessionConfig


def run_teleop(args):
    camera_ids = [c.strip() for c in args.camera_ids.split(",") if c.strip()]
    cameras = tuple(
        CameraStreamConfig(name=f"cam{i}", device_id=int(cam_id), fps=args.camera_fps)
        for i, cam_id in enumerate(camera_ids)
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

    print("Starting teleoperation session...")
    session.start()

    recording = False
    current_episode_id = None

    def maybe_export_lerobot(result):
        if not args.lerobot_repo_id:
            return
        try:
            export_summary = session.recorder.export_episode_to_lerobot(
                result,
                repo_id=args.lerobot_repo_id,
                root=args.lerobot_root,
                fps=args.camera_fps,
                task=args.lerobot_task or args.label,
            )
            print(
                f"LeRobot export complete: repo={export_summary['repo_id']} "
                f"frames={export_summary['frames_added']}"
            )
        except Exception as exc:
            print(f"LeRobot export failed: {exc}")

    print("Controls: Enter=start/stop recording, q=quit, h=home")
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
                    print(f"Saved episode {result.episode_id} -> {result.manifest_path}")
                    maybe_export_lerobot(result)
                    recording = False
                    current_episode_id = None
                else:
                    current_episode_id = session.start_recording(label=args.label)
                    recording = True
                    print(f"Recording: {current_episode_id}")

            if step % 25 == 0:
                stats = session.timing_stats()
                positions = session.latest_positions()
                main_pair = next(iter(positions.keys()), None)
                pos_preview = positions.get(main_pair, ())[:3] if main_pair else ()
                status = "REC" if recording else "   "
                print(
                    f"[{status}] Step {step} | pair={main_pair} pos={pos_preview} "
                    f"| p99_jitter={stats.p99_jitter_s * 1e3:.3f} ms"
                )

            step += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if recording:
            result = session.stop_recording()
            print(f"Saved episode {result.episode_id} -> {result.manifest_path}")
            maybe_export_lerobot(result)
        session.stop()

    stats = session.timing_stats()
    print(
        f"Stopped. Iterations={stats.iterations}, overruns={stats.overruns}, "
        f"p99_jitter={stats.p99_jitter_s * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="/dev/ttyACM0")
    parser.add_argument("--follower", default="/dev/ttyACM1")
    parser.add_argument("--right-leader", default=None)
    parser.add_argument("--right-follower", default=None)
    parser.add_argument("--rate-hz", type=float, default=350.0)
    parser.add_argument("--camera-ids", default="0,1,2")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--label", default="teleop")
    parser.add_argument("--lerobot-repo-id", default=None)
    parser.add_argument("--lerobot-root", default="lerobot_datasets")
    parser.add_argument("--lerobot-task", default=None)
    parser.add_argument("--output", default="demos")
    run_teleop(parser.parse_args())
