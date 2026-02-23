"""rfx.collection._cli — CLI integration for rfx collect."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    """Add args to the collect subcommand."""
    parser.add_argument("--robot", required=True, help="robot type (e.g. so101)")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--output", "-o", default="datasets", help="output root directory")
    parser.add_argument(
        "--episodes", "-n", type=int, default=1, help="number of episodes to collect"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=None, help="duration per episode in seconds"
    )
    parser.add_argument("--task", default="default", help="task label for episodes")
    parser.add_argument("--fps", type=int, default=30, help="recording frame rate")
    parser.add_argument("--push", action="store_true", help="push to Hub after collection")
    parser.add_argument("--mcap", action="store_true", help="also log MCAP sidecar")
    parser.add_argument("--state-dim", type=int, default=6, help="state dimension")


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    """Execute collection. Called by workflow/stages.py or CLI."""
    from ._recorder import Recorder

    recorder = Recorder.create(
        args.repo_id,
        root=args.output,
        fps=args.fps,
        robot_type=args.robot,
        state_dim=args.state_dim,
        mcap=args.mcap,
    )

    total_frames = 0
    for ep in range(args.episodes):
        recorder.start_episode(task=args.task)
        print(f"[rfx] Recording episode {ep + 1}/{args.episodes}...")

        if args.duration is not None:
            deadline = time.perf_counter() + args.duration
            while time.perf_counter() < deadline:
                time.sleep(0.01)
        else:
            print("[rfx] Press Ctrl+C to finish episode.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

        count = recorder.save_episode()
        total_frames += count
        print(f"[rfx] Episode {ep + 1} saved: {count} frames")

    if args.push:
        print("[rfx] Pushing to Hub...")
        recorder.push()
        print(f"[rfx] Pushed to https://huggingface.co/datasets/{args.repo_id}")

    return {
        "repo_id": args.repo_id,
        "episodes": args.episodes,
        "total_frames": total_frames,
        "root": str(Path(args.output).resolve()),
    }
