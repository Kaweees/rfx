#!/usr/bin/env python3
"""
Canonical Go2 control example for rfx.

Use this file to validate one command-level workflow across multiple
backends (`mock`, `genesis`, `mjx`, `real`) without changing your
control code. This is the reference script for the "one API, many
backends" story.

Usage:
    uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
    uv run --python 3.13 rfx/examples/universal_go2.py --backend mock
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import rfx
from rfx.config import GO2_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal Go2 runner (sim/real)")
    parser.add_argument("--backend", choices=["mock", "genesis", "mjx", "real"], default="genesis")
    parser.add_argument("--config", default=None, help="Optional path to a Go2 YAML config")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--vx", type=float, default=0.6)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--auto-install", action="store_true")
    args = parser.parse_args()

    config = GO2_CONFIG.to_dict()
    if args.config:
        path = Path(args.config).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        config = path

    kwargs = {}
    if args.backend == "genesis":
        kwargs["viewer"] = not args.headless
        kwargs["auto_install"] = args.auto_install

    bot = rfx.connect_robot(
        "go2",
        backend=args.backend,
        config=config,
        num_envs=args.num_envs,
        device=args.device,
        **kwargs,
    )
    print(f"Connected universal bot: robot={bot.robot_name} backend={bot.backend}")
    bot.reset()
    bot.command(vx=args.vx, vy=args.vy, yaw=args.yaw)

    try:
        for step in range(args.steps):
            obs = bot.step()
            if step % 200 == 0:
                print(f"step={step:5d} state_norm={obs['state'].norm(dim=-1).mean().item():.4f}")
            if args.backend != "real" and not args.headless:
                time.sleep(0.01)
    finally:
        bot.close()


if __name__ == "__main__":
    main()
