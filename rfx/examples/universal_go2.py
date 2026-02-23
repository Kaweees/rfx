#!/usr/bin/env python3
"""
Canonical Go2 simulation control example for the lean SDK.

Use this file to validate one control workflow across simulation
backends (`mock`, `genesis`, `mjx`) without mixed wrappers.

Usage:
    uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
    uv run --python 3.13 rfx/examples/universal_go2.py --backend mock
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rfx
from rfx.config import GO2_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Go2 runner (simulation backends)")
    parser.add_argument("--backend", choices=["mock", "genesis", "mjx"], default="genesis")
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

    robot = rfx.sim.SimRobot.from_config(
        backend=args.backend,
        config=config,
        num_envs=args.num_envs,
        device=args.device,
        **kwargs,
    )
    print(f"Connected sim robot: backend={args.backend} num_envs={args.num_envs}")
    obs = robot.reset()
    print(f"initial_state_norm={obs['state'].norm(dim=-1).mean().item():.4f}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for this simulation example") from exc

    try:
        for step in range(args.steps):
            action = torch.zeros((robot.num_envs, robot.max_action_dim), device=robot.device)
            obs = robot.observe()
            robot.act(action)
            if step % 200 == 0:
                print(f"step={step:5d} state_norm={obs['state'].norm(dim=-1).mean().item():.4f}")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
