#!/usr/bin/env python3
"""
Canonical Genesis visualization example for rfx.

Use this script to verify simulator setup, URDF loading, and control
signal flow before moving to training or deployment. It intentionally
uses a simple periodic action so visual behavior is easy to inspect.

Key concepts:
    rfx.SimRobot         — unified sim interface for all backends
    backend="genesis"    — selects the Genesis physics engine
    viewer=True          — opens a live 3D window
    robot.act(action)    — steps physics with joint position commands

Config defaults:
    Uses built-in SO-101 config from `rfx-sdk` unless --config is provided.

Usage:
    uv run rfx/examples/genesis_viewer.py
    uv run rfx/examples/genesis_viewer.py --steps 5000
    uv run rfx/examples/genesis_viewer.py --auto-install  # pip install genesis-world
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import rfx
from rfx.config import SO101_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Genesis simulation viewer")
    parser.add_argument("--config", default=None, help="Optional path to a robot YAML config")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per step")
    parser.add_argument("--dt", type=float, default=None, help="Override timestep (seconds)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sleep", type=float, default=0.0, help="Wall-clock delay per step (seconds)")
    parser.add_argument("--auto-install", action="store_true", help="Auto-install genesis-world if missing")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 1. Create a SimRobot with the Genesis backend
    #
    # SimRobot.from_config() reads the YAML and creates the specified
    # backend. Pass viewer=True to open a live 3D window.
    # -------------------------------------------------------------------
    kwargs = {"substeps": args.substeps, "viewer": True, "auto_install": args.auto_install}
    if args.dt is not None:
        kwargs["dt"] = args.dt

    config = SO101_CONFIG.to_dict()
    if args.config:
        path = Path(args.config).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        config = path

    robot = rfx.SimRobot.from_config(
        config,
        num_envs=args.num_envs,
        backend="genesis",
        device=args.device,
        **kwargs,
    )
    print(f"Robot: {robot}")

    # -------------------------------------------------------------------
    # 2. Reset and run
    #
    # We send a simple sinusoidal command to the first two joints so
    # you can see the robot move in the viewer.
    # -------------------------------------------------------------------
    obs = robot.reset()
    action_dim = robot.max_action_dim
    control_dt = args.dt if args.dt is not None else (1.0 / robot.config.control_freq_hz)

    try:
        for step in range(args.steps):
            t = step * control_dt
            action = torch.zeros(args.num_envs, action_dim, device=args.device)
            action[:, 0] = 0.25 * torch.sin(torch.tensor(t, device=args.device))
            action[:, 1] = 0.25 * torch.cos(torch.tensor(t, device=args.device))

            robot.act(action)
            obs = robot.observe()

            if step % 100 == 0:
                norm = obs["state"].norm(dim=-1).mean().item()
                print(f"step={step:5d}  state_norm={norm:.4f}")

            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        robot.close()

    print("Done!")


if __name__ == "__main__":
    main()
