#!/usr/bin/env python3
"""
Visualise a robot in the Genesis physics engine.

Opens a live 3D viewer and steps the simulation with a simple
sinusoidal joint command, so you can see the robot move.

Key concepts:
    rfx.SimRobot         — unified sim interface for all backends
    backend="genesis"    — selects the Genesis physics engine
    viewer=True          — opens a live 3D window
    robot.act(action)    — steps physics with joint position commands

Robot configs (rfx/configs/):
    so101.yaml           — SO-101 arm, 6 joints
    go2.yaml             — Unitree Go2, 12 joints

Usage:
    uv run rfx/examples/genesis_viewer.py
    uv run rfx/examples/genesis_viewer.py --config rfx/configs/go2.yaml --steps 5000
    uv run rfx/examples/genesis_viewer.py --auto-install  # pip install genesis-world
"""

from __future__ import annotations

import argparse
import time

import torch

import rfx


def main():
    parser = argparse.ArgumentParser(description="Genesis simulation viewer")
    parser.add_argument("--config", default="rfx/configs/so101.yaml", help="Robot config YAML")
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

    robot = rfx.SimRobot.from_config(
        args.config,
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
