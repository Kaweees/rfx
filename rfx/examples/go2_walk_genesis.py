#!/usr/bin/env python3
"""
Go2 walking demo in Genesis via rfx.

Usage:
    uv run --python 3.13 rfx/examples/go2_walk_genesis.py
    uv run --python 3.13 rfx/examples/go2_walk_genesis.py --vx 0.7 --yaw 0.2
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from rfx.envs import Go2Env
from rfx.sim import SimRobot


def _build_trot_action(
    phase: float,
    vx: float,
    yaw: float,
    stride_scale: float,
    lift_scale: float,
    device: str,
) -> torch.Tensor:
    base = torch.tensor(Go2Env.DEFAULT_STANDING, dtype=torch.float32, device=device)
    low = torch.tensor(Go2Env.JOINT_LIMITS_LOW, dtype=torch.float32, device=device)
    high = torch.tensor(Go2Env.JOINT_LIMITS_HIGH, dtype=torch.float32, device=device)

    # FR, FL, RR, RL. Diagonal legs in phase for trot.
    phase_offsets = [0.0, math.pi, math.pi, 0.0]
    side_sign = [-1.0, 1.0, -1.0, 1.0]  # right=-1, left=+1

    stride_amp = 0.32 * max(0.0, min(1.0, abs(vx))) * stride_scale
    turn_amp = 0.18 * max(-1.0, min(1.0, yaw))
    lift_amp = 0.42 * max(0.0, min(1.0, abs(vx) + 0.15)) * lift_scale

    target = base.clone()
    for leg in range(4):
        i = leg * 3
        phi = phase + phase_offsets[leg]
        s = math.sin(phi)
        c = math.cos(phi)
        lift = max(0.0, s)

        hip = base[i] + stride_amp * s + (turn_amp * side_sign[leg])
        thigh = base[i + 1] + 0.18 * stride_amp * c - lift_amp * lift
        calf = base[i + 2] - 0.30 * stride_amp * c + 1.6 * lift_amp * lift

        target[i] = hip
        target[i + 1] = thigh
        target[i + 2] = calf

    return torch.clamp(target, low, high)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop Go2 walking in Genesis")
    parser.add_argument("--config", default="rfx/configs/go2.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--hz", type=float, default=200.0)
    parser.add_argument("--gait-hz", type=float, default=1.8)
    parser.add_argument("--vx", type=float, default=0.6, help="Normalized forward command [0..1]")
    parser.add_argument("--yaw", type=float, default=0.0, help="Normalized yaw command [-1..1]")
    parser.add_argument("--stride-scale", type=float, default=1.0)
    parser.add_argument("--lift-scale", type=float, default=1.0)
    parser.add_argument("--auto-install", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    robot = SimRobot.from_config(
        cfg_path,
        num_envs=1,
        backend="genesis",
        device=args.device,
        viewer=not args.headless,
        auto_install=args.auto_install,
        dt=1.0 / args.hz,
        substeps=4,
    )

    print(robot)
    print("Starting open-loop trot demo. Ctrl+C to stop.")
    print(f"vx={args.vx:.2f}, yaw={args.yaw:.2f}, gait_hz={args.gait_hz:.2f}, hz={args.hz:.1f}")

    obs = robot.reset()
    dt = 1.0 / args.hz
    phase = 0.0
    phase_step = 2.0 * math.pi * args.gait_hz * dt

    try:
        for step in range(args.steps):
            joints = _build_trot_action(
                phase=phase,
                vx=args.vx,
                yaw=args.yaw,
                stride_scale=args.stride_scale,
                lift_scale=args.lift_scale,
                device=args.device,
            )
            action = torch.zeros(1, robot.max_action_dim, dtype=torch.float32, device=args.device)
            action[0, :12] = joints
            robot.act(action)

            if step % 200 == 0:
                obs = robot.observe()
                print(f"step={step:5d} state_norm={obs['state'].norm(dim=-1).mean().item():.4f}")

            phase += phase_step
            if not args.headless:
                # Keep wall-clock pacing so the viewer remains intuitive.
                time.sleep(dt)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
