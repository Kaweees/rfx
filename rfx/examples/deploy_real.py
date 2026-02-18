#!/usr/bin/env python3
"""
Deploy a trained policy to a real SO-101 arm.

This is the simplest path from a trained checkpoint to hardware. You
provide a robot config, a policy, and call rfx.run(). The Session
handles reset, warmup, rate control, torch.no_grad(), and Ctrl+C.

Key concepts:
    rfx.RealRobot        — real hardware via serial (same API as SimRobot)
    rfx.MockRobot        — drop-in fake for testing without hardware
    rfx.run()            — one-liner: connect policy to robot and go
    rfx.Session          — context-manager variant for more control

Robot configs (rfx/configs/):
    so101.yaml           — SO-101 arm, 6 joints, 50 Hz
    so101_bimanual.yaml  — bimanual SO-101, 12 joints, 50 Hz

Usage:
    # Test without hardware
    uv run rfx/examples/deploy_real.py --mock

    # Real robot
    uv run rfx/examples/deploy_real.py --port /dev/ttyACM0 --checkpoint model.pt

    # Timed run
    uv run rfx/examples/deploy_real.py --mock --duration 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import rfx


# -------------------------------------------------------------------
# 1. Define your policy
#
# A policy is any callable:  Dict[str, Tensor] → Tensor
#
#   - nn.Module with forward(obs_dict) → action
#   - @rfx.policy decorated function
#   - lambda obs: torch.zeros_like(obs["state"])
#
# The observation dict comes from robot.observe(). Its key tensor is
# "state" with shape (num_envs, max_state_dim) — padded to 64 dims
# for multi-embodiment compatibility.
# -------------------------------------------------------------------


class SimpleVLA(nn.Module):
    """Minimal MLP: obs["state"] → action."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.net(obs["state"])


def main():
    parser = argparse.ArgumentParser(description="Deploy policy to real robot")
    parser.add_argument("--config", default="so101.yaml", help="Robot config file in rfx/configs/")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for real robot")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint (.pt)")
    parser.add_argument("--mock", action="store_true", help="Use MockRobot (no hardware)")
    parser.add_argument("--rate", type=float, default=50, help="Control loop Hz")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run (default: infinite)")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 2. Create a robot
    #
    # MockRobot runs a simple spring-damper physics model in pure
    # PyTorch — same observe()/act()/reset() interface as RealRobot.
    # -------------------------------------------------------------------
    config_path = Path(__file__).parent.parent / "configs" / args.config

    if args.mock:
        robot = rfx.MockRobot(state_dim=12, action_dim=6)
    else:
        robot = rfx.RealRobot.from_config(config_path, port=args.port)

    print(f"Robot: {robot}")

    # -------------------------------------------------------------------
    # 3. Load policy
    # -------------------------------------------------------------------
    policy = SimpleVLA(robot.max_state_dim, robot.max_action_dim)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(state["model_state_dict"])
        print(f"Loaded: {args.checkpoint}")

    policy.eval()

    # -------------------------------------------------------------------
    # 4. Run
    #
    # rfx.run() handles: reset → warmup → observe → no_grad → policy →
    # act → sleep, in a tight loop at the target rate. Returns stats.
    # -------------------------------------------------------------------
    label = f"{args.rate} Hz" + (f" for {args.duration}s" if args.duration else ", Ctrl+C to stop")
    print(f"Running at {label}...")

    stats = rfx.run(robot, policy, rate_hz=args.rate, duration=args.duration)

    print(f"\nDone — {stats.iterations} steps, {stats.overruns} overruns")
    print(f"  avg period:  {stats.avg_period_s * 1000:.2f} ms  (target: {stats.target_period_s * 1000:.2f} ms)")
    print(f"  jitter p50:  {stats.p50_jitter_s * 1000:.2f} ms")
    print(f"  jitter p95:  {stats.p95_jitter_s * 1000:.2f} ms")


if __name__ == "__main__":
    main()
