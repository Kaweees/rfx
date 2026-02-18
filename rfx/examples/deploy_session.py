#!/usr/bin/env python3
"""
Run a policy on a robot with rfx.Session.

rfx.Session handles the control loop, rate control, error handling, and
clean shutdown so you don't have to. All you provide is a robot and a policy.

    robot  — anything with .observe(), .act(), .reset()  (rfx.Robot protocol)
    policy — any callable:  obs dict → action tensor

Available robot configs (rfx/configs/):
    so101.yaml          SO-101 arm    — 6 joints,  50 Hz
    so101_bimanual.yaml SO-101 x2     — 12 joints, 50 Hz
    go2.yaml            Unitree Go2   — 12 joints, 200 Hz

Usage:
    # MockRobot (no hardware needed)
    uv run rfx/examples/deploy_session.py

    # Real SO-101 arm
    uv run rfx/examples/deploy_session.py --port /dev/ttyACM0

    # With a trained checkpoint, 10-second run
    uv run rfx/examples/deploy_session.py --port /dev/ttyACM0 --checkpoint model.pt --duration 10
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import rfx

# ---------------------------------------------------------------------------
# 1. Define your policy
#
# A policy is any callable:  Dict[str, Tensor] → Tensor
# It receives the observation dict from robot.observe() and returns an
# action tensor of shape (num_envs, max_action_dim).
#
# rfx.Session automatically wraps every call in torch.no_grad().
# ---------------------------------------------------------------------------


class SimplePolicy(nn.Module):
    """Minimal MLP policy: obs["state"] → action."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.net(obs["state"])


def main():
    parser = argparse.ArgumentParser(description="Run a policy on a robot with rfx.Session")
    parser.add_argument("--config", default="so101.yaml", help="Robot config (so101.yaml, go2.yaml, ...)")
    parser.add_argument("--port", default=None, help="Serial port for real robot (e.g. /dev/ttyACM0)")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    parser.add_argument("--rate", type=float, default=50, help="Control loop rate in Hz")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to run (omit for infinite)")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 2. Create a robot
    #
    # rfx.MockRobot  — no hardware, pure PyTorch physics (great for testing)
    # rfx.RealRobot  — real hardware via serial port
    # rfx.SimRobot   — GPU-accelerated simulation (Genesis, MJX, ...)
    # -------------------------------------------------------------------
    config_path = Path(__file__).parent.parent / "configs" / args.config

    if args.port:
        robot = rfx.RealRobot.from_config(config_path, port=args.port)
    else:
        robot = rfx.MockRobot(state_dim=12, action_dim=6)

    print(f"Robot: {robot}")

    # -------------------------------------------------------------------
    # 3. Load your policy
    # -------------------------------------------------------------------
    policy = SimplePolicy(robot.max_state_dim, robot.max_action_dim)

    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location="cpu")
        policy.load_state_dict(state["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    policy.eval()

    # -------------------------------------------------------------------
    # 4. Run — one line
    #
    # rfx.run() resets the robot, warms up, then runs the control loop at
    # the target rate. It handles Ctrl+C, errors, and cleanup. Returns
    # timing stats when done.
    #
    # For more control, use rfx.Session directly:
    #
    #   with rfx.Session(robot, policy, rate_hz=50) as session:
    #       session.run(duration=10.0)
    #       print(session.step_count)
    #       print(session.stats)
    #
    # -------------------------------------------------------------------
    print(f"Running at {args.rate} Hz for {args.duration}s...")
    stats = rfx.run(robot, policy, rate_hz=args.rate, duration=args.duration)

    # -------------------------------------------------------------------
    # 5. Inspect results
    #
    # SessionStats tracks iterations, overruns, and jitter percentiles
    # so you can verify your loop is hitting its target rate.
    # -------------------------------------------------------------------
    print(f"\nDone — {stats.iterations} steps, {stats.overruns} overruns")
    print(f"  avg period:  {stats.avg_period_s * 1000:.2f} ms  (target: {stats.target_period_s * 1000:.2f} ms)")
    print(f"  jitter p50:  {stats.p50_jitter_s * 1000:.2f} ms")
    print(f"  jitter p95:  {stats.p95_jitter_s * 1000:.2f} ms")
    print(f"  jitter p99:  {stats.p99_jitter_s * 1000:.2f} ms")


if __name__ == "__main__":
    main()
