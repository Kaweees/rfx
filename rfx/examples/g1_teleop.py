#!/usr/bin/env python3
"""
G1 humanoid VR teleoperation example.

End-to-end pipeline: SteamVR -> Cartesian retargeting -> policy -> G1 robot.

Usage:
    # Test without hardware (mock robot + dummy policy)
    uv run rfx/examples/g1_teleop.py --mock

    # Real robot + VR with trained ExtremControl policy
    uv run rfx/examples/g1_teleop.py --policy runs/g1-extremcontrol

    # Real robot, simple run (policy only, no VR retargeting)
    uv run rfx/examples/g1_teleop.py --policy runs/g1-extremcontrol --no-vr
"""

import argparse

import torch

import rfx
from rfx.config import G1_CONFIG
from rfx.teleop.g1_obs import BASE_OBS_DIM


class DummyG1Policy(torch.nn.Module):
    """Minimal MLP policy for testing without a trained checkpoint.

    Input: ExtremControl observation (151 dims by default).
    Output: 29-DOF action (pre-scale, pre-offset).
    """

    def __init__(self, obs_dim: int = BASE_OBS_DIM, act_dim: int = 29):
        super().__init__()
        self._is_torch_native = True
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim),
            torch.nn.Tanh(),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.net(obs["state"])


def main():
    parser = argparse.ArgumentParser(description="G1 humanoid VR teleoperation")
    parser.add_argument("--policy", default=None, help="Path to trained policy checkpoint")
    parser.add_argument("--mock", action="store_true", help="Use MockRobot (no hardware)")
    parser.add_argument("--no-vr", action="store_true", help="Run policy without VR")
    parser.add_argument("--rate", type=float, default=50, help="Control loop Hz")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run")
    parser.add_argument("--ip", default="192.168.123.161", help="G1 robot IP address")
    parser.add_argument(
        "--calibration-time", type=float, default=3.0, help="T-pose calibration seconds"
    )
    args = parser.parse_args()

    # --- Robot ---
    if args.mock:
        robot = rfx.MockRobot(
            state_dim=G1_CONFIG.state_dim,
            action_dim=G1_CONFIG.action_dim,
            max_state_dim=G1_CONFIG.max_state_dim,
            max_action_dim=G1_CONFIG.max_action_dim,
        )
    else:
        from rfx.real import G1Robot

        robot = G1Robot(ip_address=args.ip)

    print(f"Robot: {robot}")

    # --- Policy ---
    if args.policy:
        loaded = rfx.load_policy(args.policy)
        policy = loaded
        print(f"Loaded policy: {loaded.policy_type}")
    else:
        policy = DummyG1Policy()
        policy.eval()
        print(f"Using dummy policy (obs_dim={BASE_OBS_DIM}, act_dim=29)")

    # --- Run ---
    if args.no_vr:
        print(f"Running at {args.rate} Hz (no VR)...")
        stats = rfx.run(robot, policy, rate_hz=args.rate, duration=args.duration)
        print(f"\nDone - {stats.iterations} steps, {stats.overruns} overruns")
    else:
        from rfx.teleop.g1 import G1TeleopConfig, G1TeleopSession

        config = G1TeleopConfig(
            rate_hz=args.rate,
            calibration_s=args.calibration_time,
        )
        session = G1TeleopSession(robot, policy, config=config)
        print(f"Starting VR teleop at {args.rate} Hz...")
        print("Press Ctrl+C to stop")

        session.run(duration=args.duration)

        stats = session.stats
        print(f"\nDone - {stats['iterations']} steps, {stats['overruns']} overruns")
        print(f"  avg period:  {stats['avg_period_s'] * 1000:.2f} ms")
        print(f"  jitter p50:  {stats['p50_jitter_s'] * 1000:.2f} ms")
        print(f"  jitter p95:  {stats['p95_jitter_s'] * 1000:.2f} ms")


if __name__ == "__main__":
    main()
