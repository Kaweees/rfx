#!/usr/bin/env python3
"""
Canonical training example for rfx.

Use this file as the baseline pattern for simulation-first policy
training: create a parallel robot, roll out actions, compute loss, and
handle selective resets. It is intentionally minimal so teams can swap
in their own reward, model, and optimizer logic quickly.

Key concepts:
    rfx.SimRobot         — GPU-accelerated parallel simulation
    robot.observe()      — returns Dict[str, Tensor] with "state" key
    robot.act(action)    — steps all envs in parallel
    robot.reset(env_ids) — selective reset of terminated envs

Simulation backends (pass --backend):
    mock     — pure PyTorch, no deps (default)
    genesis  — Genesis physics engine
    mjx      — MuJoCo XLA

Config defaults:
    Uses built-in SO-101 config from `rfx-sdk` unless --config is provided.

Usage:
    uv run rfx/examples/train_vla.py --num_envs 16 --steps 10000 --backend mock
    uv run rfx/examples/train_vla.py --num_envs 4096 --steps 1000000 --backend genesis
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import rfx
from rfx.config import SO101_CONFIG


# -------------------------------------------------------------------
# 1. Define your policy
#
# Any nn.Module that takes an obs dict and returns an action tensor.
# The obs dict has "state" of shape (num_envs, max_state_dim).
# The action tensor should be (num_envs, max_action_dim).
# -------------------------------------------------------------------


class SimpleVLA(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.encoder(obs["state"]))


def main():
    parser = argparse.ArgumentParser(description="Train a VLA in simulation")
    parser.add_argument("--config", default=None, help="Optional path to a robot YAML config")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--backend", default="mock", help="Sim backend: mock, genesis, mjx")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=100, help="Print every N steps")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 2. Create a parallel simulation
    #
    # SimRobot.from_config() reads a YAML config and creates num_envs
    # parallel physics instances. All tensors are on the given device.
    # -------------------------------------------------------------------
    config = SO101_CONFIG.to_dict()
    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = config_path

    robot = rfx.SimRobot.from_config(
        config,
        num_envs=args.num_envs,
        backend=args.backend,
        device=args.device,
    )
    print(f"Robot: {robot}")

    # -------------------------------------------------------------------
    # 3. Create policy and optimizer
    # -------------------------------------------------------------------
    policy = SimpleVLA(robot.max_state_dim, robot.max_action_dim).to(args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    # -------------------------------------------------------------------
    # 4. Training loop
    #
    # observe() → policy() → act() → observe() → compute reward → backward
    # The robot handles episode termination: check get_done() and reset
    # only the terminated envs.
    # -------------------------------------------------------------------
    target = torch.randn(args.num_envs, 6, device=args.device) * 0.5
    obs = robot.reset()

    print(f"\nTraining for {args.steps} steps on {args.device}...")

    for step in range(args.steps):
        action = policy(obs)
        robot.act(action.detach())
        new_obs = robot.observe()

        positions = new_obs["state"][:, :6]
        reward = -torch.norm(positions - target, dim=-1)
        loss = -(reward.unsqueeze(-1) * action).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Reset terminated envs
        done = robot.get_done()
        if done.any():
            robot.reset(done.nonzero().squeeze(-1))
            target[done] = torch.randn(done.sum(), 6, device=args.device) * 0.5

        obs = new_obs

        if step % args.log_interval == 0:
            print(f"Step {step:6d} | loss={loss.item():8.4f} | reward={reward.mean().item():8.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
