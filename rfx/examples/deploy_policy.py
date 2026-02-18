#!/usr/bin/env python3
"""
Deploy a trained Go2 locomotion policy (tinygrad).

This example uses the rfx v1 RL stack (tinygrad-based) to load a
trained walking policy and run it on the Go2 — in simulation or on
real hardware.

Key concepts:
    rfx.nn.MLP           — tinygrad MLP with .load()/.save()
    rfx.nn.JitPolicy     — TinyJit wrapper for fast inference
    rfx.envs.Go2Env      — Go2 environment (sim or real)

Usage:
    # Simulation (default)
    uv run rfx/examples/deploy_policy.py

    # Real robot
    uv run rfx/examples/deploy_policy.py --real --ip 192.168.123.161

Requirements:
    pip install tinygrad
"""

import argparse
import time

import numpy as np

import rfx
from rfx.nn import MLP, JitPolicy
from rfx.envs import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Deploy Go2 locomotion policy")
    parser.add_argument("--model", default="walking_policy.safetensors", help="Trained policy path")
    parser.add_argument("--real", action="store_true", help="Use real robot instead of sim")
    parser.add_argument("--ip", default="192.168.123.161", help="Robot IP (real mode)")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to run")
    parser.add_argument("--rate", type=float, default=50.0, help="Control loop Hz")
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # 1. Load policy
    #
    # rfx.nn.MLP is a tinygrad model. .load() reads safetensors format.
    # JitPolicy wraps it with @TinyJit for compiled inference.
    # -------------------------------------------------------------------
    print(f"Loading policy: {args.model}")
    try:
        policy = MLP.load(args.model)
    except FileNotFoundError:
        print("  Model not found — using random policy for demo")
        policy = MLP(obs_dim=48, act_dim=12, hidden=[256, 256])

    policy = JitPolicy(policy)
    print(f"  Policy: {policy}")

    # -------------------------------------------------------------------
    # 2. Create environment
    #
    # Go2Env wraps the Go2 robot with a standard gym-style interface:
    # reset() → obs, step(action) → obs, reward, done, info
    # -------------------------------------------------------------------
    if args.real:
        print(f"\nConnecting to real robot at {args.ip}...")
        env = Go2Env(sim=False, robot_ip=args.ip)
    else:
        print("\nUsing simulation...")
        env = Go2Env(sim=True)

    obs = env.reset()
    print(f"Observation shape: {obs.shape}")

    env.set_commands(vx=0.5, vy=0.0, yaw_rate=0.0)

    # -------------------------------------------------------------------
    # 3. Control loop
    #
    # For Go2Env (gym-style), we use a manual loop. For the v2 Robot
    # protocol, you'd use rfx.run() instead — see deploy_real.py.
    # -------------------------------------------------------------------
    dt = 1.0 / args.rate
    num_steps = int(args.duration * args.rate)
    total_reward = 0.0
    start_time = time.time()

    print(f"\nRunning at {args.rate} Hz for {args.duration}s...")
    print("-" * 50)

    try:
        from tinygrad import Tensor

        for step in range(num_steps):
            loop_start = time.time()

            obs_tensor = Tensor(obs.reshape(1, -1).astype(np.float32))
            action = policy(obs_tensor).numpy().flatten()

            obs, reward, done, info = env.step(action)
            total_reward += reward

            if step % int(args.rate) == 0:
                elapsed = time.time() - start_time
                print(
                    f"t={elapsed:5.1f}s | step={step:5d} | "
                    f"reward={reward:6.3f} | total={total_reward:8.2f}"
                )

            if done:
                print("Episode terminated, resetting...")
                obs = env.reset()

            sleep_s = dt - (time.time() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        env.close()

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Done — {elapsed:.1f}s, total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
