#!/usr/bin/env python3
"""
Train a Go2 walking policy with PPO (tinygrad).

Uses the rfx v1 RL stack — tinygrad-based networks, PPO trainer, and
the Go2Env gym-style environment.

Key concepts:
    rfx.nn.go2_actor_critic — creates an ActorCritic MLP for Go2
    rfx.rl.PPOTrainer       — standard PPO with GAE
    rfx.rl.collect_rollout  — fills a buffer with env transitions
    rfx.envs.Go2Env         — Go2 with gym reset()/step() interface

Usage:
    uv run rfx/examples/train_walking.py

Requirements:
    pip install tinygrad
"""

import rfx
from rfx.nn import go2_actor_critic
from rfx.rl import PPOTrainer, collect_rollout
from rfx.envs import Go2Env


def main():
    # -------------------------------------------------------------------
    # 1. Create environment
    # -------------------------------------------------------------------
    env = Go2Env(sim=True)
    print(f"Env:  obs={env.observation_space.shape}, act={env.action_space.shape}")

    # -------------------------------------------------------------------
    # 2. Create policy
    #
    # go2_actor_critic() returns an ActorCritic module with separate
    # actor (policy) and critic (value) heads.
    # -------------------------------------------------------------------
    policy = go2_actor_critic(hidden=[256, 256])
    print(f"Policy: {policy}")

    # -------------------------------------------------------------------
    # 3. Create PPO trainer
    # -------------------------------------------------------------------
    trainer = PPOTrainer(
        policy,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        update_epochs=10,
        minibatch_size=64,
    )

    # -------------------------------------------------------------------
    # 4. Training loop
    #
    # Each epoch: collect a rollout of 2048 steps, then run PPO updates.
    # -------------------------------------------------------------------
    num_epochs = 100
    steps_per_epoch = 2048
    best_reward = float("-inf")

    print(f"\nTraining: {num_epochs} epochs x {steps_per_epoch} steps")
    print("-" * 60)

    for epoch in range(num_epochs):
        rollout = collect_rollout(env, policy, steps=steps_per_epoch)
        metrics = trainer.update(rollout)

        mean_reward = metrics["mean_reward"]
        total_reward = metrics["total_reward"]

        print(
            f"Epoch {epoch:3d} | "
            f"reward={mean_reward:7.3f} (total={total_reward:8.1f}) | "
            f"pi_loss={metrics['policy_loss']:7.4f} | "
            f"v_loss={metrics['value_loss']:7.4f}"
        )

        if total_reward > best_reward:
            best_reward = total_reward
            policy.save("walking_policy_best.safetensors")

    # -------------------------------------------------------------------
    # 5. Save
    # -------------------------------------------------------------------
    policy.save("walking_policy.safetensors")
    print("-" * 60)
    print(f"Done — best reward: {best_reward:.2f}")
    print(f"  walking_policy.safetensors       (final)")
    print(f"  walking_policy_best.safetensors   (best)")


if __name__ == "__main__":
    main()
