#!/usr/bin/env python3
"""
Run a simple proportional controller on the Rust-backed mock sim.

This example uses the rfx v1 Rust simulation API directly — SimConfig,
MockSimBackend, PhysicsConfig — to step a physics sim and apply a
proportional controller to drive joints to a target position.

Key concepts:
    rfx.SimConfig        — simulation backend selector (mock, isaac, genesis, mujoco)
    rfx.MockSimBackend   — Rust-backed mock physics (12 joints)
    rfx.PhysicsConfig    — timestep, substeps, solver settings

Usage:
    uv run rfx/examples/neural_policy.py
"""

import rfx


def main():
    # -------------------------------------------------------------------
    # 1. Create a Rust-backed mock simulation
    #
    # SimConfig.mock() selects the mock backend. Pass it to
    # MockSimBackend to get a lightweight 12-joint sim with no external
    # physics engine required.
    # -------------------------------------------------------------------
    config = rfx.SimConfig.mock()
    sim = rfx.MockSimBackend(config)

    print(f"Backend: {sim.name()}")
    print(f"Initial time: {sim.sim_time():.3f}s")

    state = sim.reset()

    # -------------------------------------------------------------------
    # 2. Proportional controller
    #
    # Read joint positions from state, compute error to target, apply
    # a simple P-gain. This is the simplest possible "policy".
    # -------------------------------------------------------------------
    target_positions = [0.5] * 12
    kp = 0.5

    print(f"\nDriving all joints to 0.5 rad with kp={kp}...")

    for step in range(500):
        current = state.joint_positions()
        actions = [kp * (t - c) for t, c in zip(target_positions, current)]

        state, done = sim.step(actions)

        if step % 100 == 0:
            print(f"  step={step:4d}  t={state.sim_time():.3f}s  joint[0]={current[0]:.4f} rad")

        if done:
            print("Episode terminated!")
            break

    # -------------------------------------------------------------------
    # 3. Results
    # -------------------------------------------------------------------
    final = state.joint_positions()
    print(f"\nFinal positions (target=0.5):")
    for i, pos in enumerate(final):
        print(f"  joint {i:2d}: {pos:.4f} rad")


def show_configs():
    """Show available physics and sim backend configurations."""
    print("Physics configs:")
    for name, factory in [("default", rfx.PhysicsConfig), ("fast", rfx.PhysicsConfig.fast), ("accurate", rfx.PhysicsConfig.accurate)]:
        cfg = factory() if callable(factory) else factory
        if hasattr(cfg, 'dt'):
            print(f"  {name:10s}  dt={cfg.dt}  substeps={cfg.substeps}")

    print("\nSim backends:")
    for name, factory in [("mock", rfx.SimConfig.mock), ("genesis", rfx.SimConfig.genesis), ("mujoco", rfx.SimConfig.mujoco)]:
        cfg = factory()
        print(f"  {name:10s}  backend='{cfg.backend}'")

    parallel = rfx.SimConfig.mock().with_num_envs(4096)
    print(f"\n  Parallel envs: num_envs={parallel.num_envs}")


if __name__ == "__main__":
    show_configs()
    print()
    main()
