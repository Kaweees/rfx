"""
Load any robot from URDF and simulate it.

    python examples/urdf_sim.py --urdf go2.urdf
    python examples/urdf_sim.py --urdf my_arm.urdf --freq 50
"""

import argparse
import rfx


def main():
    parser = argparse.ArgumentParser(description="Load a URDF and simulate")
    parser.add_argument("--urdf", required=True, help="Path to URDF file")
    parser.add_argument("--freq", type=int, default=200, help="Control frequency (Hz)")
    parser.add_argument("--steps", type=int, default=100, help="Sim steps to run")
    parser.add_argument("--envs", type=int, default=1, help="Parallel environments")
    args = parser.parse_args()

    # Parse URDF
    model = rfx.URDF.load(args.urdf)
    print(model)
    model.print_tree()

    # Generate config from URDF and launch sim
    config = model.to_robot_config(control_freq_hz=args.freq)
    robot = rfx.SimRobot(config, num_envs=args.envs, backend="genesis", device="cpu")

    obs = robot.reset()
    print(f"\nRunning {args.steps} steps...")

    for step in range(args.steps):
        robot.act(obs["state"])  # feed state back as action (hold position)
        obs = robot.observe()

        if step % 25 == 0:
            q = obs["state"][0, : model.num_actuated].tolist()
            fk = model.forward_kinematics(q)

            # Print end-effector / foot positions
            leaf_links = [
                name
                for name in model.link_names
                if name not in model._children  # links with no children
            ]
            positions = {
                name: model.link_position(name, q) for name in leaf_links[:4]
            }
            print(f"  step {step}: {positions}")

    robot.close()
    print("Done.")


if __name__ == "__main__":
    main()
