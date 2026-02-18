#!/usr/bin/env python3
"""
One-import SDK workflow.

Usage:
    uv run --python 3.13 rfx/examples/python_sdk_one_import.py
"""

from __future__ import annotations

import rfx


def main() -> None:
    print("Available providers:", rfx.available_providers())
    print("Enable providers:", rfx.use("sim", "go2"))

    bot = rfx.connect_robot(
        "go2",
        backend="mock",
        config="rfx/configs/go2.yaml",
        num_envs=1,
        device="cpu",
    )
    bot.reset()
    bot.command(vx=0.5, vy=0.0, yaw=0.1)
    obs = bot.step()
    print("obs shape:", tuple(obs["state"].shape))
    bot.close()


if __name__ == "__main__":
    main()
