#!/usr/bin/env python3
"""
SO101 quickstart (real hardware).

Usage:
    uv run --python 3.13 rfx/examples/so101_quickstart.py --port auto
    uv run --python 3.13 rfx/examples/so101_quickstart.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import glob
import math
import time

import rfx


def _auto_port() -> str:
    candidates = []
    patterns = [
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
        "/dev/tty.usbmodem*",
        "/dev/cu.usbmodem*",
    ]
    for pat in patterns:
        candidates.extend(sorted(glob.glob(pat)))
    if not candidates:
        raise RuntimeError("No SO101 serial port detected. Pass --port explicitly.")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 quickstart")
    parser.add_argument("--port", default="auto")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--amplitude", type=float, default=0.25)
    args = parser.parse_args()

    port = _auto_port() if args.port == "auto" else args.port
    print(f"Using port: {port}")

    bot = rfx.connect_robot(
        "so101",
        backend="real",
        config="rfx/configs/so101.yaml",
        port=port,
    )

    print("Connected. Resetting to home...")
    bot.reset()
    dt = 1.0 / max(args.hz, 1.0)
    t0 = time.time()

    try:
        while (time.time() - t0) < args.seconds:
            t = time.time() - t0
            # Gentle motion on shoulder_pan and wrist_pitch only.
            q = [0.0] * 6
            q[0] = args.amplitude * math.sin(2.0 * math.pi * 0.20 * t)
            q[3] = 0.5 * args.amplitude * math.sin(2.0 * math.pi * 0.25 * t)
            obs = bot.move_joints(q)
            if int(t * 10) % 10 == 0:
                s = obs["state"][0, :6].tolist()
                print(f"t={t:5.2f}s joints={tuple(round(v,3) for v in s)}")
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print("Returning home and closing...")
        bot.reset()
        bot.close()


if __name__ == "__main__":
    main()
