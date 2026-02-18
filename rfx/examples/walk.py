#!/usr/bin/env python3
"""
Walk the Unitree Go2 with high-level sport-mode commands.

rfx provides a v1 API for the Go2 that wraps Unitree's sport mode. You
get simple methods — stand(), walk(), sit() — without managing sockets
or control loops yourself.

Key concepts:
    rfx.Go2              — Rust-backed Go2 connection (v1 API)
    go2.walk(vx, vy, vyaw) — velocity command in m/s and rad/s
    go2.state()          — IMU, joint positions, battery, etc.

Usage:
    uv run rfx/examples/walk.py
"""

import time

import rfx


def main():
    # -------------------------------------------------------------------
    # 1. Connect
    #
    # rfx.Go2.connect() talks to the robot over UDP at 192.168.123.161.
    # Pass an IP string to override: rfx.Go2.connect("10.0.0.1")
    # -------------------------------------------------------------------
    print("Connecting to Go2...")
    go2 = rfx.Go2.connect()

    if not go2.is_connected():
        print("Failed to connect!")
        return

    print(f"Connected: {go2}")

    try:
        # ---------------------------------------------------------------
        # 2. Stand up
        # ---------------------------------------------------------------
        print("Standing...")
        go2.stand()
        time.sleep(1.0)

        state = go2.state()
        print(f"IMU: roll={state.imu.roll_deg:.1f}, pitch={state.imu.pitch_deg:.1f}")

        # ---------------------------------------------------------------
        # 3. Walk forward at 0.3 m/s for 2 seconds
        #
        # walk(vx, vy, vyaw):
        #   vx   — forward/backward (m/s)
        #   vy   — left/right strafe (m/s)
        #   vyaw — rotation (rad/s)
        # ---------------------------------------------------------------
        print("Walking forward...")
        go2.walk(vx=0.3, vy=0.0, vyaw=0.0)

        for _ in range(20):
            time.sleep(0.1)
            state = go2.state()
            print(f"  position={state.position}, imu={state.imu.rpy}")

        # ---------------------------------------------------------------
        # 4. Turn in place
        # ---------------------------------------------------------------
        print("Stopping...")
        go2.stand()
        time.sleep(0.5)

        print("Turning...")
        go2.walk(vx=0.0, vy=0.0, vyaw=0.5)
        time.sleep(2.0)

        # ---------------------------------------------------------------
        # 5. Stop
        # ---------------------------------------------------------------
        print("Standing...")
        go2.stand()
        time.sleep(0.5)

        state = go2.state()
        print(f"Final position: {state.position}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        go2.stand()
        go2.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
