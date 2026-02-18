#!/usr/bin/env python3
"""
Low-level balance control on the Unitree Go2 (EDU mode).

This example uses the rfx v1 API for direct motor control. EDU mode
gives you per-joint position/torque commands at 500 Hz instead of the
high-level sport-mode walk/stand API.

Key concepts:
    rfx.Go2Config        — connection settings (IP, EDU mode toggle)
    rfx.Pid              — Rust-backed PID controller
    rfx.run_control_loop — Rust-backed fixed-rate loop with jitter stats
    rfx.motor_idx        — named motor indices (FR_HIP, FL_THIGH, etc.)

Hardware:
    Requires the EDU version of the Go2 firmware.

Usage:
    uv run rfx/examples/balance.py
"""

import rfx


def main():
    # -------------------------------------------------------------------
    # 1. Connect in EDU mode
    #
    # EDU mode unlocks per-motor position commands. The default sport mode
    # only exposes walk/stand. Pass .with_edu_mode() to enable it.
    # -------------------------------------------------------------------
    print("Connecting to Go2 in EDU mode...")
    config = rfx.Go2Config("192.168.123.161").with_edu_mode()
    go2 = rfx.Go2.connect(config)

    if not go2.is_connected():
        print("Failed to connect!")
        return

    print(f"Connected: {go2}")

    # -------------------------------------------------------------------
    # 2. Create PID controllers for roll and pitch stabilisation
    #
    # rfx.Pid is a Rust-backed PID with optional integral/output limits.
    # -------------------------------------------------------------------
    roll_pid = rfx.Pid.pid(kp=50.0, ki=0.5, kd=2.0)
    pitch_pid = rfx.Pid.pid(kp=50.0, ki=0.5, kd=2.0)

    KP = 20.0  # motor position gain
    KD = 0.5   # motor damping gain
    max_iterations = 5000  # ~10 s at 500 Hz

    # -------------------------------------------------------------------
    # 3. Define the control callback
    #
    # rfx.run_control_loop calls this function at a fixed rate (500 Hz).
    # Return True to keep running, False to stop.
    # -------------------------------------------------------------------
    def balance_callback(iteration: int, dt: float) -> bool:
        state = go2.state()
        roll = state.imu.roll
        pitch = state.imu.pitch

        roll_correction = roll_pid.update(setpoint=0.0, measurement=roll, dt=dt)
        pitch_correction = pitch_pid.update(setpoint=0.0, measurement=pitch, dt=dt)

        # Apply roll correction to hip abduction joints
        for motor, sign in [
            (rfx.motor_idx.FR_HIP, -1),
            (rfx.motor_idx.FL_HIP, +1),
            (rfx.motor_idx.RR_HIP, -1),
            (rfx.motor_idx.RL_HIP, +1),
        ]:
            go2.set_motor_position(
                motor,
                position=float(sign * roll_correction * 0.1),
                kp=KP,
                kd=KD,
            )

        if iteration % 100 == 0:
            print(
                f"[{iteration:5d}] roll={roll * 57.3:6.2f} pitch={pitch * 57.3:6.2f} "
                f"dt={dt * 1000:.2f}ms"
            )

        return iteration < max_iterations

    # -------------------------------------------------------------------
    # 4. Run the control loop
    #
    # rfx.run_control_loop is Rust-backed. It handles timing, jitter
    # tracking, and returns stats when the callback returns False or
    # the loop is interrupted.
    # -------------------------------------------------------------------
    print("Starting balance control at 500 Hz... (Ctrl+C to stop)\n")

    try:
        stats = rfx.run_control_loop(
            rate_hz=500.0,
            callback=balance_callback,
            name="balance",
        )

        print(f"\nDone — {stats.iterations} iterations, {stats.overruns} overruns")
        print(f"  avg dt: {stats.avg_iteration_time_ms:.3f} ms")
        print(f"  max dt: {stats.max_iteration_time_ms:.3f} ms")

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        go2.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
