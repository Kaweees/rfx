#!/usr/bin/env python3
"""
LLM agent controlling a Go2 with natural language.

Defines a set of @rfx.skill-decorated functions, then hands them to an
rfx.Agent backed by Claude or GPT. The agent picks which skills to
call based on free-form text commands.

Key concepts:
    @rfx.skill           — registers a function as an agent-callable skill
    rfx.Agent            — LLM-backed agent that maps text → skill calls
    rfx.Go2              — robot connection (v1 API)

Environment variables:
    ANTHROPIC_API_KEY    — for Claude models
    OPENAI_API_KEY       — for GPT models

Usage:
    ANTHROPIC_API_KEY=sk-... uv run rfx/examples/agent_control.py
"""

import os
import time

import rfx
from rfx.agent import Agent

go2 = None


# -------------------------------------------------------------------
# 1. Define skills
#
# Each @rfx.skill becomes a tool the LLM agent can call. The
# docstring is the tool description the LLM sees.
# -------------------------------------------------------------------


@rfx.skill
def walk_forward(distance: float = 1.0):
    """Walk forward by the specified distance in meters."""
    if go2 is None:
        return "Robot not connected"
    speed = 0.3
    go2.walk(vx=speed, vy=0, vyaw=0)
    time.sleep(distance / speed)
    go2.stand()
    return f"Walked forward {distance:.1f}m"


@rfx.skill
def walk_backward(distance: float = 1.0):
    """Walk backward by the specified distance in meters."""
    if go2 is None:
        return "Robot not connected"
    speed = 0.2
    go2.walk(vx=-speed, vy=0, vyaw=0)
    time.sleep(distance / speed)
    go2.stand()
    return f"Walked backward {distance:.1f}m"


@rfx.skill
def turn_left(angle: float = 90.0):
    """Turn left by the specified angle in degrees."""
    if go2 is None:
        return "Robot not connected"
    import math

    angular_speed = 0.5
    go2.walk(vx=0, vy=0, vyaw=angular_speed)
    time.sleep(math.radians(angle) / angular_speed)
    go2.stand()
    return f"Turned left {angle:.0f}"


@rfx.skill
def turn_right(angle: float = 90.0):
    """Turn right by the specified angle in degrees."""
    if go2 is None:
        return "Robot not connected"
    import math

    angular_speed = 0.5
    go2.walk(vx=0, vy=0, vyaw=-angular_speed)
    time.sleep(math.radians(angle) / angular_speed)
    go2.stand()
    return f"Turned right {angle:.0f}"


@rfx.skill
def look_around():
    """Rotate 360 degrees in place to survey surroundings."""
    if go2 is None:
        return "Robot not connected"
    go2.walk(vx=0, vy=0, vyaw=0.3)
    time.sleep(6.28 / 0.3)
    go2.stand()
    return "Completed 360 survey"


@rfx.skill
def stand():
    """Make the robot stand still."""
    if go2 is None:
        return "Robot not connected"
    go2.stand()
    return "Standing"


@rfx.skill
def sit():
    """Make the robot sit down."""
    if go2 is None:
        return "Robot not connected"
    go2.sit()
    return "Sitting"


@rfx.skill
def get_status() -> str:
    """Get the current robot IMU and position."""
    if go2 is None:
        return "Robot not connected"
    state = go2.state()
    return (
        f"Position: {state.position}, "
        f"IMU: roll={state.imu.roll_deg:.1f}, pitch={state.imu.pitch_deg:.1f}, yaw={state.imu.yaw_deg:.1f}"
    )


def main():
    global go2

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    demo_mode = not api_key
    if demo_mode:
        print("No API key found — running in demo mode.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY for real LLM control.\n")

    # -------------------------------------------------------------------
    # 2. Connect to the robot
    # -------------------------------------------------------------------
    print("Connecting to Go2...")
    go2 = rfx.Go2.connect()

    if not go2.is_connected():
        print("Robot not reachable — skills will return 'not connected'.\n")

    skills = [walk_forward, walk_backward, turn_left, turn_right, look_around, stand, sit, get_status]

    try:
        if demo_mode:
            # ---------------------------------------------------------
            # 3a. Demo mode — call skills directly
            # ---------------------------------------------------------
            from rfx.agent import MockAgent

            agent = MockAgent(skills=skills)
            print("Available skills:")
            print(agent.describe_skills())
            print()
            print(f"  walk_forward: {agent.execute_skill('walk_forward', distance=1.0)}")
            print(f"  turn_left:    {agent.execute_skill('turn_left', angle=45)}")
            print(f"  get_status:   {agent.execute_skill('get_status')}")
        else:
            # ---------------------------------------------------------
            # 3b. Real mode — LLM picks skills from text
            # ---------------------------------------------------------
            agent = Agent(
                model="claude-sonnet-4-20250514",
                skills=skills,
                robot=go2,
            )
            print("Agent ready. Type commands ('quit' to exit):\n")

            while True:
                try:
                    command = input("> ").strip()
                    if not command:
                        continue
                    if command.lower() in ("quit", "exit", "q"):
                        break
                    result = agent.execute(command)
                    print(f"  {result}\n")
                except EOFError:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        if go2 and go2.is_connected():
            go2.stand()
            go2.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
