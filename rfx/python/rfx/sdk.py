"""
Universal SDK entrypoint for rfx.

Goal: one simple API across simulation and real hardware backends.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal, Optional

from .envs import Go2Env
from .providers import use as use_providers

BackendName = Literal["mock", "genesis", "mjx", "real"]
RobotName = Literal["go2", "so101"]


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0


class _Go2SimCommandAdapter:
    def __init__(self, robot, gait_hz: float = 1.8):
        self.robot = robot
        self.gait_hz = gait_hz
        self._phase = 0.0
        self._dt = 1.0 / max(1.0, float(robot.config.control_freq_hz))

    def _build_action(self, cmd: VelocityCommand):
        import torch

        device = self.robot.device
        base = torch.tensor(Go2Env.DEFAULT_STANDING, dtype=torch.float32, device=device)
        low = torch.tensor(Go2Env.JOINT_LIMITS_LOW, dtype=torch.float32, device=device)
        high = torch.tensor(Go2Env.JOINT_LIMITS_HIGH, dtype=torch.float32, device=device)

        phase_offsets = [0.0, math.pi, math.pi, 0.0]  # FR FL RR RL
        side_sign = [-1.0, 1.0, -1.0, 1.0]

        stride_amp = 0.32 * max(0.0, min(1.0, abs(cmd.vx)))
        turn_amp = 0.18 * max(-1.0, min(1.0, cmd.yaw))
        lift_amp = 0.42 * max(0.0, min(1.0, abs(cmd.vx) + 0.15))

        target = base.clone()
        for leg in range(4):
            i = leg * 3
            phi = self._phase + phase_offsets[leg]
            s = math.sin(phi)
            c = math.cos(phi)
            lift = max(0.0, s)

            target[i] = base[i] + stride_amp * s + (turn_amp * side_sign[leg])
            target[i + 1] = base[i + 1] + 0.18 * stride_amp * c - lift_amp * lift
            target[i + 2] = base[i + 2] - 0.30 * stride_amp * c + 1.6 * lift_amp * lift

        target = torch.clamp(target, low, high)
        action = torch.zeros(1, self.robot.max_action_dim, dtype=torch.float32, device=device)
        action[0, :12] = target
        return action

    def step(self, cmd: VelocityCommand):
        action = self._build_action(cmd)
        self.robot.act(action)
        self._phase += 2.0 * math.pi * self.gait_hz * self._dt
        return self.robot.observe()


class UniversalRobot:
    """
    Wrapper that provides a single API for sim/hardware robot control.
    """

    def __init__(self, robot_name: RobotName, backend: BackendName, impl):
        self.robot_name = robot_name
        self.backend = backend
        self._impl = impl
        self._cmd = VelocityCommand()
        self._adapter = None
        self._last_action = None

        if self.robot_name == "go2" and backend in {"mock", "genesis", "mjx"}:
            self._adapter = _Go2SimCommandAdapter(impl)

    @property
    def impl(self):
        return self._impl

    def observe(self):
        return self._impl.observe()

    def act(self, action):
        self._last_action = action
        return self._impl.act(action)

    def reset(self):
        return self._impl.reset()

    def close(self) -> None:
        close_fn = getattr(self._impl, "close", None)
        if callable(close_fn):
            close_fn()
            return
        disconnect_fn = getattr(self._impl, "disconnect", None)
        if callable(disconnect_fn):
            disconnect_fn()

    def command(self, vx: float = 0.0, vy: float = 0.0, yaw: float = 0.0) -> None:
        self._cmd = VelocityCommand(vx=vx, vy=vy, yaw=yaw)

    def move_joints(self, positions: list[float] | tuple[float, ...]):
        """
        Command joint positions for joint-space robots (e.g. so101).
        """
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Torch is required for move_joints(). Install with: uv pip install torch"
            ) from exc

        action_dim = int(getattr(self._impl, "action_dim", len(positions)))
        max_action_dim = int(getattr(self._impl, "max_action_dim", action_dim))
        vals = list(positions)[:action_dim]
        if len(vals) < action_dim:
            vals += [0.0] * (action_dim - len(vals))

        device = str(getattr(self._impl, "device", "cpu"))
        action = torch.zeros(1, max_action_dim, dtype=torch.float32, device=device)
        action[0, :action_dim] = torch.tensor(vals, dtype=torch.float32, device=device)
        self.act(action)
        return self.observe()

    def step(self):
        """
        Step using the active command-level interface.

        - real go2: uses built-in Unitree gait command
        - sim go2: uses a gait adapter that outputs joint targets
        """
        if self.robot_name == "go2" and self.backend == "real":
            backend = getattr(self._impl, "_backend", None)
            if backend is None or not hasattr(backend, "walk"):
                raise RuntimeError("Real Go2 backend does not expose walk()")
            backend.walk(self._cmd.vx, self._cmd.vy, self._cmd.yaw)
            return self._impl.observe()

        if self._adapter is not None:
            return self._adapter.step(self._cmd)

        if self._last_action is not None:
            self._impl.act(self._last_action)
            return self._impl.observe()

        raise NotImplementedError(
            f"Command stepping not available for robot={self.robot_name} backend={self.backend}"
        )


def connect(
    robot: RobotName = "go2",
    backend: BackendName = "mock",
    config: Optional[str] = None,
    *,
    num_envs: int = 1,
    device: str = "cpu",
    **kwargs,
) -> UniversalRobot:
    """
    Universal connection helper.

    Examples:
        >>> bot = connect("go2", backend="genesis", config="rfx/configs/go2.yaml")
        >>> bot.reset()
        >>> bot.command(vx=0.5)
        >>> obs = bot.step()
    """
    from .real import RealRobot
    from .sim import SimRobot

    # Best-effort provider activation from the single `rfx` import surface.
    if backend in {"mock", "genesis", "mjx"}:
        use_providers("sim")
    if robot == "go2":
        use_providers("go2")

    if config is None:
        if robot == "go2":
            config = "rfx/configs/go2.yaml"
        else:
            config = "rfx/configs/so101.yaml"

    config_path = Path(config)
    if backend == "real":
        impl = RealRobot.from_config(config_path, **kwargs)
    else:
        impl = SimRobot.from_config(
            config_path,
            num_envs=num_envs,
            backend=backend,
            device=device,
            **kwargs,
        )
    return UniversalRobot(robot_name=robot, backend=backend, impl=impl)
