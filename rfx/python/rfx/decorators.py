"""
Decorators for control loops and neural policies.

Provides decorators for defining control loops and tinygrad-based neural policies.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from .jit import PolicyJitRuntime

P = ParamSpec("P")
R = TypeVar("R")

try:
    from tinygrad.engine.jit import TinyJit

    TINYGRAD_AVAILABLE = True
except ImportError:

    def TinyJit(x):
        return x  # no-op if tinygrad not available

    TINYGRAD_AVAILABLE = False


def control_loop(
    rate_hz: float = 500.0,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to mark a function as a control loop callback.

    The decorated function should accept (state) and return a command.
    The loop will run at the specified rate.

    Args:
        rate_hz: Target loop rate in Hz (default: 500)
        name: Optional name for the loop

    Example:
        >>> @rfx.control_loop(rate_hz=500)
        >>> def balance_policy(state: rfx.Go2State) -> rfx.MotorCommands:
        ...     roll_error = state.imu.roll
        ...     return rfx.MotorCommands.from_positions({
        ...         "FL_hip": -0.5 * roll_error,
        ...         "FR_hip": -0.5 * roll_error,
        ...     })
        >>>
        >>> go2.run(balance_policy, timeout=30.0)
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return func(*args, **kwargs)

        wrapper._rfx_control_loop = True  # type: ignore[attr-defined]
        wrapper._rfx_rate_hz = rate_hz  # type: ignore[attr-defined]
        wrapper._rfx_name = name or func.__name__  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


def policy(
    model: str | None = None,
    jit: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to mark a function as a neural policy.

    When jit=True, the function will be JIT compiled using tinygrad's TinyJit.
    If `RFX_JIT=1`, NumPy tensor calls are routed through `rfxJIT`.

    Args:
        model: Optional model path. If provided, raises NotImplementedError at runtime.
        jit: Whether to JIT compile the policy (default: False)

    Example:
        >>> from tinygrad import Tensor
        >>> import rfx
        >>>
        >>> @rfx.policy(jit=True)
        >>> def walking_policy(obs: Tensor) -> Tensor:
        ...     return obs @ weights
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        runtime = None  # type: PolicyJitRuntime | None

        if model is not None:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "Neural network policies from model files are not implemented yet."
                )
        elif jit:
            if TINYGRAD_AVAILABLE:
                fallback = TinyJit(func)
            else:
                fallback = func

            runtime = PolicyJitRuntime(
                func,
                fallback=fallback,
                name=func.__name__,
            )

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return runtime(*args, **kwargs)
        else:

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

        wrapper._rfx_policy = True  # type: ignore[attr-defined]
        wrapper._rfx_jit = jit  # type: ignore[attr-defined]
        wrapper._rfx_model = model  # type: ignore[attr-defined]
        wrapper._rfx_jit_backend = runtime.backend if runtime is not None else "disabled"  # type: ignore[attr-defined]
        wrapper._rfx_jit_runtime = runtime  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return decorator


class MotorCommands:
    """
    Motor commands to send to the robot.

    A convenience class for constructing motor command arrays from
    named positions, velocities, or torques.
    """

    def __init__(
        self,
        positions: dict[str, float] | None = None,
        velocities: dict[str, float] | None = None,
        torques: dict[str, float] | None = None,
        kp: float = 20.0,
        kd: float = 0.5,
    ) -> None:
        self.positions = positions or {}
        self.velocities = velocities or {}
        self.torques = torques or {}
        self.kp = kp
        self.kd = kd

    @classmethod
    def from_positions(
        cls,
        positions: dict[str, float],
        kp: float = 20.0,
        kd: float = 0.5,
    ) -> MotorCommands:
        """Create commands from named positions."""
        return cls(positions=positions, kp=kp, kd=kd)

    @classmethod
    def from_velocities(
        cls,
        velocities: dict[str, float],
        kd: float = 0.5,
    ) -> MotorCommands:
        """Create commands from named velocities."""
        return cls(velocities=velocities, kd=kd)

    @classmethod
    def from_torques(
        cls,
        torques: dict[str, float],
    ) -> MotorCommands:
        """Create commands from named torques."""
        return cls(torques=torques)

    def to_array(self, num_motors: int = 12) -> list[float]:
        """Convert to position array."""
        from . import motor_index_by_name

        result = [0.0] * num_motors
        for name, value in self.positions.items():
            idx = motor_index_by_name(name)
            if idx is not None:
                result[idx] = value
        return result

    def __repr__(self) -> str:
        parts = []
        if self.positions:
            parts.append(f"positions={self.positions}")
        if self.velocities:
            parts.append(f"velocities={self.velocities}")
        if self.torques:
            parts.append(f"torques={self.torques}")
        return f"MotorCommands({', '.join(parts)})"
