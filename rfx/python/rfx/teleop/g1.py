"""
rfx.teleop.g1 - G1 humanoid VR teleoperation session.

Orchestrates VR pose reading, Cartesian retargeting, policy inference,
safety checks, and robot command sending in a 50Hz control loop matching
the ExtremControl pipeline.

Action processing: q_target = action * action_scale + default_dof_pos

Example:
    >>> from rfx.real import G1Robot
    >>> from rfx.teleop.g1 import G1TeleopSession
    >>> robot = G1Robot()
    >>> policy = rfx.load_policy("runs/g1-extremcontrol")
    >>> session = G1TeleopSession(robot, policy)
    >>> session.run()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..robot.config import G1_CONFIG, RobotConfig

if TYPE_CHECKING:
    from .g1_obs import G1ObservationBuilder
    from .retarget import RetargetBase
    from .vr import VRMotionPublisher

NUM_MOTORS = 29


# ---------------------------------------------------------------------------
# Safety layer
# ---------------------------------------------------------------------------


@dataclass
class G1SafetyConfig:
    """Configuration for G1 safety limits."""

    ramp_up_steps: int = 50
    enforce_joint_limits: bool = True
    max_joint_velocity: float = 2.0  # rad/step
    action_scale: float = 0.15  # ExtremControl default
    velocity_eta: float = 0.9  # feedforward velocity smoothing


class G1SafetyLayer:
    """Applies safety constraints to G1 actions before sending to hardware.

    Ramp-up blends from DEFAULT STANDING POSE (not zero).
    Action processing: q_target = raw_action * action_scale + default_dof_pos

    Pipeline: scale -> offset -> ramp-up -> joint limits -> velocity limits -> e-stop.
    """

    def __init__(
        self,
        config: G1SafetyConfig | None = None,
        robot_config: RobotConfig | None = None,
    ):
        self.config = config or G1SafetyConfig()
        self._robot_config = robot_config or G1_CONFIG

        from ..real.g1 import G1_DEFAULT_DOF_POS

        self._default_pos = torch.tensor(G1_DEFAULT_DOF_POS, dtype=torch.float32)
        self._step_count = 0
        self._estopped = False
        self._last_target: torch.Tensor | None = None

        # Build limit tensors from joint configs
        joints = self._robot_config.joints
        if joints:
            self._pos_min = torch.tensor([j.position_min for j in joints], dtype=torch.float32)
            self._pos_max = torch.tensor([j.position_max for j in joints], dtype=torch.float32)
        else:
            self._pos_min = torch.full((NUM_MOTORS,), -3.14159)
            self._pos_max = torch.full((NUM_MOTORS,), 3.14159)

    def apply(self, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply safety constraints.

        Args:
            raw_action: Raw policy output, shape (1, >=29) or (29,).

        Returns:
            (position_target, velocity_feedforward) — both shape (29,).
        """
        if self._estopped:
            target = (
                self._last_target if self._last_target is not None else self._default_pos.clone()
            )
            return target, torch.zeros(NUM_MOTORS, dtype=torch.float32)

        if raw_action.dim() == 2:
            act = raw_action[0, :NUM_MOTORS].clone()
        else:
            act = raw_action[:NUM_MOTORS].clone()

        # 1. Scale + offset: q_target = action * action_scale + default_dof_pos
        target = act * self.config.action_scale + self._default_pos

        # 2. Ramp-up: blend from default pose
        if self._step_count < self.config.ramp_up_steps:
            alpha = (self._step_count + 1) / self.config.ramp_up_steps
            target = self._default_pos + alpha * (target - self._default_pos)

        # 3. Joint limits
        if self.config.enforce_joint_limits:
            target = torch.clamp(target, self._pos_min, self._pos_max)

        # 4. Velocity limits
        if self._last_target is not None:
            delta = target - self._last_target
            delta = torch.clamp(
                delta,
                -self.config.max_joint_velocity,
                self.config.max_joint_velocity,
            )
            target = self._last_target + delta

        # Velocity feedforward: v = eta * (target - last_target) * rate_hz
        if self._last_target is not None:
            vel_ff = self.config.velocity_eta * (target - self._last_target)
        else:
            vel_ff = torch.zeros(NUM_MOTORS, dtype=torch.float32)

        self._last_target = target.clone()
        self._step_count += 1

        return target, vel_ff

    def estop(self) -> None:
        """Activate emergency stop — freezes output at last safe target."""
        self._estopped = True

    def release_estop(self) -> None:
        """Release emergency stop and reset ramp-up counter."""
        self._estopped = False
        self._step_count = 0

    def reset(self) -> None:
        """Reset internal state."""
        self._step_count = 0
        self._estopped = False
        self._last_target = None

    @property
    def is_estopped(self) -> bool:
        return self._estopped


# ---------------------------------------------------------------------------
# Teleop session orchestrator
# ---------------------------------------------------------------------------


@dataclass
class G1TeleopConfig:
    """Configuration for G1 VR teleop session."""

    rate_hz: float = 50.0  # ExtremControl default
    warmup_s: float = 1.0
    calibration_s: float = 3.0  # seconds to hold T-pose for calibration
    max_timing_samples: int = 10_000
    record: bool = False
    record_dir: str = "demos/"


class G1TeleopSession:
    """Orchestrates G1 humanoid VR teleoperation.

    Control loop (50Hz):
      1. Read VR poses
      2. Cartesian retarget to 6 link SE(3) targets
      3. Get raw robot observation
      4. Build ExtremControl observation vector
      5. Run policy inference (torch.no_grad)
      6. Apply safety layer (scale + offset + ramp + limits)
      7. Send position + velocity commands to robot
      8. Update history buffers

    Args:
        robot: G1 robot instance.
        policy: Callable that maps obs dict -> action tensor.
        config: Teleop session configuration.
        safety_config: Safety layer configuration.
        retargeter: Cartesian retargeter (defaults to G1Retargeter).
        obs_builder: Observation builder (defaults to G1ObservationBuilder).
        vr_publisher: VR motion publisher (defaults to VRMotionPublisher).
        transport: Optional rfx transport for data publishing.
    """

    def __init__(
        self,
        robot: Any,
        policy: Any,
        config: G1TeleopConfig | None = None,
        safety_config: G1SafetyConfig | None = None,
        retargeter: RetargetBase | None = None,
        obs_builder: G1ObservationBuilder | None = None,
        vr_publisher: VRMotionPublisher | None = None,
        transport: Any | None = None,
    ):
        self._robot = robot
        self._policy = policy
        self.config = config or G1TeleopConfig()
        self._safety_config = safety_config
        self._target_period = 1.0 / self.config.rate_hz

        # Components (lazy-init if not provided)
        self._safety = G1SafetyLayer(config=safety_config)
        self._retargeter = retargeter
        self._obs_builder = obs_builder
        self._vr_publisher = vr_publisher
        self._transport = transport

        # Threading
        self._control_thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stop_requested = threading.Event()
        self._loop_error: Exception | None = None

        # Timing stats
        self._lock = threading.Lock()
        self._period_samples: deque[float] = deque(maxlen=self.config.max_timing_samples)
        self._jitter_samples: deque[float] = deque(maxlen=self.config.max_timing_samples)
        self._iterations = 0
        self._overruns = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        """Start the teleop session."""
        if self._running.is_set():
            return

        self._loop_error = None
        self._stop_requested.clear()

        # Lazy-init components
        if self._obs_builder is None:
            from .g1_obs import G1ObservationBuilder

            self._obs_builder = G1ObservationBuilder()

        if self._retargeter is None:
            from .retarget import G1Retargeter

            self._retargeter = G1Retargeter()

        if self._vr_publisher is None:
            from .vr import VRMotionPublisher

            self._vr_publisher = VRMotionPublisher()

        # Reset components
        self._safety.reset()
        self._obs_builder.reset()

        # Start VR publisher
        self._vr_publisher.start()

        # Reset robot to standing
        self._robot.reset()

        # Calibration phase
        if hasattr(self._retargeter, "calibrate"):
            print(f"Hold T-pose for calibration ({self.config.calibration_s}s)...")
            time.sleep(self.config.calibration_s)
            vr_poses = self._vr_publisher.latest_poses
            self._retargeter.calibrate(vr_poses)
            print("Calibration complete.")

        if self.config.warmup_s > 0:
            time.sleep(self.config.warmup_s)

        # Start control loop
        self._running.set()
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="g1-teleop-control",
            daemon=True,
        )
        self._control_thread.start()

    def stop(self) -> None:
        """Stop the teleop session."""
        if not self._running.is_set() and self._control_thread is None:
            return

        self._stop_requested.set()
        self._running.clear()

        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

        if self._vr_publisher is not None:
            self._vr_publisher.stop()

    def run(self, duration: float | None = None) -> None:
        """Blocking run (matches Session.run() pattern)."""
        if not self._running.is_set():
            self.start()

        try:
            if duration is None:
                while self._running.is_set():
                    self._check_health()
                    time.sleep(0.05)
            else:
                deadline = time.perf_counter() + duration
                while self._running.is_set() and time.perf_counter() < deadline:
                    self._check_health()
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

        self._check_health()

    def estop(self) -> None:
        """Activate emergency stop."""
        self._safety.estop()

    def release_estop(self) -> None:
        """Release emergency stop."""
        self._safety.release_estop()

    def _check_health(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError("Control loop failed") from self._loop_error

    def _control_loop(self) -> None:
        target_period = self._target_period
        next_deadline = time.perf_counter()
        last_loop_start = next_deadline

        # Check if robot backend supports raw observations
        backend = getattr(self._robot, "_backend", None)
        has_raw_observe = hasattr(backend, "observe_raw")
        has_act_with_vel = hasattr(backend, "act_with_vel")

        while self._running.is_set() and not self._stop_requested.is_set():
            loop_start = time.perf_counter()
            dt = loop_start - last_loop_start
            last_loop_start = loop_start

            try:
                # 1. Read VR poses
                vr_poses = self._vr_publisher.latest_poses

                # 2. Cartesian retarget to 6 SE(3) link targets
                retarget_targets = self._retargeter.retarget(vr_poses=vr_poses)

                # Compute tracking deltas for observation
                tracking_deltas = None
                if hasattr(self._retargeter, "get_tracking_deltas"):
                    tracking_deltas = self._retargeter.get_tracking_deltas(retarget_targets)

                # 3. Get raw robot observation
                if has_raw_observe:
                    raw_obs = backend.observe_raw()
                else:
                    # Fallback: use standard observe and extract
                    obs = self._robot.observe()
                    state = obs["state"][0]
                    raw_obs = {
                        "dof_pos": state[:29],
                        "dof_vel": state[29:58],
                        "quat": state[58:62],
                        "ang_vel": state[62:65],
                        "foot_contacts": state[65:69],
                    }

                # 4. Build ExtremControl observation
                policy_obs = self._obs_builder.build(
                    raw_obs=raw_obs,
                    tracking_deltas=tracking_deltas,
                )

                # 5. Policy inference
                with torch.no_grad():
                    raw_action = self._policy(policy_obs)

                # 6. Safety layer (scale + offset + ramp + limits)
                pos_target, vel_ff = self._safety.apply(raw_action)

                # 7. Send to robot
                if has_act_with_vel:
                    backend.act_with_vel(
                        positions=pos_target.cpu().tolist(),
                        velocities=vel_ff.cpu().tolist(),
                    )
                else:
                    action_padded = torch.zeros(
                        1,
                        self._robot._action_dim if hasattr(self._robot, "_action_dim") else 29,
                        dtype=torch.float32,
                    )
                    action_padded[0, :NUM_MOTORS] = pos_target
                    self._robot.act(action_padded)

                # 8. Update history
                self._obs_builder.update_action(raw_action)

                # Publish to transport if available
                if self._transport is not None:
                    ts = time.time_ns()
                    self._transport.publish(
                        key="g1/action",
                        payload=memoryview(pos_target.numpy()),
                        timestamp_ns=ts,
                    )

            except Exception as exc:
                self._loop_error = exc
                self._running.clear()
                break

            # Timing bookkeeping
            with self._lock:
                self._iterations += 1
                if dt > target_period:
                    self._overruns += 1
                jitter = abs(dt - target_period)
                self._period_samples.append(dt)
                self._jitter_samples.append(jitter)

            # Hybrid sleep + busy-spin rate control
            next_deadline += target_period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                spin_window_s = 0.0012
                if sleep_s > spin_window_s:
                    time.sleep(sleep_s - spin_window_s)
                while time.perf_counter() < next_deadline:
                    pass
            else:
                next_deadline = time.perf_counter()

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            periods = np.asarray(tuple(self._period_samples), dtype=np.float64)
            jitters = np.asarray(tuple(self._jitter_samples), dtype=np.float64)
            iters = self._iterations
            overruns = self._overruns

        if periods.size == 0:
            return {
                "iterations": iters,
                "overruns": overruns,
                "target_period_s": self._target_period,
                "avg_period_s": 0.0,
                "p50_jitter_s": 0.0,
                "p95_jitter_s": 0.0,
                "max_jitter_s": 0.0,
            }
        return {
            "iterations": iters,
            "overruns": overruns,
            "target_period_s": self._target_period,
            "avg_period_s": float(np.mean(periods)),
            "p50_jitter_s": float(np.percentile(jitters, 50)),
            "p95_jitter_s": float(np.percentile(jitters, 95)),
            "max_jitter_s": float(np.max(jitters)),
        }

    def __enter__(self) -> G1TeleopSession:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
