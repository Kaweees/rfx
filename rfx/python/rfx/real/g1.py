"""
rfx.real.g1 - Unitree G1 humanoid hardware backend

Uses unitree_sdk2py with the unitree_hg (humanoid) message types.
29 active DOF across 35-slot motor arrays.

Joint ordering (G1JointIndex):
  left_leg[0:6], right_leg[6:12], waist[12:15],
  left_arm[15:22], right_arm[22:29]

References:
  - unitree_sdk2py/example/g1/low_level/g1_low_level_example.py
  - ExtremControl teleop pipeline (PD gains, default pose)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import torch

from ..observation import make_observation
from ..robot.config import G1_CONFIG, RobotConfig

if TYPE_CHECKING:
    pass

NUM_MOTORS = 29
# G1 motor arrays have 35 slots; only indices 0-28 are active
NUM_MOTOR_SLOTS = 35

# Default standing pose (radians) — matches ExtremControl default_dof_pos
G1_DEFAULT_DOF_POS = [
    # Left leg (0-5): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    # Right leg (6-11)
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    # Waist (12-14): yaw, roll, pitch
    0.0,
    0.0,
    0.0,
    # Left arm (15-21): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #                    wrist_roll, wrist_pitch, wrist_yaw
    0.3,
    0.2,
    0.0,
    -0.5,
    0.0,
    0.0,
    0.0,
    # Right arm (22-28)
    0.3,
    -0.2,
    0.0,
    -0.5,
    0.0,
    0.0,
    0.0,
]

# Per-joint PD gains from ExtremControl
# Format: kp values for all 29 joints
G1_KP = [
    # Left leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    100.0,
    100.0,
    100.0,
    100.0,
    30.0,
    30.0,
    # Right leg
    100.0,
    100.0,
    100.0,
    100.0,
    30.0,
    30.0,
    # Waist: yaw, roll, pitch
    100.0,
    150.0,
    150.0,
    # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    #           wrist_roll, wrist_pitch, wrist_yaw
    30.0,
    30.0,
    11.0,
    15.0,
    10.0,
    13.0,
    12.0,
    # Right arm
    30.0,
    30.0,
    11.0,
    15.0,
    10.0,
    13.0,
    12.0,
]

G1_KD = [
    # Left leg
    20.0,
    20.0,
    20.0,
    20.0,
    6.0,
    6.0,
    # Right leg
    20.0,
    20.0,
    20.0,
    20.0,
    6.0,
    6.0,
    # Waist
    10.0,
    10.0,
    10.0,
    # Left arm
    6.0,
    6.0,
    2.2,
    3.0,
    2.0,
    2.6,
    2.4,
    # Right arm
    6.0,
    6.0,
    2.2,
    3.0,
    2.0,
    2.6,
    2.4,
]


class G1Backend:
    """Unitree G1 humanoid hardware backend using unitree_sdk2py.

    Uses unitree_hg message types (NOT unitree_go).
    Echoes mode_machine from LowState_ and computes CRC for each LowCmd_.
    """

    _channel_initialized = False

    def __init__(
        self,
        config: RobotConfig,
        ip_address: str = "192.168.123.161",
        kp: list[float] | None = None,
        kd: list[float] | None = None,
        **kwargs,
    ):
        self.config = config
        self.ip_address = ip_address
        self._kp = kp or list(G1_KP)
        self._kd = kd or list(G1_KD)
        self._latest_lowstate = None
        self._state_lock = threading.Lock()
        self._connected = False

        self._init_sdk()

    def _init_sdk(self) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as HgLowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as HgLowState_
            from unitree_sdk2py.utils.crc import CRC
        except ImportError as err:
            raise ImportError(
                "unitree_sdk2py is required for G1 backend. "
                "Install with: pip install unitree_sdk2py"
            ) from err

        self._HgLowCmd_ = HgLowCmd_
        self._crc = CRC()

        if not G1Backend._channel_initialized:
            ChannelFactoryInitialize(0)
            G1Backend._channel_initialized = True

        # State subscriber
        def _on_state(msg):
            with self._state_lock:
                self._latest_lowstate = msg

        self._state_sub = ChannelSubscriber("rt/lowstate", HgLowState_)
        self._state_sub.Init(_on_state, 10)

        # Command publisher
        self._cmd_pub = ChannelPublisher("rt/lowcmd", HgLowCmd_)
        self._cmd_pub.Init()

        self._connected = True

    def _get_mode_machine(self) -> int:
        """Read mode_machine from latest state to echo back in commands."""
        with self._state_lock:
            if self._latest_lowstate is not None:
                return self._latest_lowstate.mode_machine
        return 0

    def _build_cmd(
        self,
        positions: list[float],
        velocities: list[float] | None = None,
        kp: list[float] | None = None,
        kd: list[float] | None = None,
    ):
        """Build a LowCmd_ with CRC and mode_machine."""
        cmd = self._HgLowCmd_()
        cmd.mode_pr = 0  # position control mode
        cmd.mode_machine = self._get_mode_machine()

        _kp = kp or self._kp
        _kd = kd or self._kd
        _vel = velocities or [0.0] * NUM_MOTORS

        for i in range(NUM_MOTORS):
            cmd.motor_cmd[i].mode = 1  # servo mode
            cmd.motor_cmd[i].q = float(positions[i])
            cmd.motor_cmd[i].dq = float(_vel[i])
            cmd.motor_cmd[i].kp = float(_kp[i])
            cmd.motor_cmd[i].kd = float(_kd[i])
            cmd.motor_cmd[i].tau = 0.0

        # Zero out unused motor slots (29-34)
        for i in range(NUM_MOTORS, NUM_MOTOR_SLOTS):
            cmd.motor_cmd[i].mode = 0
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        # Compute and set CRC
        cmd.crc = self._crc.Crc(cmd)
        return cmd

    def is_connected(self) -> bool:
        return self._connected

    def observe(self) -> dict[str, torch.Tensor]:
        with self._state_lock:
            low_state = self._latest_lowstate

        if low_state is None:
            positions = torch.zeros(NUM_MOTORS, dtype=torch.float32)
            velocities = torch.zeros(NUM_MOTORS, dtype=torch.float32)
            orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            gyroscope = torch.zeros(3, dtype=torch.float32)
            foot_contacts = torch.zeros(4, dtype=torch.float32)
        else:
            positions = torch.tensor(
                [low_state.motor_state[i].q for i in range(NUM_MOTORS)],
                dtype=torch.float32,
            )
            velocities = torch.tensor(
                [low_state.motor_state[i].dq for i in range(NUM_MOTORS)],
                dtype=torch.float32,
            )
            imu = low_state.imu_state
            orientation = torch.tensor(list(imu.quaternion), dtype=torch.float32)
            gyroscope = torch.tensor(list(imu.gyroscope), dtype=torch.float32)
            # Foot contact sensors (binary threshold on force)
            foot_contacts = torch.tensor(
                [low_state.foot_force[i] > 10.0 for i in range(4)],
                dtype=torch.float32,
            )

        # state_dim=69: 29 pos + 29 vel + 4 quat + 3 gyro + 4 foot_contacts
        raw_state = torch.cat(
            [positions, velocities, orientation, gyroscope, foot_contacts]
        ).unsqueeze(0)

        return make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

    def observe_raw(self) -> dict[str, torch.Tensor]:
        """Return raw (unpadded) observation tensors for the teleop obs builder."""
        with self._state_lock:
            low_state = self._latest_lowstate

        if low_state is None:
            return {
                "dof_pos": torch.zeros(NUM_MOTORS, dtype=torch.float32),
                "dof_vel": torch.zeros(NUM_MOTORS, dtype=torch.float32),
                "quat": torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "ang_vel": torch.zeros(3, dtype=torch.float32),
                "foot_contacts": torch.zeros(4, dtype=torch.float32),
            }

        return {
            "dof_pos": torch.tensor(
                [low_state.motor_state[i].q for i in range(NUM_MOTORS)],
                dtype=torch.float32,
            ),
            "dof_vel": torch.tensor(
                [low_state.motor_state[i].dq for i in range(NUM_MOTORS)],
                dtype=torch.float32,
            ),
            "quat": torch.tensor(list(low_state.imu_state.quaternion), dtype=torch.float32),
            "ang_vel": torch.tensor(list(low_state.imu_state.gyroscope), dtype=torch.float32),
            "foot_contacts": torch.tensor(
                [low_state.foot_force[i] > 10.0 for i in range(4)],
                dtype=torch.float32,
            ),
        }

    def act(self, action: torch.Tensor) -> None:
        """Send position commands. action shape: (1, >=29) or (29,)."""
        if action.dim() == 2:
            positions = action[0, :NUM_MOTORS].cpu().tolist()
        else:
            positions = action[:NUM_MOTORS].cpu().tolist()

        cmd = self._build_cmd(positions)
        self._cmd_pub.Write(cmd)

    def act_with_vel(
        self,
        positions: list[float],
        velocities: list[float],
    ) -> None:
        """Send position + velocity feedforward commands (used by teleop)."""
        cmd = self._build_cmd(positions, velocities)
        self._cmd_pub.Write(cmd)

    def reset(self) -> dict[str, torch.Tensor]:
        """Ramp to default standing position over 200 steps (2 seconds)."""
        with self._state_lock:
            low_state = self._latest_lowstate

        if low_state is not None:
            current = [low_state.motor_state[i].q for i in range(NUM_MOTORS)]
        else:
            current = list(G1_DEFAULT_DOF_POS)

        target = list(G1_DEFAULT_DOF_POS)
        ramp_steps = 200

        for step in range(ramp_steps):
            alpha = (step + 1) / ramp_steps
            interp = [c + alpha * (t - c) for c, t in zip(current, target, strict=False)]
            cmd = self._build_cmd(interp)
            self._cmd_pub.Write(cmd)
            time.sleep(0.01)

        return self.observe()

    def go_home(self) -> None:
        self.reset()

    def disconnect(self) -> None:
        if not self._connected:
            return
        # Ramp torques to zero gently
        try:
            cmd = self._HgLowCmd_()
            cmd.mode_pr = 0
            cmd.mode_machine = self._get_mode_machine()
            for i in range(NUM_MOTOR_SLOTS):
                cmd.motor_cmd[i].mode = 0  # idle
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0
            cmd.crc = self._crc.Crc(cmd)
            self._cmd_pub.Write(cmd)
        except Exception:
            pass
        self._connected = False


class G1Robot:
    """Convenience class for Unitree G1 humanoid robot.

    Examples:
        >>> g1 = G1Robot()
        >>> obs = g1.observe()
        >>> g1.act(action)
    """

    def __new__(
        cls,
        ip_address: str = "192.168.123.161",
        kp: list[float] | None = None,
        kd: list[float] | None = None,
        **kwargs,
    ):
        from .base import RealRobot

        return RealRobot(
            config=G1_CONFIG,
            robot_type="g1",
            ip_address=ip_address,
            kp=kp,
            kd=kd,
            **kwargs,
        )
