"""
rfx.teleop.retarget - Cartesian-space retargeting from VR poses to G1 link targets.

ExtremControl does NOT use IK in the deployed loop. Instead, it uses
Cartesian-space mapping: VR hand/head/foot poses are scaled to match the
G1's proportions and output as SE(3) targets for 6 tracked links. The
policy itself learned to solve IK during RL training.

6 tracked links: left_ankle_roll, right_ankle_roll, left_wrist_yaw,
    right_wrist_yaw, torso, pelvis

Two implementations:
  - G1Retargeter: Full Cartesian retarget (calibration + proportional scaling).
  - SimpleRetargeter: Minimal arm-only mapping for quick testing.

References:
  - OpenTeleVision/TeleVision retargeting
  - ExtremControl Section 3.2 (Cartesian-space motion mapping)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

NUM_MOTORS = 29

# G1 physical constants (meters) from ExtremControl / OpenTeleVision
G1_SHOULDER_Y = 0.100        # lateral distance from torso center to shoulder
G1_ARM_LENGTH = 0.398         # upper arm + forearm total length
G1_PELVIS_Z = 0.769           # pelvis height in standing pose
G1_PELVIS_SHOULDER_Z = 0.289  # vertical distance pelvis -> shoulder
G1_PELVIS_TORSO_Z = 0.044     # vertical distance pelvis -> torso IMU
G1_LEG_LENGTH = 0.500         # approximate standing leg length

# Number of tracked links (SE(3) targets)
NUM_TRACKED_LINKS = 6
# Link names in order
TRACKED_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
    "pelvis",
]


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to 6D representation (first two columns)."""
    return R[:, :2].T.flatten()  # shape (6,)


def _rotation_to_rpy(R: np.ndarray) -> np.ndarray:
    """Extract roll, pitch, yaw from a 3x3 rotation matrix (XYZ convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _yaw_rotation(yaw: float) -> np.ndarray:
    """Create a 4x4 rotation matrix around Z axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.eye(4, dtype=np.float64)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


@dataclass
class RetargetConfig:
    """Configuration for Cartesian retargeting."""

    retarget_arms: bool = True
    retarget_head: bool = True
    retarget_legs: bool = True
    arm_scale: float | None = None  # auto-computed from calibration
    leg_scale: float | None = None
    lock_feet_to_ground: bool = True


@dataclass
class CalibrationData:
    """Stores human proportions from the T-pose calibration phase."""

    # Human measurements (meters)
    human_shoulder_y: float = 0.0
    human_arm_length: float = 0.0
    human_pelvis_z: float = 0.0
    human_pelvis_shoulder_z: float = 0.0
    human_leg_length: float = 0.0

    # Head reference for yaw alignment
    head_yaw_offset: float = 0.0
    head_position_ref: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    calibrated: bool = False


class RetargetBase(ABC):
    """Abstract base class for VR-to-robot retargeting."""

    @abstractmethod
    def retarget(
        self,
        vr_poses: dict[str, np.ndarray],
        current_joints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map VR poses to 6 SE(3) link targets.

        Args:
            vr_poses: Dict with "head", "left_hand", "right_hand" keys,
                      each a 4x4 homogeneous transform in Z-up frame.
            current_joints: Current 29-DOF joint positions (optional).

        Returns:
            Array of shape (6, 4, 4) — SE(3) targets for the 6 tracked links.
        """
        ...

    @abstractmethod
    def calibrate(self, vr_poses: dict[str, np.ndarray]) -> None:
        """Record a T-pose calibration frame."""
        ...


class G1Retargeter(RetargetBase):
    """Cartesian-space retargeter matching the ExtremControl pipeline.

    Maps VR poses to 6 link SE(3) targets using proportional scaling.
    Requires a calibration T-pose to measure human proportions.

    The output is NOT joint angles — the policy learned IK during training.
    """

    def __init__(self, config: RetargetConfig | None = None):
        self.config = config or RetargetConfig()
        self.calibration = CalibrationData()

        # Default standing targets (identity-ish poses at default link positions)
        self._default_targets = np.zeros((NUM_TRACKED_LINKS, 4, 4), dtype=np.float64)
        for i in range(NUM_TRACKED_LINKS):
            self._default_targets[i] = np.eye(4, dtype=np.float64)

        # Set default positions for each link
        # left_ankle_roll: at ground, left side
        self._default_targets[0][:3, 3] = [0.0, 0.05, 0.0]
        # right_ankle_roll: at ground, right side
        self._default_targets[1][:3, 3] = [0.0, -0.05, 0.0]
        # left_wrist_yaw: at side
        self._default_targets[2][:3, 3] = [0.0, G1_SHOULDER_Y + G1_ARM_LENGTH * 0.3, G1_PELVIS_Z + G1_PELVIS_SHOULDER_Z]
        # right_wrist_yaw: at side
        self._default_targets[3][:3, 3] = [0.0, -(G1_SHOULDER_Y + G1_ARM_LENGTH * 0.3), G1_PELVIS_Z + G1_PELVIS_SHOULDER_Z]
        # torso
        self._default_targets[4][:3, 3] = [0.0, 0.0, G1_PELVIS_Z + G1_PELVIS_TORSO_Z]
        # pelvis
        self._default_targets[5][:3, 3] = [0.0, 0.0, G1_PELVIS_Z]

    def calibrate(self, vr_poses: dict[str, np.ndarray]) -> None:
        """Record a T-pose to compute human proportions and scale factors.

        User should stand in T-pose (arms out, legs straight) and call this
        once with the VR poses.
        """
        head = vr_poses.get("head")
        left = vr_poses.get("left_hand")
        right = vr_poses.get("right_hand")

        if head is None or left is None or right is None:
            return

        head_pos = head[:3, 3]
        left_pos = left[:3, 3]
        right_pos = right[:3, 3]

        # Shoulder width ~ distance between hands * some factor
        # In T-pose, hands are at full arm extension from shoulders
        hand_span = np.linalg.norm(left_pos - right_pos)
        # Approximate: hand_span = 2 * (shoulder_y + arm_length)
        human_arm_plus_shoulder = hand_span / 2.0
        # Assume similar ratio as G1
        ratio = G1_SHOULDER_Y / (G1_SHOULDER_Y + G1_ARM_LENGTH)
        self.calibration.human_shoulder_y = human_arm_plus_shoulder * ratio
        self.calibration.human_arm_length = human_arm_plus_shoulder * (1 - ratio)

        # Head height -> pelvis estimate
        # Assume pelvis is ~55% of head height from ground
        self.calibration.human_pelvis_z = head_pos[2] * 0.55
        self.calibration.human_pelvis_shoulder_z = head_pos[2] * 0.15
        self.calibration.human_leg_length = head_pos[2] * 0.48

        # Head yaw reference
        head_rpy = _rotation_to_rpy(head[:3, :3])
        self.calibration.head_yaw_offset = head_rpy[2]
        self.calibration.head_position_ref = head_pos.copy()

        self.calibration.calibrated = True

    def retarget(
        self,
        vr_poses: dict[str, np.ndarray],
        current_joints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Map VR poses to 6 tracked link SE(3) targets.

        Returns:
            Array of shape (6, 4, 4) — SE(3) targets for tracked links.
        """
        if not self.calibration.calibrated:
            return self._default_targets.copy()

        head = vr_poses.get("head", np.eye(4, dtype=np.float64))
        left_hand = vr_poses.get("left_hand", np.eye(4, dtype=np.float64))
        right_hand = vr_poses.get("right_hand", np.eye(4, dtype=np.float64))

        targets = self._default_targets.copy()

        # --- Global alignment: extract head yaw for frame rotation ---
        head_rpy = _rotation_to_rpy(head[:3, :3])
        base_yaw = head_rpy[2] - self.calibration.head_yaw_offset
        R_yaw_inv = _yaw_rotation(-base_yaw)

        # Head position delta (relative to calibration reference)
        head_delta = head[:3, 3] - self.calibration.head_position_ref

        # --- Arm retargeting ---
        if self.config.retarget_arms:
            arm_scale = self.config.arm_scale
            if arm_scale is None and self.calibration.human_arm_length > 0.01:
                arm_scale = G1_ARM_LENGTH / self.calibration.human_arm_length
            else:
                arm_scale = arm_scale or 1.0

            for hand_pose, link_idx, sign in [
                (left_hand, 2, 1.0),   # left_wrist_yaw
                (right_hand, 3, -1.0),  # right_wrist_yaw
            ]:
                # Hand position relative to head, in local frame
                hand_pos_world = hand_pose[:3, 3]
                hand_rel = hand_pos_world - head[:3, 3]

                # Rotate into yaw-aligned frame
                hand_rel_local = R_yaw_inv[:3, :3] @ hand_rel

                # Scale to G1 proportions
                hand_rel_scaled = hand_rel_local * arm_scale

                # Place relative to G1 shoulder
                shoulder_pos = np.array([
                    0.0,
                    sign * G1_SHOULDER_Y,
                    G1_PELVIS_Z + G1_PELVIS_SHOULDER_Z,
                ])
                target_pos = shoulder_pos + hand_rel_scaled

                # Rotation: rotate hand orientation into yaw-aligned frame
                target_rot = R_yaw_inv[:3, :3] @ hand_pose[:3, :3]

                targets[link_idx][:3, :3] = target_rot
                targets[link_idx][:3, 3] = target_pos

        # --- Torso retargeting ---
        if self.config.retarget_head:
            # Torso tracks head orientation (pitch/roll) scaled down
            torso_rot = R_yaw_inv[:3, :3] @ head[:3, :3]
            targets[4][:3, :3] = torso_rot
            targets[4][:3, 3] = [0.0, 0.0, G1_PELVIS_Z + G1_PELVIS_TORSO_Z]

        # --- Pelvis ---
        # Pelvis tracks base yaw and XY position
        pelvis_R = _yaw_rotation(base_yaw)
        targets[5][:3, :3] = pelvis_R[:3, :3]

        # Scale XY translation
        if self.calibration.human_pelvis_z > 0.01:
            pos_scale = G1_PELVIS_Z / self.calibration.human_pelvis_z
        else:
            pos_scale = 1.0
        pelvis_xy = (R_yaw_inv[:3, :3] @ head_delta)[:2] * pos_scale
        targets[5][0, 3] = pelvis_xy[0]
        targets[5][1, 3] = pelvis_xy[1]
        targets[5][2, 3] = G1_PELVIS_Z

        # --- Lower body ---
        if self.config.retarget_legs:
            # Simple: feet stay at ground level, track pelvis XY
            for foot_idx in [0, 1]:  # left_ankle, right_ankle
                targets[foot_idx][:3, :3] = pelvis_R[:3, :3]
                targets[foot_idx][0, 3] = pelvis_xy[0]
                targets[foot_idx][1, 3] = self._default_targets[foot_idx][1, 3]
                if self.config.lock_feet_to_ground:
                    targets[foot_idx][2, 3] = 0.0
        else:
            # Keep feet at default
            targets[0] = self._default_targets[0].copy()
            targets[1] = self._default_targets[1].copy()

        return targets

    def get_tracking_deltas(
        self,
        targets: np.ndarray,
        current_link_poses: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute position and rotation deltas for the observation vector.

        Args:
            targets: Shape (6, 4, 4) — desired SE(3) link poses.
            current_link_poses: Shape (6, 4, 4) — current link poses from FK.
                If None, uses default standing poses.

        Returns:
            Dict with "pos_delta" (18,) and "rot_6d_delta" (36,).
        """
        if current_link_poses is None:
            current_link_poses = self._default_targets

        pos_deltas = []
        rot_6d_deltas = []

        for i in range(NUM_TRACKED_LINKS):
            # Position delta
            dp = targets[i][:3, 3] - current_link_poses[i][:3, 3]
            pos_deltas.append(dp)

            # Rotation delta in 6D representation
            R_current = current_link_poses[i][:3, :3]
            R_target = targets[i][:3, :3]
            R_delta = R_current.T @ R_target
            rot_6d_deltas.append(rotation_matrix_to_6d(R_delta))

        return {
            "pos_delta": np.concatenate(pos_deltas).astype(np.float32),  # (18,)
            "rot_6d_delta": np.concatenate(rot_6d_deltas).astype(np.float32),  # (36,)
        }


class SimpleRetargeter(RetargetBase):
    """Minimal arm-only Cartesian retargeter for testing without calibration.

    Maps VR hand positions directly to wrist link targets using a fixed scale.
    No calibration needed — uses hardcoded proportions.
    """

    def __init__(self, config: RetargetConfig | None = None):
        self.config = config or RetargetConfig()
        self._default_targets = np.zeros((NUM_TRACKED_LINKS, 4, 4), dtype=np.float64)
        for i in range(NUM_TRACKED_LINKS):
            self._default_targets[i] = np.eye(4, dtype=np.float64)

    def calibrate(self, vr_poses: dict[str, np.ndarray]) -> None:
        pass  # No calibration needed

    def retarget(
        self,
        vr_poses: dict[str, np.ndarray],
        current_joints: np.ndarray | None = None,
    ) -> np.ndarray:
        targets = self._default_targets.copy()

        left = vr_poses.get("left_hand")
        right = vr_poses.get("right_hand")

        if left is not None:
            targets[2][:3, :3] = left[:3, :3]
            targets[2][:3, 3] = left[:3, 3] * 0.5  # rough scale
            targets[2][2, 3] = max(targets[2][2, 3], 0.3)

        if right is not None:
            targets[3][:3, :3] = right[:3, :3]
            targets[3][:3, 3] = right[:3, 3] * 0.5
            targets[3][2, 3] = max(targets[3][2, 3], 0.3)

        return targets
