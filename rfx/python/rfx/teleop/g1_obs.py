"""
rfx.teleop.g1_obs - Observation builder matching the ExtremControl format.

Assembles the flat observation vector that ExtremControl-style policies expect:

  last_action(29)
  + dof_pos(29)
  + dof_vel * 0.1 (29)
  + base_ang_vel_local(3)
  + diff_base_yaw(1)
  + diff_base_pos_local_yaw(3)
  + diff_tracking_link_pos_local_yaw(18)     — 6 links * 3
  + diff_tracking_link_rotation_6D(36)       — 6 links * 6
  + projected_gravity(3)
  = 151 base dims (+ optional motion_obs)

6D rotation = first two columns of rotation matrix, flattened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass

NUM_MOTORS = 29
NUM_TRACKED_LINKS = 6
VEL_SCALE = 0.1

# Base observation dimensionality (without motion_obs)
BASE_OBS_DIM = 29 + 29 + 29 + 3 + 1 + 3 + 18 + 36 + 3  # = 151


def _quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
        ]
    )


def _projected_gravity(quat: torch.Tensor) -> torch.Tensor:
    """Compute projected gravity vector from quaternion (w, x, y, z)."""
    R = _quat_to_rotation_matrix(quat)
    # Gravity in world frame = [0, 0, -1]
    # In body frame = R^T @ [0, 0, -1]
    grav_body = R.T @ torch.tensor([0.0, 0.0, -1.0], dtype=quat.dtype)
    return grav_body


@dataclass
class G1ObsConfig:
    """Configuration for ExtremControl observation builder."""

    include_motion_obs: bool = False
    motion_obs_dim: int = 0
    default_dof_pos: list[float] | None = None

    @property
    def obs_dim(self) -> int:
        return BASE_OBS_DIM + self.motion_obs_dim


class G1ObservationBuilder:
    """Builds the ExtremControl-format observation vector.

    Call ``update_action()`` after each step and ``build()`` each frame
    to get the policy observation.
    """

    def __init__(self, config: G1ObsConfig | None = None):
        self.config = config or G1ObsConfig()

        if self.config.default_dof_pos is not None:
            self._default_dof_pos = torch.tensor(self.config.default_dof_pos, dtype=torch.float32)
        else:
            from ..real.g1 import G1_DEFAULT_DOF_POS

            self._default_dof_pos = torch.tensor(G1_DEFAULT_DOF_POS, dtype=torch.float32)

        self._last_action = torch.zeros(NUM_MOTORS, dtype=torch.float32)
        self._ref_base_yaw: float = 0.0
        self._ref_base_pos = torch.zeros(3, dtype=torch.float32)

    @property
    def obs_dim(self) -> int:
        return self.config.obs_dim

    def update_action(self, action: torch.Tensor) -> None:
        """Record the latest action (29-DOF)."""
        if action.dim() == 2:
            self._last_action = action[0, :NUM_MOTORS].detach().clone()
        else:
            self._last_action = action[:NUM_MOTORS].detach().clone()

    def set_reference_pose(self, base_yaw: float, base_pos: torch.Tensor) -> None:
        """Set the reference base yaw and position from retarget targets."""
        self._ref_base_yaw = base_yaw
        self._ref_base_pos = base_pos.clone()

    def build(
        self,
        raw_obs: dict[str, torch.Tensor],
        tracking_deltas: dict[str, np.ndarray] | None = None,
        motion_obs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Assemble the full ExtremControl observation vector.

        Args:
            raw_obs: Dict from G1Backend.observe_raw() with keys:
                "dof_pos" (29,), "dof_vel" (29,), "quat" (4,), "ang_vel" (3,)
            tracking_deltas: Dict from retargeter.get_tracking_deltas() with:
                "pos_delta" (18,), "rot_6d_delta" (36,)
            motion_obs: Optional additional motion reference (variable dim).

        Returns:
            Dict with "state" key -> tensor of shape (1, obs_dim).
        """
        dof_pos = raw_obs["dof_pos"]
        dof_vel = raw_obs["dof_vel"]
        quat = raw_obs["quat"]
        ang_vel = raw_obs["ang_vel"]

        # Compute body-frame angular velocity
        R = _quat_to_rotation_matrix(quat)
        ang_vel_local = R.T @ ang_vel

        # Compute projected gravity
        proj_grav = _projected_gravity(quat)

        # Compute base yaw from quaternion
        # yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        current_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).item()

        # Diff base yaw (scalar)
        diff_yaw = torch.tensor([self._ref_base_yaw - current_yaw], dtype=torch.float32)
        # Wrap to [-pi, pi]
        diff_yaw = torch.remainder(diff_yaw + np.pi, 2 * np.pi) - np.pi

        # Diff base position in local yaw frame
        # (simplified: assume flat ground, rotate by -current_yaw)
        c_yaw = np.cos(-current_yaw)
        s_yaw = np.sin(-current_yaw)
        dp = self._ref_base_pos - torch.tensor([0.0, 0.0, 0.0])  # base at origin
        diff_pos_local = torch.tensor(
            [
                c_yaw * dp[0].item() - s_yaw * dp[1].item(),
                s_yaw * dp[0].item() + c_yaw * dp[1].item(),
                dp[2].item(),
            ],
            dtype=torch.float32,
        )

        # Tracking link deltas
        if tracking_deltas is not None:
            # Rotate position deltas into local yaw frame
            pos_delta_np = tracking_deltas["pos_delta"]  # (18,)
            rot_6d_np = tracking_deltas["rot_6d_delta"]  # (36,)

            pos_delta_reshaped = pos_delta_np.reshape(NUM_TRACKED_LINKS, 3)
            pos_delta_local = np.zeros_like(pos_delta_reshaped)
            for i in range(NUM_TRACKED_LINKS):
                p = pos_delta_reshaped[i]
                pos_delta_local[i, 0] = c_yaw * p[0] - s_yaw * p[1]
                pos_delta_local[i, 1] = s_yaw * p[0] + c_yaw * p[1]
                pos_delta_local[i, 2] = p[2]

            link_pos_delta = torch.from_numpy(pos_delta_local.flatten().astype(np.float32))
            link_rot_6d = torch.from_numpy(rot_6d_np)
        else:
            link_pos_delta = torch.zeros(18, dtype=torch.float32)
            link_rot_6d = torch.zeros(36, dtype=torch.float32)

        # Assemble observation in ExtremControl order
        parts = [
            self._last_action,  # (29,)
            dof_pos,  # (29,)
            dof_vel * VEL_SCALE,  # (29,)
            ang_vel_local,  # (3,)
            diff_yaw,  # (1,)
            diff_pos_local,  # (3,)
            link_pos_delta,  # (18,)
            link_rot_6d,  # (36,)
            proj_grav,  # (3,)
        ]

        if motion_obs is not None:
            parts.append(motion_obs)

        obs_flat = torch.cat(parts, dim=0).unsqueeze(0)
        return {"state": obs_flat}

    def reset(self) -> None:
        """Clear internal buffers."""
        self._last_action = torch.zeros(NUM_MOTORS, dtype=torch.float32)
        self._ref_base_yaw = 0.0
        self._ref_base_pos = torch.zeros(3, dtype=torch.float32)
