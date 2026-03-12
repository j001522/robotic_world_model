# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach visualization environment.

Uses multiple inheritance to combine ManagerBasedVisualizeEnv (visualization logic)
with FrankaReachManagerBasedMBRLEnv (task-specific abstract method implementations).

The key method _reset_imagination_sim() takes the parsed WM prediction and writes
the predicted joint states back into the simulator for the "imagination" robot instances,
allowing side-by-side comparison of real vs imagined rollouts.
"""

from __future__ import annotations

import torch

from mbrl.mbrl.envs import ManagerBasedVisualizeEnv
from .franka_reach_mbrl_env import FrankaReachManagerBasedMBRLEnv
from mbrl.mbrl.envs.mdp.events import reset_joints_to_specified


class FrankaReachManagerBasedVisualizeEnv(ManagerBasedVisualizeEnv, FrankaReachManagerBasedMBRLEnv):
    """Visualization env for Franka Reach: side-by-side real vs imagined rollouts."""

    def _reset_imagination_sim(self, parsed_imagination_states):
        """Write predicted joint states into the imagination robot instances.

        For Franka reach, we only need to update joint positions and velocities.
        The robot base is fixed (not a floating base like ANYmal), so no root state update.

        The predicted joint_pos and joint_vel are *relative* to defaults (matching the
        system_state definition), so we add back the defaults before writing to sim.

        Args:
            parsed_imagination_states: Dict with "joint_pos", "joint_vel", "joint_torque", "ee_pos"
        """
        joint_pos = parsed_imagination_states["joint_pos"]  # (N, 7) relative
        joint_vel = parsed_imagination_states["joint_vel"]  # (N, 7) relative

        # Convert from relative to absolute
        joint_pos = joint_pos + self.default_joint_pos
        joint_vel = joint_vel + self.default_joint_vel

        # Pad with finger joints (keep at default positions, zero velocity)
        default_finger_pos = self.scene["robot"].data.default_joint_pos[0, 7:9]
        default_finger_vel = self.scene["robot"].data.default_joint_vel[0, 7:9]

        finger_pos = default_finger_pos.unsqueeze(0).expand(joint_pos.shape[0], -1)
        finger_vel = default_finger_vel.unsqueeze(0).expand(joint_vel.shape[0], -1)

        joint_pos_full = torch.cat([joint_pos, finger_pos], dim=1)  # (N, 9)
        joint_vel_full = torch.cat([joint_vel, finger_vel], dim=1)  # (N, 9)

        # Write to the imagination robot instances
        reset_joints_to_specified(self, self.env_ids_imagination, joint_pos_full, joint_vel_full)
