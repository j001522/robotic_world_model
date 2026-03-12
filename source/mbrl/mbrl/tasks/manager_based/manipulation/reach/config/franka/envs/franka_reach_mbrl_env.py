# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Reach MBRL environment.

Implements the 7 abstract methods from ManagerBasedMBRLEnv for the Franka Panda reach task.

System state layout (24D):
    [0:7]   joint_pos_rel  (7 arm joints, relative to default)
    [7:14]  joint_vel_rel  (7 arm joints)
    [14:21] joint_torque   (7 arm joints)
    [21:24] ee_pos         (end-effector position in robot root frame)

System action layout (7D):
    [0:7]   arm joint position targets

No contacts (0D) and no early termination (0D) for reach.

Imagination reward terms (must match RewardsCfg names exactly):
    - end_effector_position_tracking:            L2 distance to goal
    - end_effector_position_tracking_fine_grained: tanh(distance / 0.1)
    - end_effector_orientation_tracking:         set to 0 (orientation not in system_state)
    - action_rate:                               L2 of action difference
    - joint_vel:                                 L2 of joint velocities
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.mbrl.envs import ManagerBasedMBRLEnv


class FrankaReachManagerBasedMBRLEnv(ManagerBasedMBRLEnv):
    """MBRL environment for Franka Panda reach task."""

    # -- System state dimensions --
    # joint_pos_rel: 7, joint_vel_rel: 7, joint_torque: 7, ee_pos: 3
    STATE_DIM = 24
    ACTION_DIM = 7
    # Policy obs: joint_pos(9) + joint_vel(9) + pose_command(7) + last_action(7) = 32
    POLICY_OBS_DIM = 32

    def _init_additional_attributes(self):
        """Store default joint positions/velocities for denormalization."""
        # Default positions for the 7 arm joints (excluding finger joints)
        # The asset has 9 joints total (7 arm + 2 finger), but we only use arm joints
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0, :7]
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0, :7]

    def _init_additional_imagination_attributes(self):
        """No additional imagination attributes needed for reach.

        Unlike locomotion (which tracks air_time, contact_time), reach has no
        stateful reward terms that need per-step tracking.
        """
        pass

    def _reset_additional_imagination_attributes(self, env_ids):
        """No additional attributes to reset."""
        pass

    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        """Construct policy observation from predicted system state.

        Policy observation (32D):
            joint_pos_rel(9) + joint_vel_rel(9) + pose_command(7) + last_action(7)

        Note: The system state only has 7 arm joints, but the policy observation includes
        all 9 joints (7 arm + 2 finger). We pad the finger joints with zeros (they don't
        move in reach tasks).

        Args:
            state_history: Normalized state history. Shape: (N, H, 24)
            action_history: Normalized action history. Shape: (N, H, 7)
            observation_noise: Whether to add observation noise.

        Returns:
            TensorDict with "policy" key containing the observation.
        """
        # Denormalize the latest state
        current_state = self.imagination_state_normalizer.inverse(state_history[:, -1])

        # Extract components from system state
        obs_joint_pos_arm = current_state[:, 0:7]    # 7D arm joint pos (relative)
        obs_joint_vel_arm = current_state[:, 7:14]    # 7D arm joint vel (relative)
        # joint_torque [14:21] and ee_pos [21:24] not directly in policy obs

        # Denormalize last action
        self.obs_last_action = self.imagination_action_normalizer.inverse(action_history[:, -1])  # 7D

        # Add observation noise (matching IsaacLab reach config: +-0.01 for both)
        if observation_noise:
            obs_joint_pos_arm = obs_joint_pos_arm + 2 * (torch.rand_like(obs_joint_pos_arm) - 0.5) * 0.01
            obs_joint_vel_arm = obs_joint_vel_arm + 2 * (torch.rand_like(obs_joint_vel_arm) - 0.5) * 0.01

        # Pad with 2 finger joints (zeros, since fingers are fixed in reach)
        finger_pos = torch.zeros(self.num_imagination_envs, 2, device=self.device)
        finger_vel = torch.zeros(self.num_imagination_envs, 2, device=self.device)

        obs_joint_pos = torch.cat([obs_joint_pos_arm, finger_pos], dim=1)  # 9D
        obs_joint_vel = torch.cat([obs_joint_vel_arm, finger_vel], dim=1)  # 9D

        # Command: ee_pose is 7D [x, y, z, qw, qx, qy, qz] in robot base frame
        # Accessed via self.ee_pose (set by _init_imagination_command)
        obs_command = self.ee_pose  # 7D

        # Construct full policy observation (32D)
        obs = torch.cat([
            obs_joint_pos,        # 9D
            obs_joint_vel,        # 9D
            obs_command,          # 7D
            self.obs_last_action, # 7D
        ], dim=1)

        obs = TensorDict({"policy": obs}, batch_size=[self.num_imagination_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        """Parse the flat denormalized state vector into named components.

        System state layout (24D):
            [0:7]   joint_pos_rel
            [7:14]  joint_vel_rel
            [14:21] joint_torque
            [21:24] ee_pos (in robot root frame)

        Args:
            imagination_states_denormalized: Shape (N, 24)

        Returns:
            Dict mapping component names to tensors.
        """
        joint_pos = imagination_states_denormalized[:, 0:7]
        joint_vel = imagination_states_denormalized[:, 7:14]
        joint_torque = imagination_states_denormalized[:, 14:21]
        ee_pos = imagination_states_denormalized[:, 21:24]

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
            "ee_pos": ee_pos,
        }

    def _parse_extensions(self, extensions):
        """No extensions for reach task."""
        if extensions is None:
            return None
        return {}

    def _parse_contacts(self, contacts):
        """No contacts for reach task."""
        if contacts is None:
            return None
        return {}

    def _parse_terminations(self, terminations):
        """No early termination for reach (only timeout, handled by base class).

        We return None so the base class uses only timeout-based termination.
        """
        return None

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        """Compute reward terms during imagination rollouts.

        These MUST match the names in RewardsCfg exactly so that
        _post_imagination_step() can look up the weights via reward_manager.get_term_cfg().

        Reward terms:
            1. end_effector_position_tracking: -weight * L2(ee_pos, goal_pos)
            2. end_effector_position_tracking_fine_grained: +weight * (1 - tanh(dist / 0.1))
            3. end_effector_orientation_tracking: set to 0 (not tracked)
            4. action_rate: L2(last_action - current_action)
            5. joint_vel: sum of squared joint velocities

        Args:
            parsed_imagination_states: Dict from _parse_imagination_states
            rollout_action: Current action. Shape (N, 7)
            parsed_extensions: None/empty for reach
            parsed_contacts: None/empty for reach
        """
        ee_pos = parsed_imagination_states["ee_pos"]        # (N, 3)
        joint_vel = parsed_imagination_states["joint_vel"]  # (N, 7)

        # Goal position from command (first 3 elements of ee_pose command)
        # ee_pose is [x, y, z, qw, qx, qy, qz] in robot base frame
        goal_pos = self.ee_pose[:, :3]  # (N, 3)

        # 1. Position tracking: L2 distance
        distance = torch.norm(ee_pos - goal_pos, dim=1)  # (N,)
        end_effector_position_tracking = distance

        # 2. Fine-grained position tracking: tanh kernel with std=0.1
        end_effector_position_tracking_fine_grained = 1.0 - torch.tanh(distance / 0.1)

        # 3. Orientation tracking: set to 0 (orientation not in system state)
        # We simplified this because tracking orientation would require quaternions
        # in the system state, adding complexity without much benefit for basic reach.
        end_effector_orientation_tracking = torch.zeros(self.num_imagination_envs, device=self.device)

        # 4. Action rate: L2 of action difference
        action_rate = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)

        # 5. Joint velocity penalty: L2 of joint velocities
        joint_vel_l2 = torch.sum(torch.square(joint_vel), dim=1)

        # Store in the dict that _post_imagination_step reads
        self.imagination_reward_per_step = {
            "end_effector_position_tracking": end_effector_position_tracking,
            "end_effector_position_tracking_fine_grained": end_effector_position_tracking_fine_grained,
            "end_effector_orientation_tracking": end_effector_orientation_tracking,
            "action_rate": action_rate,
            "joint_vel": joint_vel_l2,
        }

        # Update last_obs for next step (policy obs format)
        # Pad arm joints with finger zeros for policy obs
        joint_pos = parsed_imagination_states["joint_pos"]
        finger_pos = torch.zeros(self.num_imagination_envs, 2, device=self.device)
        finger_vel = torch.zeros(self.num_imagination_envs, 2, device=self.device)
        obs_joint_pos = torch.cat([joint_pos, finger_pos], dim=1)  # 9D
        obs_joint_vel = torch.cat([joint_vel, finger_vel], dim=1)  # 9D

        last_obs = torch.cat([
            obs_joint_pos,       # 9D
            obs_joint_vel,       # 9D
            self.ee_pose,        # 7D (command)
            rollout_action,      # 7D
        ], dim=1)
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_imagination_envs], device=self.device)
