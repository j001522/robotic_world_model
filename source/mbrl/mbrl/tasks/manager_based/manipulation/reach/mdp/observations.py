# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation terms for the Franka Reach MBRL environment.

These observation terms are used to compute the end-effector position in the robot's root frame,
which is included in the system_state so that the world model learns to predict it directly.
This avoids needing to reimplement forward kinematics in PyTorch for imagination rewards.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ee_position_in_robot_root_frame(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """End-effector position expressed in the robot's root frame.

    This computes the 3D position of the end-effector body (e.g. panda_hand) relative to the
    robot's root link frame. The result is a (num_envs, 3) tensor of [x, y, z] coordinates.

    Including this in the system_state allows the world model to predict ee position directly,
    which is needed for computing reach rewards during imagination without reimplementing FK.

    Args:
        env: The environment.
        asset_cfg: SceneEntityCfg specifying the robot asset and the body to track.

    Returns:
        End-effector position in robot root frame. Shape: (num_envs, 3).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Get world-frame quantities
    root_pos_w = asset.data.root_pos_w       # (num_envs, 3)
    root_quat_w = asset.data.root_quat_w     # (num_envs, 4) [w, x, y, z]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # (num_envs, 3)
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # (num_envs, 4)

    # Transform ee position from world frame to robot root frame
    ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    return ee_pos_b  # (num_envs, 3)
