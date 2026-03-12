# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configurations for the Franka Reach MBRL task.

Defines the following config variants:
- FrankaReachFlatEnvCfg: Base config (extends IsaacLab FrankaReachEnvCfg)
- FrankaReachFlatEnvCfg_INIT: For standard RL training (no WM)
- FrankaReachFlatEnvCfg_PRETRAIN: Adds system_state/action/termination obs groups for WM training
- FrankaReachFlatEnvCfg_FINETUNE: MBRL finetuning with imagination
- FrankaReachFlatEnvCfg_VISUALIZE: Visualization mode
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ObservationsCfg

import mbrl.tasks.manager_based.manipulation.reach.mdp as mdp

from mbrl.mbrl.envs.mdp.commands import SampleUniformPoseCommand, UniformPoseCommand_Visualize


##
# Base environment config
##


@configclass
class FrankaReachFlatEnvCfg(FrankaReachEnvCfg):
    """Base config for Franka Reach MBRL. Inherits from IsaacLab's FrankaReachEnvCfg."""

    def __post_init__(self):
        super().__post_init__()

        # -- Fix 1 (CRITICAL): Disable reward-weight curriculum --
        # The CurriculumCfg from ReachEnvCfg abruptly multiplies action_rate by 50x
        # and joint_vel by 10x at step 4500 via modify_reward_weight. This mutates
        # the shared reward_manager, corrupting both real and imagination rewards.
        # Setting to None disables the curriculum term.
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # -- Fix 2 (HIGH): Rebalance reward weights --
        # The original reward structure is net-negative: the dominant term is a penalty
        # (-0.2 * distance) with only a small positive bonus (+0.1 * tanh_kernel).
        # Rebalance so positive reward dominates (matching ANYmal D design philosophy).
        self.rewards.end_effector_position_tracking.weight = -0.1               # was -0.2
        self.rewards.end_effector_position_tracking_fine_grained.weight = 1.0   # was 0.1
        # Set penalty weights to their intended final values directly (no curriculum)
        self.rewards.action_rate.weight = -0.005
        self.rewards.joint_vel.weight = -0.001

        # -- Fix 3 (LOW): Restrict joint_vel to arm joints only --
        # Real env computes joint_vel over all 9 joints (7 arm + 2 finger), but
        # imagination only uses 7 arm joints. Restrict real env to match.
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["panda_joint.*"]
        )


##
# Init config (standard RL, no world model)
##


@configclass
class FrankaReachFlatEnvCfg_INIT(FrankaReachFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()


##
# Pretrain observation config (adds system_state, system_action, system_termination groups)
##


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):
    """Extended observations that include system_state, system_action, and system_termination groups.

    System state (24D):
        - joint_pos_rel:  7D (arm joints only, relative to default)
        - joint_vel_rel:  7D (arm joints only, relative to default)
        - joint_torque:   7D (arm joints only)
        - ee_pos:         3D (end-effector position in robot root frame)

    System action (7D):
        - last_action:    7D (arm joint position targets)

    System termination (0D):
        - No early termination for reach (only timeout, handled separately)
    """

    @configclass
    class SystemStateCfg(ObsGroup):
        """System state observation group (24D total)."""

        # Joint state for arm joints only (7 arm joints, excluding 2 finger joints)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )
        # End-effector position in robot root frame (3D)
        ee_pos = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        """System action observation group (7D total)."""

        pred_actions = ObsTerm(
            func=mdp.last_action,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemExtensionCfg(ObsGroup):
        """No extensions needed for reach task."""

        pass

    @configclass
    class SystemContactCfg(ObsGroup):
        """No contacts needed for reach task."""

        pass

    @configclass
    class SystemTerminationCfg(ObsGroup):
        """No early termination for reach task (only timeout)."""

        pass

    # Observation groups
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    # system_extension: SystemExtensionCfg = SystemExtensionCfg()
    # system_contact: SystemContactCfg = SystemContactCfg()
    # system_termination: SystemTerminationCfg = SystemTerminationCfg()


##
# Pretrain config
##


@configclass
class FrankaReachFlatEnvCfg_PRETRAIN(FrankaReachFlatEnvCfg):
    """Pretrain config: real rollouts + WM training, no imagination."""

    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()


##
# Finetune config
##


@configclass
class FrankaReachFlatEnvCfg_FINETUNE(FrankaReachFlatEnvCfg_PRETRAIN):
    """Finetune config: real + imagination rollouts."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable observation noise for imagination
        self.observations.policy.enable_corruption = False
        # override commands to use SampleUniformPoseCommand (has sample_command for imagination)
        self.commands.ee_pose.class_type = SampleUniformPoseCommand


##
# Visualize config
##


@configclass
class FrankaReachFlatEnvCfg_VISUALIZE(FrankaReachFlatEnvCfg_PRETRAIN):
    """Visualize config: side-by-side real vs imagination comparison."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable observation noise
        self.observations.policy.enable_corruption = False
        # override commands to use visualization variant
        self.commands.ee_pose.class_type = UniformPoseCommand_Visualize
