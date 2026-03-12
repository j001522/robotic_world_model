# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Reach-Franka-Init-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FrankaReachFlatEnvCfg_INIT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Reach-Franka-Pretrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FrankaReachFlatEnvCfg_PRETRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachFlatPPOPretrainRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Reach-Franka-Finetune-v0",
    entry_point="mbrl.tasks.manager_based.manipulation.reach.config.franka.envs.franka_reach_mbrl_env:FrankaReachManagerBasedMBRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FrankaReachFlatEnvCfg_FINETUNE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachFlatPPOFinetuneRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Reach-Franka-Visualize-v0",
    entry_point="mbrl.tasks.manager_based.manipulation.reach.config.franka.envs.franka_reach_visualize_env:FrankaReachManagerBasedVisualizeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FrankaReachFlatEnvCfg_VISUALIZE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachFlatPPOVisualizeRunnerCfg",
    },
)
