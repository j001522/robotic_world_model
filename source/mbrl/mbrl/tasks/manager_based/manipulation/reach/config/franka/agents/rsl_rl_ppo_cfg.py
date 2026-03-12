# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configurations for the Franka Reach MBRL task.

Defines four runner configs following the locomotion template:
- FrankaReachFlatPPORunnerCfg:          Base PPO config (extends IsaacLab's FrankaReachPPORunnerCfg)
- FrankaReachFlatPPOPretrainRunnerCfg:  Pretrain with WM (no imagination)
- FrankaReachFlatPPOFinetuneRunnerCfg:  Finetune with imagination rollouts
- FrankaReachFlatPPOVisualizeRunnerCfg: Visualization mode

System state normalizer (24D):
    [0:7]   joint_pos_rel:  mean=0.0, std=0.5
    [7:14]  joint_vel_rel:  mean=0.0, std=1.0
    [14:21] joint_torque:   mean=0.0, std=10.0
    [21:24] ee_pos:         mean=[0.5, 0.0, 0.3], std=[0.15, 0.2, 0.15]

Action normalizer (7D):
    All mean=0.0, std=1.0
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaReachPPORunnerCfg,
)
from mbrl.rl.rsl_rl import (
    RslRlSystemDynamicsCfg,
    RslRlNormalizerCfg,
    RslRlMbrlImaginationCfg,
    RslRlMbrlPpoAlgorithmCfg,
)


@configclass
class FrankaReachFlatPPORunnerCfg(FrankaReachPPORunnerCfg):
    """Base PPO runner for Franka Reach (standard RL, no world model)."""

    pass


@configclass
class FrankaReachFlatPPOPretrainRunnerCfg(FrankaReachPPORunnerCfg):
    """Pretrain runner: real rollouts + world model training, no imagination."""

    class_name: str = "MBPOOnPolicyRunner"

    system_dynamics = RslRlSystemDynamicsCfg(
        ensemble_size=1,
        history_horizon=32,
        architecture_config={
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
            # xLSTM defaults (used when type=xlstm)
            "xlstm_embedding_dim": 256,
            "xlstm_num_blocks": 4,
            "xlstm_num_heads": 4,
            "xlstm_context_length": 64,
            "xlstm_conv1d_kernel_size": 4,
            "xlstm_slstm_at": [],
            "xlstm_slstm_backend": "vanilla",
            "xlstm_proj_factor_mlstm": 2.0,
            "xlstm_proj_factor_slstm_ff": 1.3,
            "xlstm_dropout": 0.0,
        },
        freeze_auxiliary=False,
        uncertainty_metric="std",
        prior_scale=0.0,
        prior_hidden_div=4,
        bootstrap=True,
    )

    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,
        num_steps=0,
        max_episode_length=0,
        command_resample_interval=-1,
        uncertainty_penalty_weight=-0.0,
        state_normalizer=RslRlNormalizerCfg(
            mean=[
                # joint_pos_rel (7D)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # joint_vel_rel (7D)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # joint_torque (7D)
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                # ee_pos (3D)
                0.5, 0.0, 0.3,
            ],
            std=[
                # joint_pos_rel (7D)
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # joint_vel_rel (7D)
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                # joint_torque (7D)
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                # ee_pos (3D)
                0.15, 0.2, 0.15,
            ],
        ),
        action_normalizer=RslRlNormalizerCfg(
            mean=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )

    algorithm = RslRlMbrlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        policy_learning_rate=1.0e-3,
        system_dynamics_learning_rate=1.0e-3,
        system_dynamics_weight_decay=0.0,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        system_dynamics_forecast_horizon=8,
        system_dynamics_loss_weights={
            "state": 1.0,
            "sequence": 1.0,
            "bound": 1.0,
            "kl": 0.1,
            "extension": 1.0,
            "contact": 0.0,        # no contacts for reach
            "termination": 0.0,    # no early termination for reach
            "reward": 0.0,
            "value": 0.0,
        },
        system_dynamics_num_mini_batches=20,
        system_dynamics_mini_batch_size=5000,
        system_dynamics_replay_buffer_size=1000,
        system_dynamics_num_eval_trajectories=100,
        system_dynamics_len_eval_trajectory=400,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
    )

    run_name = "pretrain"
    load_system_dynamics = False
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    system_dynamics_num_visualizations = 4
    system_dynamics_state_idx_dict = {
        r"$q$" + "\n" + r"$[rad]$": [0, 1, 2, 3, 4, 5, 6],
        r"$\dot{q}$" + "\n" + r"$[rad/s]$": [7, 8, 9, 10, 11, 12, 13],
        r"$\tau$" + "\n" + r"$[Nm]$": [14, 15, 16, 17, 18, 19, 20],
        r"$ee$" + "\n" + r"$[m]$": [21, 22, 23],
    }
    pca_obs_buf_size = 10000

    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000


@configclass
class FrankaReachFlatPPOFinetuneRunnerCfg(FrankaReachFlatPPOPretrainRunnerCfg):
    """Finetune runner: real + imagination rollouts."""

    resume = True
    load_run = None  # set to specific pretrain run directory
    load_system_dynamics = True
    system_dynamics_load_path = None  # set to specific pretrain model path
    system_dynamics_warmup_iterations = 500
    run_name = "finetune"

    def __post_init__(self):
        super().__post_init__()
        # Enable imagination
        self.imagination.num_envs = 8192
        self.imagination.num_steps = 24
        self.imagination.max_episode_length = 256
        self.imagination.command_resample_interval = 110
        self.imagination.uncertainty_penalty_weight = -0.0


@configclass
class FrankaReachFlatPPOVisualizeRunnerCfg(FrankaReachFlatPPOPretrainRunnerCfg):
    """Visualization runner: side-by-side real vs imagined rollouts."""

    resume = True
    load_system_dynamics = True
    system_dynamics_load_path = None  # set to specific pretrain model path
    run_name = "visualize"
