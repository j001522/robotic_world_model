# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


@configclass
class RslRlSystemDynamicsCfg:
    """Configuration for the system dynamics networks."""
    
    ensemble_size: int = MISSING
    """The ensemble size of the system dynamics network."""

    history_horizon: int = MISSING
    """The prediction horizon of the system dynamics network."""

    architecture_config: dict = MISSING
    """The architecture configuration of the system dynamics network."""
    
    freeze_auxiliary: bool = MISSING
    """Whether to freeze the auxiliary networks."""

    uncertainty_metric: str = "std"
    """The metric for epistemic uncertainty: 'std' (default, original) or 'variance' (theoretically sound, additive)."""

    prior_scale: float = 0.0
    """Scale for randomized priors (Osband et al., 2018). 0 = disabled (default), 1.0 = standard prior strength."""

    prior_hidden_div: int = 4
    """Divisor for prior network hidden dimensions. Prior uses hidden_dim // prior_hidden_div."""

    bootstrap: bool = True
    """Whether to use bootstrapping (random data subsets per ensemble member). Default True for backward compatibility."""

    # --- Latent space parameters (Phase 1) ---
    latent_mode: bool = False
    """Whether to use latent-space dynamics. Default False preserves original raw-state behavior."""

    latent_dim: int = 256
    """Latent representation dimension. Must be divisible by simnorm_dim. Only used if latent_mode=True."""

    simnorm_dim: int = 8
    """SimNorm group size. Each group of simnorm_dim elements is normalized to a probability simplex."""

    encoder_hidden_dims: list[int] | None = None
    """Hidden layer widths for the state encoder. Defaults to [256] if None."""

    decoder_hidden_dims: list[int] | None = None
    """Hidden layer widths for the state decoder. Defaults to [256] if None."""

    latent_head_hidden_dims: list[int] | None = None
    """Hidden layer widths for the latent dynamics head. Defaults to [256] if None."""

    encoder_dropout: float = 0.0
    """Dropout rate for encoder hidden layers."""

    consistency_coef: float = 2.0
    """Weight for consistency loss (MSE in latent space between predicted and target latent states)."""

    reconstruction_coef: float = 1.0
    """Weight for reconstruction loss (MSE between decoded prediction and raw target state)."""

    target_encoder_momentum: float = 0.99
    """EMA momentum for the target encoder. θ_target ← momentum * θ_target + (1 - momentum) * θ_online.
    Only used when latent_mode=True."""

    encoder_consistency_coef: float = 0.0
    """Weight for bidirectional encoder consistency loss (TD-MPC2 style). 0.0 = disabled (default).
    When > 0, adds a loss term that trains the online encoder to produce representations 
    consistent with dynamics predictions."""

    residual_decoder: bool = False
    """Whether the decoder predicts residuals (s_{t+1} = s_t + Decoder(z, s_t)) instead of absolute states.
    Only used when latent_mode=True. Default False preserves original behavior."""


@configclass
class RslRlNormalizerCfg:
    """Configuration for the normalizer."""

    mean: list[float] = MISSING
    """The mean of the normalizer."""

    std: list[float] = MISSING
    """The std of the normalizer."""


@configclass
class RslRlMbrlImaginationCfg:
    """Configuration for the imagination."""
    
    num_envs: int = MISSING
    """The number of environments for the imagination."""
    
    num_steps: int = MISSING
    """The number of steps for the imagination."""
    
    max_episode_length: int = MISSING
    """The maximum episode length for the imagination."""

    command_resample_interval: int = MISSING
    """The resample interval for the command."""
    
    uncertainty_penalty_weight: float = MISSING
    """The weight for the uncertainty penalty."""
    
    state_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the state."""
    
    action_normalizer: RslRlNormalizerCfg = MISSING
    """The normalizer for the action."""


@configclass
class RslRlMbrlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "MBPOPPO"
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    policy_learning_rate: float = MISSING
    """The learning rate for the policy."""

    system_dynamics_learning_rate: float = MISSING
    """The learning rate for the system dynamics."""
    
    system_dynamics_weight_decay: float = MISSING
    """The weight decay for the system dynamics."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""
    
    system_dynamics_forecast_horizon: int = MISSING
    """The forecast horizon for the system dynamics."""
    
    system_dynamics_loss_weights: dict[str, float] = MISSING
    """The loss weights for the system dynamics."""
    
    system_dynamics_num_mini_batches: int = MISSING
    """The number of mini-batches for the system dynamics."""
    
    system_dynamics_mini_batch_size: int = MISSING
    """The mini-batch size for the system dynamics."""
    
    system_dynamics_replay_buffer_size: int = MISSING
    """The replay buffer size for the system dynamics."""
    
    system_dynamics_num_eval_trajectories: int = MISSING
    """The number of evaluation trajectories for the system dynamics."""
    
    system_dynamics_len_eval_trajectory: int = MISSING
    """The length of the evaluation trajectory for the system dynamics."""
    
    system_dynamics_eval_traj_noise_scale: list[float] = MISSING
    """The noise scale for the evaluation trajectory for the system dynamics."""

    latent_native_imagination: bool = False
    """Whether to keep imagination rollouts in latent space.
    
    When True and latent_mode is True, the imagination loop encodes the initial
    state history once and then carries the latent state forward through dynamics
    predictions without re-encoding. Decoding still happens every step for
    analytical reward computation and actor observations, but the dynamics
    transition z_t -> z_{t+1} stays in latent space, eliminating the
    encode(decode(z)) != z compounding drift."""
