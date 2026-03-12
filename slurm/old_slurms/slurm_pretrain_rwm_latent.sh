#!/bin/bash
#SBATCH --job-name=rwm-latent-pretrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-latent-pretrain-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-latent-pretrain-%j.err

# =============================================================================
# Latent-Space World Model Pretraining (Phase 1+2 Hybrid)
# =============================================================================
# Trains an ensemble world model with latent-space bottleneck.
# Architecture: Encoder (SimNorm) -> GRU backbone -> LatentDynamicsHead (SimNorm)
#               + Decoder for reconstruction loss
#
# Training signal (combined as state_loss):
#   state_loss = consistency_coef * MSE(z_pred, sg(enc(s_next)))   [JEPA-style]
#              + reconstruction_coef * MSE(dec(z_pred), s_next)    [reconstruction]
#
# The encoder targets are stop-gradiented (no EMA target encoder yet).
#
# Usage:
#   sbatch slurm_pretrain_rwm_latent.sh
# =============================================================================

# ======================= CONFIGURATION =======================
# Training parameters
MAX_ITERATIONS=2500
NUM_ENVS=4096
ENSEMBLE_SIZE=5

# Latent-space architecture
LATENT_MODE=True
LATENT_DIM=256
SIMNORM_DIM=8
# Hidden dims for encoder, decoder, and latent dynamics head
# These are passed as Hydra list overrides
ENCODER_HIDDEN='[256]'
DECODER_HIDDEN='[256]'
LATENT_HEAD_HIDDEN='[256]'
ENCODER_DROPOUT=0.0

# Loss coefficients (combined into state_loss inside compute_state_loss)
CONSISTENCY_COEF=1.0
RECONSTRUCTION_COEF=1.0

# Randomized priors (set PRIOR_SCALE>0 to enable)
PRIOR_SCALE=1.0
PRIOR_HIDDEN_DIV=4
BOOTSTRAP=False

# World model training hyperparameters
WM_LEARNING_RATE=1e-4
WM_WEIGHT_DECAY=1e-5
WM_BATCH_SIZE=1024

# Uncertainty
UNCERTAINTY_PENALTY=0.0
UNCERTAINTY_METRIC="std"
# =============================================================

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

# ======================= LOGGING =======================
LOGGER="tensorboard"
WANDB_PROJECT="rwm-anymal"
RUN_NAME="pretrain-latent-dim${LATENT_DIM}-c${CONSISTENCY_COEF}-r${RECONSTRUCTION_COEF}"
# =======================================================

# Print environment info
echo ""
echo "Environment Information:"
echo "  Working directory: $(pwd)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Container: /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif"
echo ""

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

echo "========================================"
echo "Latent-Space World Model Pretraining"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0"
echo ""
echo "Latent Architecture:"
echo "  - Latent Mode: $LATENT_MODE"
echo "  - Latent Dim: $LATENT_DIM"
echo "  - SimNorm Dim: $SIMNORM_DIM (${LATENT_DIM}/${SIMNORM_DIM} = $((LATENT_DIM / SIMNORM_DIM)) groups)"
echo "  - Encoder Hidden: $ENCODER_HIDDEN"
echo "  - Decoder Hidden: $DECODER_HIDDEN"
echo "  - Dynamics Head Hidden: $LATENT_HEAD_HIDDEN"
echo "  - Encoder Dropout: $ENCODER_DROPOUT"
echo ""
echo "Loss Configuration:"
echo "  - Consistency Coef: $CONSISTENCY_COEF (JEPA-style latent MSE)"
echo "  - Reconstruction Coef: $RECONSTRUCTION_COEF (decoded state MSE)"
echo "  - state_loss = ${CONSISTENCY_COEF}*L_consist + ${RECONSTRUCTION_COEF}*L_recon"
echo ""
echo "Ensemble & Priors:"
echo "  - Ensemble Size: $ENSEMBLE_SIZE"
echo "  - Prior Scale: $PRIOR_SCALE"
echo "  - Prior Hidden Div: $PRIOR_HIDDEN_DIV"
echo "  - Bootstrap: $BOOTSTRAP"
echo ""
echo "Training Hyperparameters:"
echo "  - Real Envs: $NUM_ENVS"
echo "  - Max Iterations: $MAX_ITERATIONS"
echo "  - WM Learning Rate: $WM_LEARNING_RATE"
echo "  - WM Weight Decay: $WM_WEIGHT_DECAY"
echo "  - WM Batch Size: $WM_BATCH_SIZE"
echo "  - Imagination: DISABLED (pretrain phase)"
echo ""
echo "Logging:"
echo "  - Logger: $LOGGER"
echo "  - Run Name: $RUN_NAME"
echo ""
echo "Expected training time: ~12-18 hours (encoder/decoder add overhead)"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run training inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --logger $LOGGER \
    --log_project_name $WANDB_PROJECT \
    --run_name $RUN_NAME \
    agent.system_dynamics.ensemble_size=$ENSEMBLE_SIZE \
    agent.system_dynamics.uncertainty_metric=$UNCERTAINTY_METRIC \
    agent.system_dynamics.prior_scale=$PRIOR_SCALE \
    agent.system_dynamics.prior_hidden_div=$PRIOR_HIDDEN_DIV \
    agent.system_dynamics.bootstrap=$BOOTSTRAP \
    agent.system_dynamics.latent_mode=$LATENT_MODE \
    agent.system_dynamics.latent_dim=$LATENT_DIM \
    agent.system_dynamics.simnorm_dim=$SIMNORM_DIM \
    "agent.system_dynamics.encoder_hidden_dims=$ENCODER_HIDDEN" \
    "agent.system_dynamics.decoder_hidden_dims=$DECODER_HIDDEN" \
    "agent.system_dynamics.latent_head_hidden_dims=$LATENT_HEAD_HIDDEN" \
    agent.system_dynamics.encoder_dropout=$ENCODER_DROPOUT \
    agent.system_dynamics.consistency_coef=$CONSISTENCY_COEF \
    agent.system_dynamics.reconstruction_coef=$RECONSTRUCTION_COEF \
    agent.algorithm.system_dynamics_learning_rate=$WM_LEARNING_RATE \
    agent.algorithm.system_dynamics_weight_decay=$WM_WEIGHT_DECAY \
    agent.algorithm.system_dynamics_mini_batch_size=$WM_BATCH_SIZE \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo ""
echo "========================================"
echo "Latent Pretrain Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Latent-space world model pretraining completed!"
    echo ""
    echo "Checkpoint location:"
    LATEST_LOG=$(ls -td logs/rsl_rl/anymal_d_flat/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  $LATEST_LOG"
        echo ""
        echo "Files saved:"
        ls -lh "$LATEST_LOG"model_*.pt 2>/dev/null | tail -3
    fi
    echo ""
    echo "The checkpoint contains:"
    echo "  - StateEncoder (${LATENT_DIM}-dim latent with SimNorm)"
    echo "  - StateDecoder (latent -> raw state)"
    echo "  - ${ENSEMBLE_SIZE}-member LatentDynamicsHead ensemble"
    echo "  - Consistency + reconstruction losses"
    echo ""
    echo "Next steps:"
    echo "  1. Finetune with imagination:"
    echo "     sbatch slurm_finetune_rwm_latent.sh"
    echo "     (Update PRETRAIN_DIR to point to this run)"
    echo ""
    echo "  2. Check tensorboard for latent-specific metrics:"
    echo "     - System Dynamics/consistency_loss  (should decrease)"
    echo "     - System Dynamics/reconstruction_loss  (should be <= 1.1x baseline)"
    echo ""
    echo "  3. Validate Phase 1 success criteria:"
    echo "     - Reconstruction error <= 1.1x baseline state prediction error"
    echo "     - No NaN/Inf in latent rollouts"
    echo "     - SimNorm groups sum to 1.0"
else
    echo ""
    echo "FAILED: Latent pretrain failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/rwm-latent-pretrain-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - latent_dim not divisible by simnorm_dim"
    echo "  - OOM: Encoder/decoder add parameters; reduce NUM_ENVS or WM_BATCH_SIZE"
    echo "  - Hydra list override syntax: use quotes around '[256]' style args"
    echo "  - Missing latent_mode fields: ensure rsl_rl_rwm is on feature/latent-space branch"
fi

exit $EXITCODE
