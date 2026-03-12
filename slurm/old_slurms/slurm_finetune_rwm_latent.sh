#!/bin/bash
#SBATCH --job-name=rwm-latent-finetune
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-latent-finetune-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-latent-finetune-%j.err

# =============================================================================
# Latent-Space World Model Finetuning (MBPO-PPO with Imagination)
# =============================================================================
# Finetunes a policy using imagination rollouts from a pretrained latent-space
# ensemble world model. The ensemble provides epistemic uncertainty in latent
# space (ensemble disagreement of predicted latent states).
#
# Prerequisites:
#   - Completed latent pretrain run (slurm_pretrain_rwm_latent.sh)
#   - Update PRETRAIN_DIR below to point to your pretrain logs
#
# IMPORTANT: Latent architecture parameters MUST match the pretrain checkpoint.
# Mismatched latent_dim, simnorm_dim, hidden_dims, or prior_scale will cause
# state_dict loading errors.
#
# Usage:
#   sbatch slurm_finetune_rwm_latent.sh
# =============================================================================

# ======================= CONFIGURATION =======================
# UPDATE THIS: Path to your latent pretrain run directory
PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/CHANGE_ME_pretrain-latent"
CHECKPOINT_ITER=2500

# Finetuning parameters
MAX_ITERATIONS=2500
NUM_ENVS=4096
IMAGINATION_ENVS=4096
IMAGINATION_STEPS=100

# Ensemble parameters (must match pretrain!)
ENSEMBLE_SIZE=5

# Latent architecture (must match pretrain!)
LATENT_MODE=True
LATENT_DIM=256
SIMNORM_DIM=8
ENCODER_HIDDEN='[256]'
DECODER_HIDDEN='[256]'
LATENT_HEAD_HIDDEN='[256]'
ENCODER_DROPOUT=0.0

# Loss coefficients (must match pretrain or adjust for finetuning)
CONSISTENCY_COEF=2.0
RECONSTRUCTION_COEF=1.0

# Randomized priors (must match pretrain!)
PRIOR_SCALE=1.0
PRIOR_HIDDEN_DIV=4
BOOTSTRAP=False

# Uncertainty penalty weight (negative = penalty for high uncertainty)
# -0.1 = mild conservative
# -1.0 = strong conservative (as used in offline U-RWM)
# 0.0 = no penalty
UNCERTAINTY_PENALTY=-1.0
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
RUN_NAME="finetune-latent-dim${LATENT_DIM}-${UNCERTAINTY_METRIC}${UNCERTAINTY_PENALTY}"
# =======================================================

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Construct checkpoint path
CHECKPOINT_PATH="${PRETRAIN_DIR}/model_${CHECKPOINT_ITER}.pt"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Please update PRETRAIN_DIR in this script to point to your latent pretrain run."
    echo "Available runs:"
    ls -la logs/rsl_rl/anymal_d_flat/ 2>/dev/null || echo "  No runs found in logs/rsl_rl/anymal_d_flat/"
    echo ""
    exit 1
fi

# Print configuration
echo ""
echo "Environment Information:"
echo "  Working directory: $(pwd)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Container: /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif"
echo ""

echo "========================================"
echo "Latent-Space World Model Finetuning"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0"
echo ""
echo "Loading from latent pretrain:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""
echo "Latent Architecture (must match pretrain):"
echo "  - Latent Dim: $LATENT_DIM"
echo "  - SimNorm Dim: $SIMNORM_DIM"
echo "  - Encoder Hidden: $ENCODER_HIDDEN"
echo "  - Decoder Hidden: $DECODER_HIDDEN"
echo "  - Dynamics Head Hidden: $LATENT_HEAD_HIDDEN"
echo ""
echo "Ensemble & Priors (must match pretrain):"
echo "  - Ensemble Size: $ENSEMBLE_SIZE"
echo "  - Prior Scale: $PRIOR_SCALE"
echo "  - Prior Hidden Div: $PRIOR_HIDDEN_DIV"
echo "  - Bootstrap: $BOOTSTRAP"
echo ""
echo "Imagination Configuration:"
echo "  - Imagination Envs: $IMAGINATION_ENVS"
echo "  - Imagination Steps: $IMAGINATION_STEPS"
echo "  - Real Envs: $NUM_ENVS"
echo "  - Max Iterations: $MAX_ITERATIONS"
echo "  - Uncertainty Penalty: $UNCERTAINTY_PENALTY"
echo "  - Uncertainty Metric: $UNCERTAINTY_METRIC"
echo ""
echo "Loss Configuration:"
echo "  - Consistency Coef: $CONSISTENCY_COEF"
echo "  - Reconstruction Coef: $RECONSTRUCTION_COEF"
echo ""
echo "Logging:"
echo "  - Logger: $LOGGER"
echo "  - Run Name: $RUN_NAME"
echo ""
echo "Expected training time: ~14-20 hours"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run finetuning inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --headless \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --resume True \
    --system_dynamics_load_path $CHECKPOINT_PATH \
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
    agent.imagination.num_envs=$IMAGINATION_ENVS \
    agent.imagination.num_steps=$IMAGINATION_STEPS \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo ""
echo "========================================"
echo "Latent Finetune Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Latent-space MBPPO finetuning completed!"
    echo ""
    echo "The finetuned policy was trained using:"
    echo "  - Real environment rollouts (for world model updates)"
    echo "  - Imagined rollouts ($IMAGINATION_ENVS envs x $IMAGINATION_STEPS steps)"
    echo "  - ${ENSEMBLE_SIZE}-member latent ensemble with trajectory sampling"
    echo "  - Latent-space dynamics (${LATENT_DIM}-dim with SimNorm)"
    echo "  - Uncertainty penalty weight: $UNCERTAINTY_PENALTY"
    echo ""
    echo "Checkpoint location:"
    LATEST_LOG=$(ls -td logs/rsl_rl/anymal_d_flat/*finetune*latent*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_LOG" ]; then
        LATEST_LOG=$(ls -td logs/rsl_rl/anymal_d_flat/*/ 2>/dev/null | head -1)
    fi
    if [ -n "$LATEST_LOG" ]; then
        echo "  $LATEST_LOG"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate the finetuned policy:"
    echo "     python scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "       --task Isaac-Velocity-Flat-Anymal-D-Play-v0 \\"
    echo "       --checkpoint <finetuned_model>.pt \\"
    echo "       --video --video_length 400 --headless"
    echo ""
    echo "  2. Compare with raw-state baseline finetuning"
    echo ""
    echo "  3. Check tensorboard for latent-space metrics:"
    echo "     - System Dynamics/consistency_loss"
    echo "     - System Dynamics/reconstruction_loss"
else
    echo ""
    echo "FAILED: Latent finetuning failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/rwm-latent-finetune-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - Architecture mismatch: latent_dim, simnorm_dim, hidden_dims, or prior_scale"
    echo "    don't match the pretrain checkpoint"
    echo "  - Checkpoint path incorrect (update PRETRAIN_DIR)"
    echo "  - OOM: reduce NUM_ENVS or IMAGINATION_ENVS"
fi

exit $EXITCODE
