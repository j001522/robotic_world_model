#!/bin/bash
#SBATCH --job-name=rwm-ensemble-finetune
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-ensemble-finetune-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-ensemble-finetune-%j.err

# =============================================================================
# U-RWM Phase 2: Finetune with Ensemble MBPO-PPO (Experimental)
# =============================================================================
# This script finetunes a policy using imagination rollouts from a pretrained
# ENSEMBLE world model. The ensemble provides epistemic uncertainty which is
# used to penalize the policy for visiting high-uncertainty regions.
#
# Key differences from single-model finetune:
#   - Uses 5-model ensemble (must match pretrain)
#   - Each imagination env follows a single random ensemble member
#   - Uncertainty penalty applied to rewards (conservative exploration)
#
# Prerequisites:
#   - Completed ENSEMBLE pretrain run (slurm_pretrain_rwm_ensemble.sh)
#   - Update PRETRAIN_DIR below to point to your ensemble pretrain logs
#
# Usage:
#   sbatch slurm_finetune_rwm_ensemble.sh
# =============================================================================

# ======================= CONFIGURATION =======================
# UPDATE THIS: Path to your ENSEMBLE pretrain run directory
# Example: logs/rsl_rl/anymal_d_flat/2026-01-25_12-00-00
#PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-25_13-09-26_pretrain-ensemble-prior1.0" # PRIORS + bootstrap
PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-25_12-55-50_pretrain-ensemble-prior1.0-noboot" # PRIORS (no bootstrap)
DYNAMICS_MODE="PRIORS_noboot" # For logging purposes, change according to pretrain type
BOOTSTRAP=False 
# Checkpoint iteration to load (default: final checkpoint from 2500 iterations)
CHECKPOINT_ITER=2500

# Finetuning parameters (from Table S11)
MAX_ITERATIONS=2500
NUM_ENVS=4096
IMAGINATION_ENVS=4096
IMAGINATION_STEPS=100

# Ensemble parameters
ENSEMBLE_SIZE=5

# Randomized prior parameters (must match pretrain!)
# If pretrain used priors, finetune MUST use same prior_scale and prior_hidden_div
PRIOR_SCALE=1.0       # 0.0 = no priors (standard), 1.0 = standard prior strength
PRIOR_HIDDEN_DIV=4    # Divisor for prior hidden dimensions

# Uncertainty penalty weight (negative = penalty for high uncertainty)
# -0.1 = mild penalty (conservative)
# -1.0 = strong penalty (very conservative, as used in offline U-RWM)
# 0.0 = no penalty (same as single model, but with trajectory sampling)
UNCERTAINTY_PENALTY=-1.0

# Uncertainty metric: "std" (original) or "variance" (theoretically sound)
# - std: same units as state, milder penalty, original RWM implementation
# - variance: additive across dimensions, stronger penalty for high disagreement
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
# Weights & Biases logging (set to "tensorboard" to disable wandb)
LOGGER="tensorboard"
WANDB_PROJECT="rwm-anymal"

# Run name for easy identification (will appear in wandb dashboard)
RUN_NAME="finetune-ensemble-${UNCERTAINTY_METRIC}${UNCERTAINTY_PENALTY}-${DYNAMICS_MODE}"
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
    echo "Please update PRETRAIN_DIR in this script to point to your ENSEMBLE pretrain run."
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
echo "U-RWM Phase 2: Ensemble MBPO-PPO Finetuning"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0"
echo ""
echo "Loading from ensemble pretrain:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""
echo "Ensemble Configuration:"
echo "  - Ensemble Size: $ENSEMBLE_SIZE"
echo "  - Uncertainty Penalty: $UNCERTAINTY_PENALTY"
echo "  - Uncertainty Metric: $UNCERTAINTY_METRIC"
echo "  - Prior Scale: $PRIOR_SCALE (must match pretrain!)"
echo "  - Prior Hidden Divisor: $PRIOR_HIDDEN_DIV"
echo "  - (Each imagination env follows a random ensemble member)"
echo ""
echo "Bootstrap Settings:"
echo "  - Bootstrap: $BOOTSTRAP (ALL members see ALL data)"
echo "  - Hypothesis: Priors provide diversity, bootstrap unnecessary"
echo "Imagination Configuration (from Table S11):"
echo "  - Imagination Envs: $IMAGINATION_ENVS"
echo "  - Imagination Steps: $IMAGINATION_STEPS"
echo "  - Real Envs: $NUM_ENVS"
echo "  - Max Iterations: $MAX_ITERATIONS"
echo "  - Policy Learning Rate: 0.001"
echo "  - WM Warmup Iterations: 500 (default)"
echo ""
echo "Logging:"
echo "  - Logger: $LOGGER"
echo "  - W&B Project: $WANDB_PROJECT"
echo "  - Run Name: $RUN_NAME"
echo ""
echo "Expected training time: ~12-18 hours"
echo "Checkpoints will be saved to:"
echo "  logs/rsl_rl/anymal_d_flat/<timestamp>_finetune/"
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
    --load_run 2026-01-25_11-57-10_pretrain-ensemble \
    --checkpoint model_${CHECKPOINT_ITER}.pt \
    --system_dynamics_load_path $CHECKPOINT_PATH \
    --logger $LOGGER \
    --log_project_name $WANDB_PROJECT \
    --run_name $RUN_NAME \
    agent.system_dynamics.ensemble_size=$ENSEMBLE_SIZE \
    agent.system_dynamics.uncertainty_metric=$UNCERTAINTY_METRIC \
    agent.system_dynamics.prior_scale=$PRIOR_SCALE \
    agent.system_dynamics.prior_hidden_div=$PRIOR_HIDDEN_DIV \
    agent.system_dynamics.bootstrap=$BOOTSTRAP \
    agent.imagination.num_envs=$IMAGINATION_ENVS \
    agent.imagination.num_steps=$IMAGINATION_STEPS \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo ""
echo "========================================"
echo "Ensemble Finetuning Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Ensemble MBPPO finetuning completed!"
    echo ""
    echo "The finetuned policy was trained using:"
    echo "  - Real environment rollouts (for world model updates)"
    echo "  - Imagined rollouts ($IMAGINATION_ENVS envs x $IMAGINATION_STEPS steps per iteration)"
    echo "  - ${ENSEMBLE_SIZE}-member ensemble with trajectory sampling"
    echo "  - Uncertainty penalty weight: $UNCERTAINTY_PENALTY"
    echo ""
    echo "Checkpoint location:"
    LATEST_LOG=$(ls -td logs/rsl_rl/anymal_d_flat/*finetune*/ 2>/dev/null | head -1)
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
    echo "  2. Compare with single-model finetune to measure ensemble benefit"
    echo ""
    echo "  3. Try different uncertainty penalties:"
    echo "     - 0.0: No penalty (pure trajectory sampling)"
    echo "     - -0.1: Mild conservative (current)"
    echo "     - -1.0: Strong conservative (as in offline U-RWM)"
else
    echo ""
    echo "FAILED: Ensemble finetuning failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/rwm-ensemble-finetune-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - Checkpoint was trained with different ensemble_size"
    echo "  - Checkpoint was trained with different prior_scale (mismatch causes state_dict error)"
    echo "  - Checkpoint path incorrect (update PRETRAIN_DIR)"
    echo "  - OOM: reduce NUM_ENVS or IMAGINATION_ENVS"
fi

exit $EXITCODE
