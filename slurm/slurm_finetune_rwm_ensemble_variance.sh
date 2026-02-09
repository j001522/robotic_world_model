#!/bin/bash
#SBATCH --job-name=rwm-ensemble-ft-var
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-ensemble-finetune-variance-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-ensemble-finetune-variance-%j.err

# =============================================================================
# U-RWM Phase 2: Finetune with Ensemble MBPO-PPO (VARIANCE metric)
# =============================================================================
# Same as slurm_finetune_rwm_ensemble.sh but uses VARIANCE as uncertainty metric
# instead of STD for comparison.
#
# Key differences:
#   - Uses variance (theoretically sound, additive across dims, stronger penalty)
#   - STD version uses same units as state, milder penalty
#
# Usage:
#   sbatch slurm_finetune_rwm_ensemble_variance.sh
# =============================================================================

# ======================= CONFIGURATION =======================
PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-25_11-57-10_pretrain-ensemble"

# Checkpoint iteration to load
CHECKPOINT_ITER=2500

# Finetuning parameters (from Table S11)
MAX_ITERATIONS=2500
NUM_ENVS=4096
IMAGINATION_ENVS=4096
IMAGINATION_STEPS=100

# Ensemble parameters
ENSEMBLE_SIZE=5

# Uncertainty penalty weight (negative = penalty for high uncertainty)
UNCERTAINTY_PENALTY=-0.1

# Uncertainty metric: VARIANCE for this experiment
UNCERTAINTY_METRIC="variance"
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
RUN_NAME="finetune-ensemble-variance-penalty${UNCERTAINTY_PENALTY}"
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
echo "  - (Each imagination env follows a random ensemble member)"
echo ""
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
    echo "  - Checkpoint path incorrect (update PRETRAIN_DIR)"
    echo "  - OOM: reduce NUM_ENVS or IMAGINATION_ENVS"
fi

exit $EXITCODE
