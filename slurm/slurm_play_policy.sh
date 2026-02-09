#!/bin/bash
#SBATCH --job-name=play-policy
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/play-policy-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/play-policy-%j.err

# =============================================================================
# Policy Play/Testing
# =============================================================================
# This script runs a trained policy in Isaac Sim for evaluation and testing.
# Uses GPU partition for headless evaluation (no video).
#
# For video recording, use slurm_play_policy_video.sh instead.
#
# Usage:
#   sbatch slurm_play_policy.sh
#   or edit CHECKPOINT_PATH below then submit
# =============================================================================

# ======================= CONFIGURATION =======================
# Path to model checkpoint (relative to robotic_world_model/)
CHECKPOINT_PATH="logs/rsl_rl/anymal_d_flat/2026-01-26_15-32-54_finetune-ensemble-std-0.1-PRIORS_noboot/model_4999.pt"

# Task for policy evaluation
TASK="Isaac-Velocity-Flat-Anymal-D-Play-v0"

# Number of environments to evaluate (default: 32)
NUM_ENVS=32

# Run in real-time? (use for visual inspection with X11 forwarding)
REAL_TIME=false

# Random seed for reproducible evaluation
SEED=42
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

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Please update CHECKPOINT_PATH in this script."
    echo "Available checkpoints:"
    find logs/rsl_rl/anymal_d_flat/ -name "model_*.pt" -type f | head -20
    echo ""
    exit 1
fi

# Print configuration
echo ""
echo "Policy Play Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Task: $TASK"
echo "  Num Envs: $NUM_ENVS"
echo "  Real Time: $REAL_TIME"
echo "  Seed: $SEED"
echo ""
echo "This will:"
echo "  - Load trained policy from checkpoint"
echo "  - Run policy in Isaac Sim simulation"
echo "  - Display evaluation metrics in real-time"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run policy evaluation inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
    --task $TASK \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_envs $NUM_ENVS \
    --seed $SEED \
    $(if [ "$REAL_TIME" = "true" ]; then echo "--real-time"; fi) \
    --headless

EXITCODE=$?

echo ""
echo "========================================"
echo "Policy Play Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Policy evaluation completed!"
    echo ""
    echo "Check the output log for performance metrics:"
    echo "  - Mean reward"
    echo "  - Episode length"
    echo "  - Velocity tracking errors"
else
    echo ""
    echo "FAILED: Policy evaluation failed with exit code $EXITCODE"
    echo ""
    echo "Check error log:"
    echo "  logs/slurm/play-policy-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: reduce NUM_ENVS"
    echo "  - Checkpoint path incorrect"
    echo "  - Task mismatch with checkpoint"
fi

exit $EXITCODE