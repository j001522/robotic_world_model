#!/bin/bash
#SBATCH --job-name=rwm-visualize
#SBATCH --partition=gpu_vis
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-visualize-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-visualize-%j.err

# =============================================================================
# RWM Visualization: Compare Real vs Imagined Rollouts
# =============================================================================
# This script visualizes the learned dynamics model by rolling out the model
# autoregressively in imagination, conditioned on actions from the learned policy.
#
# Usage:
#   sbatch slurm_visualize_rwm.sh
# =============================================================================

# ======================= CONFIGURATION =======================
# Path to pretrain run directory (relative to robotic_world_model/)
PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-25_11-36-30_pretrain"

# Checkpoint iteration to load
CHECKPOINT_ITER=2500

# Video settings
VIDEO_LENGTH=400
NUM_ENVS=4
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

# Construct checkpoint path (absolute path)
CHECKPOINT_PATH="/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/${PRETRAIN_DIR}/model_${CHECKPOINT_ITER}.pt"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Available runs:"
    ls -la logs/rsl_rl/anymal_d_flat/ 2>/dev/null || echo "  No runs found"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo "RWM Visualization: Real vs Imagination"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0"
echo ""
echo "Loading from:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo ""
echo "Video settings:"
echo "  - Length: $VIDEO_LENGTH steps"
echo "  - Num envs: $NUM_ENVS"
echo ""
echo "Output will be saved to:"
echo "  ${PRETRAIN_DIR}/videos/play/"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run visualization inside container (headless with video recording)
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/visualize.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \
    --headless \
    --video \
    --video_length $VIDEO_LENGTH \
    --num_envs $NUM_ENVS \
    --checkpoint $CHECKPOINT_PATH \
    --system_dynamics_load_path $CHECKPOINT_PATH

EXITCODE=$?

echo ""
echo "========================================"
echo "Visualization Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Visualization completed!"
    echo ""
    echo "Video saved to:"
    find "${PRETRAIN_DIR}/videos" -name "*.mp4" -mmin -5 2>/dev/null | head -5
    echo ""
    echo "Copy to local machine with:"
    echo "  scp snellius:$(pwd)/${PRETRAIN_DIR}/videos/play/*.mp4 ."
else
    echo ""
    echo "FAILED: Visualization failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/rwm-visualize-$SLURM_JOB_ID.err"
fi

exit $EXITCODE
