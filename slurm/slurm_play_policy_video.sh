#!/bin/bash
#SBATCH --job-name=play-policy-video
#SBATCH --partition=vis
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --mem=80G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/play-policy-video-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/play-policy-video-%j.err

# =============================================================================
# Policy Video Recording
# =============================================================================
# This script runs a trained policy in Isaac Sim and records video footage.
# Uses vis partition for display/GPU capabilities required for video encoding.
#
# For headless evaluation without video, use slurm_play_policy.sh instead.
#
# Usage:
#   sbatch slurm_play_policy_video.sh
#   or edit configuration below then submit
# =============================================================================

# ======================= CONFIGURATION =======================
# Path to model checkpoint (relative to robotic_world_model/)
CHECKPOINT_PATH="logs/rsl_rl/anymal_d_flat/2026-01-26_15-32-54_finetune-ensemble-std-0.1-PRIORS_noboot/model_4999.pt"

# Task for policy evaluation
TASK="Isaac-Velocity-Flat-Anymal-D-Play-v0"

# Number of environments to evaluate (default: 32)
NUM_ENVS=32

# Video recording parameters
VIDEO_LENGTH=400  # Number of steps to record
VIDEO_QUALITY="high"  # Can be low, medium, high

# Random seed for reproducible evaluation
SEED=42

# Output directory for videos (auto-generated if not specified)
VIDEO_OUTPUT_DIR=""  # Empty = <checkpoint_dir>/videos
# =============================================================

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
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

# Set video output directory if not specified
if [ -z "$VIDEO_OUTPUT_DIR" ]; then
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
    VIDEO_OUTPUT_DIR="${CHECKPOINT_DIR}/videos"
fi

# Print configuration
echo ""
echo "Policy Video Recording Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Task: $TASK"
echo "  Num Envs: $NUM_ENVS"
echo "  Video Length: $VIDEO_LENGTH steps"
echo "  Video Quality: $VIDEO_QUALITY"
echo "  Seed: $SEED"
echo "  Output Dir: $VIDEO_OUTPUT_DIR"
echo ""
echo "This will:"
echo "  - Load trained policy from checkpoint"
echo "  - Run policy in Isaac Sim simulation"
echo "  - Record video of robot performance"
echo "  - Save video to output directory"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run policy evaluation with video recording inside container
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
    --video \
    --video_length $VIDEO_LENGTH \
    --headless

EXITCODE=$?

echo ""
echo "========================================"
echo "Policy Video Recording Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Policy video recording completed!"
    echo ""
    echo "Video(s) saved to: $VIDEO_OUTPUT_DIR"
    echo ""
    echo "To copy videos to your local machine:"
    echo "  scp -r $VIDEO_OUTPUT_DIR/*.mp4 user@your-machine:/path/to/destination/"
    echo ""
    echo "Common video players:"
    echo "  - VLC Media Player"
    echo "  - mpv"
    echo "  - QuickTime (macOS)"
else
    echo ""
    echo "FAILED: Policy video recording failed with exit code $EXITCODE"
    echo ""
    echo "Check error log:"
    echo "  logs/slurm/play-policy-video-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - vis partition busy (try gpu partition with --headless if display not needed)"
    echo "  - Out of memory: reduce NUM_ENVS or VIDEO_LENGTH"
    echo "  - Disk space: ensure enough space for video output"
fi

exit $EXITCODE