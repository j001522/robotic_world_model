#!/bin/bash
#SBATCH --job-name=rwm-vis-stream
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-vis-stream-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-vis-stream-%j.err

# =============================================================================
# RWM Visualization with Isaac Sim Streaming
# =============================================================================
# Launches the visualization with WebRTC streaming so you can view it
# using the Isaac Sim Streaming Client on your local machine.
#
# IMPORTANT: A100 GPUs have NVENC support required for streaming.
#            H100 GPUs do NOT have NVENC and will show a black screen.
#
# Usage:
#   1. sbatch slurm/slurm_visualize_rwm_streaming.sh
#   2. Wait for job to start, check the .out file for connection instructions
#   3. Set up SSH tunnel (see instructions below)
#   4. Connect with Isaac Sim Streaming Client to localhost:49100
# =============================================================================

# ======================= CONFIGURATION =======================
# Path to pretrain run directory
PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-25_11-36-30_pretrain"

# Checkpoint iteration to load
CHECKPOINT_ITER=2500

# Number of environments to visualize
NUM_ENVS=4
# =============================================================

# WebRTC streaming port
STREAMING_PORT=49100

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start Time: $(date)"
echo "========================================"

# Get node info for SSH tunnel instructions
NODE_HOSTNAME=$(hostname)
NODE_SHORT=$(hostname -s)

echo ""
echo "============================================"
echo "TO CONNECT WITH ISAAC SIM STREAMING CLIENT:"
echo "============================================"
echo ""
echo "1. Set up SSH tunnel (run on your laptop):"
echo ""
echo "   ssh -J cmeo@snellius.surf.nl -N -L ${STREAMING_PORT}:localhost:${STREAMING_PORT} cmeo@${NODE_SHORT}"
echo ""
echo "2. Open Isaac Sim Streaming Client and connect to:"
echo "   localhost:${STREAMING_PORT}"
echo ""
echo "============================================"
echo ""

# Environment setup
export ISAAC_SIM_ROOT=/isaac-sim
export ISAAC_SIM_CACHE_DIR=/projects/0/prjs0951/Giacomo/isaac-sim/cache
export OMNI_USER=$ISAAC_SIM_CACHE_DIR/ov
export OMNI_KIT_ALLOW_ROOT=1
export WARP_CACHE_ROOT=$ISAAC_SIM_CACHE_DIR/warp
export CARB_APP_PATH=$ISAAC_SIM_ROOT/kit
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=YES
export GIT_PYTHON_REFRESH=quiet

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Construct checkpoint path (absolute)
CHECKPOINT_PATH="/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/${PRETRAIN_DIR}/model_${CHECKPOINT_ITER}.pt"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    exit 1
fi

echo "Loading checkpoint: $CHECKPOINT_PATH"
echo ""
echo "Starting Isaac Sim with streaming enabled..."
echo ""

# Create logs directory
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run visualization with streaming (NOT headless)
apptainer exec --nv \
  --writable-tmpfs \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --bind $ISAAC_SIM_CACHE_DIR:/root/.cache \
  --bind $ISAAC_SIM_CACHE_DIR/config:/root/.config \
  --bind $ISAAC_SIM_CACHE_DIR/documents:/root/Documents \
  --bind $ISAAC_SIM_CACHE_DIR/Kit:/root/.local/share/ov/Kit \
  --bind $ISAAC_SIM_CACHE_DIR/pkg:/root/.local/share/ov/pkg \
  --env OMNI_USER=$OMNI_USER \
  --env OMNI_KIT_ALLOW_ROOT=$OMNI_KIT_ALLOW_ROOT \
  --env WARP_CACHE_ROOT=$WARP_CACHE_ROOT \
  --env CARB_APP_PATH=$CARB_APP_PATH \
  --env OMNI_KIT_ACCEPT_EULA=$OMNI_KIT_ACCEPT_EULA \
  --env ACCEPT_EULA=$ACCEPT_EULA \
  --env PRIVACY_CONSENT=$PRIVACY_CONSENT \
  --env GIT_PYTHON_REFRESH=$GIT_PYTHON_REFRESH \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/visualize.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \
    --num_envs $NUM_ENVS \
    --checkpoint $CHECKPOINT_PATH \
    --system_dynamics_load_path $CHECKPOINT_PATH \
    --livestream 2 \
    --enable_cameras

EXITCODE=$?

echo ""
echo "========================================"
echo "Visualization Ended"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

exit $EXITCODE
