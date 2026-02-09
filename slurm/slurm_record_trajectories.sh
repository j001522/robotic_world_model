#!/bin/bash
#SBATCH --job-name=record-trajectories
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/record-traj-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/record-traj-%j.err

# =============================================================================
# Record Trajectories for Offline Evaluation
# =============================================================================
# This script runs trained policies in Isaac Sim to record state-action
# trajectories that can be used for offline world model evaluation:
#   - Hallucination horizon analysis (Plot A)
#   - Noise robustness with corrected metric (Plot C)
#
# The recorded trajectories are saved as .pt files containing:
#   - states: [N, T, state_dim] - System states over time
#   - actions: [N, T, action_dim] - Actions taken
#   - rewards: [N, T] - Rewards received
#   - dones: [N, T] - Episode termination flags
#
# Usage:
#   sbatch slurm_record_trajectories.sh
#   or edit CHECKPOINT_PATH and other settings below then submit
# =============================================================================

# ======================= CONFIGURATION =======================
# Path to model checkpoint (relative to robotic_world_model/)
# You can also pass this as an argument: sbatch slurm_record_trajectories.sh <checkpoint_path>
#CHECKPOINT_PATH="${1:-logs/rsl_rl/anymal_d_flat/2026-01-26_15-32-54_finetune-ensemble-std-0.1-PRIORS_noboot/model_4999.pt}"
CHECKPOINT_PATH="logs/rsl_rl/anymal_d_flat/2026-02-02_23-55-05_finetune-bs-nopen-seed42/model_4999.pt"
# Task for trajectory recording (should match training task)
TASK="Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0"

# Output file for trajectories
OUTPUT_DIR="results/paper/trajectories"
OUTPUT_FILE="$OUTPUT_DIR/trajectories_$(basename $(dirname $CHECKPOINT_PATH)).pt"

# Recording parameters
NUM_TRAJECTORIES=200     # Number of trajectories to record
TRAJECTORY_LENGTH=100    # Length of each trajectory (timesteps)
NUM_ENVS=64              # Number of parallel environments

# Random seed
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Please update CHECKPOINT_PATH in this script or pass as argument."
    echo "Usage: sbatch slurm_record_trajectories.sh <checkpoint_path>"
    echo ""
    echo "Available checkpoints:"
    find logs/rsl_rl/anymal_d_flat/ -name "model_*.pt" -type f 2>/dev/null | head -20
    echo ""
    exit 1
fi

# Print configuration
echo ""
echo "Trajectory Recording Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Task: $TASK"
echo "  Output: $OUTPUT_FILE"
echo "  Num Trajectories: $NUM_TRAJECTORIES"
echo "  Trajectory Length: $TRAJECTORY_LENGTH"
echo "  Num Envs: $NUM_ENVS"
echo "  Seed: $SEED"
echo ""
echo "This will:"
echo "  - Load trained policy from checkpoint"
echo "  - Run policy in Isaac Sim simulation"
echo "  - Record state-action trajectories"
echo "  - Save to .pt file for offline evaluation"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run trajectory recording inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/analysis/record_trajectories.py \
    --task $TASK \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$OUTPUT_FILE" \
    --num_trajectories $NUM_TRAJECTORIES \
    --trajectory_length $TRAJECTORY_LENGTH \
    --num_envs $NUM_ENVS \
    --seed $SEED \
    --headless

EXITCODE=$?

echo ""
echo "========================================"
echo "Trajectory Recording Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Trajectories recorded!"
    echo ""
    echo "Output saved to: $OUTPUT_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. Use for hallucination horizon analysis:"
    echo "     python scripts/analysis/evaluate_hallucination_horizon.py \\"
    echo "       --log_dir logs/rsl_rl/anymal_d_flat \\"
    echo "       --trajectory_data $OUTPUT_FILE \\"
    echo "       --output_dir results/paper/hallucination_horizon/"
    echo ""
    echo "  2. Use for corrected noise robustness:"
    echo "     python scripts/analysis/evaluate_noise_robustness.py \\"
    echo "       --checkpoint $CHECKPOINT_PATH \\"
    echo "       --trajectory_data $OUTPUT_FILE \\"
    echo "       --output_dir results/paper/noise_robustness/"
else
    echo ""
    echo "FAILED: Trajectory recording failed with exit code $EXITCODE"
    echo ""
    echo "Check error log:"
    echo "  logs/slurm/record-traj-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: reduce NUM_ENVS or NUM_TRAJECTORIES"
    echo "  - Checkpoint path incorrect"
    echo "  - Task mismatch with checkpoint"
fi

exit $EXITCODE
