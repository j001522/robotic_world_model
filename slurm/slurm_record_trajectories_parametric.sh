#!/bin/bash
#SBATCH --job-name=record-traj
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# =============================================================================
# Parametric Trajectory Recording Script
# =============================================================================
# Usage: sbatch slurm_record_trajectories_parametric.sh \
#          <checkpoint> <task> <output_file> <num_traj> <traj_len> <num_envs> <seed>
# =============================================================================

CHECKPOINT_PATH="$1"
TASK="$2"
OUTPUT_FILE="$3"
NUM_TRAJECTORIES="${4:-200}"
TRAJECTORY_LENGTH="${5:-200}"
NUM_ENVS="${6:-64}"
SEED="${7:-42}"

if [ -z "$CHECKPOINT_PATH" ] || [ -z "$TASK" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: sbatch slurm_record_trajectories_parametric.sh <checkpoint> <task> <output> [num_traj] [traj_len] [num_envs] [seed]"
    exit 1
fi

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Task: $TASK"
echo "  Output: $OUTPUT_FILE"
echo "  Num Trajectories: $NUM_TRAJECTORIES"
echo "  Trajectory Length: $TRAJECTORY_LENGTH"
echo "  Num Envs: $NUM_ENVS"
echo "  Seed: $SEED"
echo "========================================"

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Create output directory
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Validate checkpoint
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Run inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/analysis/record_trajectories.py \
    --task "$TASK" \
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
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

exit $EXITCODE
