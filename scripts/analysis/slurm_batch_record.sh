#!/bin/bash
#SBATCH --job-name=batch-record
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/batch-record-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/batch-record-%j.err

# =============================================================================
# Batch Record Trajectories (SLURM)
# =============================================================================
# Records trajectories for ALL finetune runs in a single job.
# Runs inside Isaac Sim container.
#
# Usage:
#   sbatch slurm_batch_record.sh
#
# Or with custom parameters:
#   sbatch slurm_batch_record.sh --output_dir results/paper/trajectories --num_trajectories 200
#
# Environment variables (optional):
#   OUTPUT_DIR          - Where to save trajectories (default: results/paper/trajectories)
#   NUM_TRAJECTORIES    - Trajectories per checkpoint (default: 100)
#   TRAJECTORY_LENGTH   - Steps per trajectory (default: 200)
#   SKIP_EXISTING       - Skip runs with existing files (default: true)
# =============================================================================

# Configuration (can override via environment)
OUTPUT_DIR="${OUTPUT_DIR:-results/paper/trajectories}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-100}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-200}"
NUM_ENVS="${NUM_ENVS:-64}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

# Fixed configuration
TASK="Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0"
LOG_DIR="logs/rsl_rl/anymal_d_flat"

echo "========================================"
echo "Batch Record Trajectories"
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Log Dir: $LOG_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Num Trajectories: $NUM_TRAJECTORIES"
echo "  Trajectory Length: $TRAJECTORY_LENGTH"
echo "  Num Envs: $NUM_ENVS"
echo "  Skip Existing: $SKIP_EXISTING"
echo "========================================"

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/slurm

# Build arguments
ARGS=(
    --task "$TASK"
    --log_dir "$LOG_DIR"
    --output_dir "$OUTPUT_DIR"
    --num_trajectories "$NUM_TRAJECTORIES"
    --trajectory_length "$TRAJECTORY_LENGTH"
    --num_envs "$NUM_ENVS"
)

if [ "$SKIP_EXISTING" = "true" ]; then
    ARGS+=(--skip_existing)
fi

# Run batch recording inside Isaac Sim container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/analysis/batch_record_trajectories.py "${ARGS[@]}"

EXITCODE=$?

echo ""
echo "========================================"
echo "Batch Recording Complete"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS! Trajectories saved to: $OUTPUT_DIR"
    echo ""
    echo "Next step - run evaluations:"
    echo "  sbatch scripts/analysis/slurm_batch_evaluate.sh"
else
    echo ""
    echo "FAILED! Check logs/slurm/batch-record-$SLURM_JOB_ID.err"
fi

exit $EXITCODE
