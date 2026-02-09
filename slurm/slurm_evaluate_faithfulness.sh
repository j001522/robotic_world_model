#!/bin/bash
#SBATCH --job-name=eval-faith
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --mem=64G

# =============================================================================
# Faithfulness Gap Evaluation SLURM Script
# =============================================================================
# Evaluates real rewards from policy checkpoints in Isaac Sim to measure
# the gap between imagined and real rewards.
#
# Usage: sbatch slurm_evaluate_faithfulness.sh [output_dir] [tensorboard_dir]
#
# Environment variables:
#   INCLUDE_RUNS  - Comma-separated conditions to include (optional)
#   EXCLUDE_RUNS  - Comma-separated conditions to exclude (optional)
#   CONDITIONS    - Pre-computed list of conditions to evaluate (optional)
# =============================================================================

OUTPUT_DIR="${1:-results/paper}"
TENSORBOARD_DIR="${2:-results/paper/tensorboard/aggregated}"

# Filtering options
INCLUDE_RUNS="${INCLUDE_RUNS:-}"
EXCLUDE_RUNS="${EXCLUDE_RUNS:-}"
CONDITIONS="${CONDITIONS:-}"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Output Directory: $OUTPUT_DIR"
echo "  TensorBoard Data: $TENSORBOARD_DIR"
if [ -n "$INCLUDE_RUNS" ]; then
    echo "  Include runs: $INCLUDE_RUNS"
fi
if [ -n "$EXCLUDE_RUNS" ]; then
    echo "  Exclude runs: $EXCLUDE_RUNS"
fi
if [ -n "$CONDITIONS" ]; then
    echo "  Conditions: $CONDITIONS"
fi
echo "========================================"

# Build filter argument for the Python script
FILTER_ARG=""
if [ -n "$CONDITIONS" ]; then
    # If specific conditions provided, build a regex filter
    # This is a simple approach - the Python script will filter further
    FILTER_ARG="--filter finetune"
fi

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

# Deactivate conda if active (apptainer will use its own python)
conda deactivate 2>/dev/null || true

cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

LOG_DIR="logs/rsl_rl/anymal_d_flat"
TASK="Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0"

mkdir -p "$OUTPUT_DIR/faithfulness_gap"

echo ""
echo "=== Faithfulness Gap Evaluation ==="
echo "Running policy evaluation in Isaac Sim to measure real rewards..."

# Run inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/analysis/evaluate_real_rewards.py \
    --task "$TASK" \
    --log_dir "$LOG_DIR" \
    --filter "finetune" \
    --checkpoint_steps 3000 4000 4999 \
    --num_episodes 100 \
    --num_envs 64 \
    --output_dir "$OUTPUT_DIR/faithfulness_gap" \
    --imagination_data_dir "$TENSORBOARD_DIR" \
    --headless

EXITCODE=$?

echo ""
echo "========================================"
echo "Faithfulness Gap Evaluation Complete"
echo "Results in: $OUTPUT_DIR/faithfulness_gap"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

exit $EXITCODE
