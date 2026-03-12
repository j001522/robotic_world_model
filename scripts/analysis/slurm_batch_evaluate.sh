#!/bin/bash
#SBATCH --job-name=batch-eval
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/batch-eval-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/batch-eval-%j.err

# =============================================================================
# Batch Evaluate All Models (SLURM)
# =============================================================================
# Runs all offline evaluations (hallucination horizon, noise robustness,
# error vs uncertainty, correlation by horizon) on pre-recorded trajectories.
#
# This does NOT require Isaac Sim - only GPU for inference.
#
# Usage:
#   sbatch slurm_batch_evaluate.sh
#
# Environment variables (optional):
#   TRAJECTORY_DIR  - Directory with trajectories (default: results/paper/trajectories)
#   OUTPUT_DIR      - Where to save results (default: results/paper)
#   HORIZON         - Prediction horizon (default: 200)
#   EVALUATIONS     - Space-separated list of evaluations (default: all)
# =============================================================================

# Configuration (can override via environment)
TRAJECTORY_DIR="${TRAJECTORY_DIR:-results/paper/trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper}"
HORIZON="${HORIZON:-200}"
EVALUATIONS="${EVALUATIONS:-hallucination_horizon noise_robustness error_vs_uncertainty correlation_by_horizon}"
# NEW: IROS paper evaluation aggregations
IROS_AGGREGATIONS="${IROS_AGGREGATIONS:-growth_rates horizon_auc risk_coverage auroc hallucination_stats}"

# Fixed configuration
LOG_DIR="logs/rsl_rl/anymal_d_flat"
#LOG_DIR="logs/rsl_rl/franka_reach"

echo "========================================"
echo "Batch Evaluate Models"
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""
echo "Configuration:"
echo "  Log Dir: $LOG_DIR"
echo "  Trajectory Dir: $TRAJECTORY_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Horizon: $HORIZON"
echo "  Evaluations: $EVALUATIONS"
echo "========================================"

cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Activate conda environment
source ~/.bashrc
conda activate base 2>/dev/null || true

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs/slurm

# Check if trajectories exist
if [ ! -d "$TRAJECTORY_DIR" ] || [ -z "$(ls -A $TRAJECTORY_DIR 2>/dev/null)" ]; then
    echo ""
    echo "ERROR: No trajectory files found in $TRAJECTORY_DIR"
    echo ""
    echo "Run trajectory recording first:"
    echo "  sbatch scripts/analysis/slurm_batch_record.sh"
    exit 1
fi

# Run batch evaluation
python scripts/analysis/batch_evaluate.py \
    --log_dir "$LOG_DIR" \
    --trajectory_dir "$TRAJECTORY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --horizon "$HORIZON" \
    --evaluations $EVALUATIONS \
    --iros_aggregations $IROS_AGGREGATIONS \
    --device cuda

EXITCODE=$?

echo ""
echo "========================================"
echo "Batch Evaluation Complete"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS! Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -name "*.csv" -newer "$0" -type f 2>/dev/null | head -20
    echo ""
    echo "Next step - generate plots (can run locally):"
    echo "  python scripts/analysis/plot_all.py \\"
    echo "      --results_dir $OUTPUT_DIR \\"
    echo "      --output_dir $OUTPUT_DIR/figures"
else
    echo ""
    echo "FAILED! Check logs/slurm/batch-eval-$SLURM_JOB_ID.err"
fi

exit $EXITCODE
