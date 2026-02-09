#!/bin/bash
#SBATCH --job-name=eval-world-model
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/eval-wm-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/eval-wm-%j.err

# =============================================================================
# World Model Evaluation (No Isaac Sim)
# =============================================================================
# This script evaluates a trained world model by generating training curve plots
# and extracting performance metrics from TensorBoard logs.
#
# No Isaac Sim required - just analyzes logs and generates matplotlib plots.
#
# Usage:
#   sbatch slurm_eval_world_model.sh
#   or edit CHECKPOINT_PATH below then submit
# =============================================================================

# ======================= CONFIGURATION =======================
# Path to model checkpoint (relative to robotic_world_model/)
# Examples:
#   - "logs/rsl_rl/anymal_d_flat/2026-01-26_15-32-54_finetune-ensemble-std-0.1-PRIORS_noboot/model_4999.pt"
#   - "logs/rsl_rl/anymal_d_flat/2026-01-25_12-55-50_pretrain-ensemble-prior1.0-noboot/model_2500.pt"
#CHECKPOINT_PATH="logs/rsl_rl/anymal_d_flat/2026-01-26_15-32-54_finetune-ensemble-std-0.1-PRIORS_noboot/model_4999.pt"
CHECKPOINT_PATH="logs/rsl_rl/anymal_d_flat/2026-01-25_18-59-18_finetune-ensemble-penalty-0.1/model_4999.pt"
#CHECKPOINT_PATH="logs/rsl_rl/anymal_d_flat/2026-01-26_11-38-52_finetune-ensemble-penalty-0.25/model_4999.pt"

# Output directory for evaluation plots (auto-generated if not specified)
OUTPUT_DIR=""  # Empty = <checkpoint_dir>/eval_plots

# Device for analysis (cuda or cpu)
DEVICE="cuda"
# =============================================================

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

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

# Set output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    CHECKPOINT_DIR=$(dirname "$CHECKPOINT_PATH")
    OUTPUT_DIR="${CHECKPOINT_DIR}/eval_plots"
fi

# Print configuration
echo ""
echo "World Model Evaluation Configuration:"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo ""
echo "This will generate:"
echo "  - Training curves (reward, loss, episode length)"
echo "  - World model auxiliary losses"
echo "  - Autoregressive error plots with different noise levels"
echo "  - Summary statistics"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run world model evaluation
python scripts/evaluate_world_model.py \
  --checkpoint "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"

EXITCODE=$?

echo ""
echo "========================================"
echo "World Model Evaluation Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: World model evaluation completed!"
    echo ""
    echo "Generated plots:"
    echo "  - $OUTPUT_DIR/training_curves.png"
    echo "  - $OUTPUT_DIR/wm_auxiliary_losses.png"
    echo "  - $OUTPUT_DIR/autoregressive_error_noised.png"
    echo ""
    echo "To copy plots to your local machine:"
    echo "  scp -r $OUTPUT_DIR/*.png user@your-machine:/path/to/destination/"
else
    echo ""
    echo "FAILED: World model evaluation failed with exit code $EXITCODE"
    echo ""
    echo "Check error log:"
    echo "  logs/slurm/eval-wm-$SLURM_JOB_ID.err"
fi

exit $EXITCODE