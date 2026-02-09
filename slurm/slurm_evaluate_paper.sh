#!/bin/bash
#SBATCH --job-name=eval-paper
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# =============================================================================
# Paper Evaluation SLURM Script
# =============================================================================
# Runs offline evaluations (hallucination horizon + noise robustness) on GPU.
# Automatically processes all trajectory files found in OUTPUT_DIR/trajectories/
#
# Usage: sbatch slurm_evaluate_paper.sh [output_dir]
#
# Environment variables:
#   HORIZON       - Prediction horizon for hallucination eval (default: 50)
#   INCLUDE_RUNS  - Comma-separated conditions to include (optional)
#   EXCLUDE_RUNS  - Comma-separated conditions to exclude (optional)
# =============================================================================

OUTPUT_DIR="${1:-results/paper}"

# Configurable parameters (can be set via environment or defaults)
HORIZON="${HORIZON:-200}"
INCLUDE_RUNS="${INCLUDE_RUNS:-}"
EXCLUDE_RUNS="${EXCLUDE_RUNS:-}"

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Horizon: $HORIZON"
if [ -n "$INCLUDE_RUNS" ]; then
    echo "  Include runs: $INCLUDE_RUNS"
fi
if [ -n "$EXCLUDE_RUNS" ]; then
    echo "  Exclude runs: $EXCLUDE_RUNS"
fi
echo "========================================"

# Helper function to check if a condition should be processed
should_process_condition() {
    local cond_key="$1"
    
    # If INCLUDE_RUNS is set, only include conditions in that list
    if [ -n "$INCLUDE_RUNS" ]; then
        if [[ ",$INCLUDE_RUNS," == *",$cond_key,"* ]]; then
            return 0  # Include
        else
            return 1  # Exclude
        fi
    fi
    
    # If EXCLUDE_RUNS is set, exclude conditions in that list
    if [ -n "$EXCLUDE_RUNS" ]; then
        if [[ ",$EXCLUDE_RUNS," == *",$cond_key,"* ]]; then
            return 1  # Exclude
        fi
    fi
    
    return 0  # Include by default
}

# Environment setup
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Activate conda if needed (adjust to your environment)
source ~/.bashrc
conda activate base 2>/dev/null || true

LOG_DIR="logs/rsl_rl/anymal_d_flat"

mkdir -p "$OUTPUT_DIR/hallucination_horizon"
mkdir -p "$OUTPUT_DIR/noise_robustness"

# Map short keys to full patterns for finding run directories
get_pattern_from_key() {
    local key="$1"
    case "$key" in
        "bs-nopen") echo "finetune-bs-nopen" ;;
        "bs-pen025") echo "finetune-bs-pen025" ;;
        "bs-pen006") echo "finetune-bs-pen006" ;;
        "rp-pen025-std") echo "finetune-rp-pen025-std" ;;
        "rp-pen025-var") echo "finetune-rp-pen025-var" ;;
        "rp-pen006-std") echo "finetune-rp-pen006-std" ;;
        *) echo "finetune-$key" ;;
    esac
}

echo ""
echo "=== Detecting trajectory files ==="

# Find all trajectory files and extract condition/seed
TRAJ_FILES=$(find "$OUTPUT_DIR/trajectories" -name "traj_*.pt" 2>/dev/null | sort)

if [ -z "$TRAJ_FILES" ]; then
    echo "ERROR: No trajectory files found in $OUTPUT_DIR/trajectories/"
    exit 1
fi

echo "Found trajectory files:"
for traj_file in $TRAJ_FILES; do
    echo "  $(basename $traj_file)"
done

echo ""
echo "=== Hallucination Horizon Evaluation ==="
echo "  Mode: velocity-only (computing errors on v, w, q_dot components only)"
echo "  Horizon: $HORIZON"

for traj_file in $TRAJ_FILES; do
    # Parse trajectory filename: traj_<cond_key>_seed<N>.pt
    filename=$(basename "$traj_file")
    if [[ $filename =~ ^traj_(.+)_seed([0-9]+)\.pt$ ]]; then
        cond_key="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"
    else
        echo "  Skipping $filename: could not parse condition/seed"
        continue
    fi
    
    # Check if this condition should be processed
    if ! should_process_condition "$cond_key"; then
        echo "  Skipping $cond_key seed $seed: filtered out by INCLUDE/EXCLUDE"
        continue
    fi
    
    # Get the full pattern to find the run directory
    pattern=$(get_pattern_from_key "$cond_key")
    
    run_dir=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*${pattern}-seed${seed}*" 2>/dev/null | head -1)
    
    if [ -z "$run_dir" ]; then
        echo "  Skipping $cond_key seed $seed: run directory not found (pattern: $pattern)"
        continue
    fi
    
    # Find latest checkpoint (always use the most recent one)
    checkpoint=$(ls "$run_dir"/model_*.pt 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$checkpoint" ]; then
        echo "  Skipping $cond_key seed $seed: no checkpoint found"
        continue
    fi
    
    echo "  Evaluating: $cond_key seed $seed"
    echo "    Checkpoint: $checkpoint (latest)"
    echo "    Trajectories: $traj_file"
    
    # Use --velocity_only to compute errors on velocity components only (v, w, q_dot)
    # Use --save_components to also save per-component breakdown
    python scripts/analysis/evaluate_hallucination_horizon.py \
        --checkpoint "$checkpoint" \
        --trajectory_data "$traj_file" \
        --horizon "$HORIZON" \
        --velocity_only \
        --save_components \
        --output_dir "$OUTPUT_DIR/hallucination_horizon" \
        --device cuda || echo "    WARNING: Hallucination horizon evaluation failed"
done

echo ""
echo "=== Noise Robustness Evaluation ==="

for traj_file in $TRAJ_FILES; do
    # Parse trajectory filename: traj_<cond_key>_seed<N>.pt
    filename=$(basename "$traj_file")
    if [[ $filename =~ ^traj_(.+)_seed([0-9]+)\.pt$ ]]; then
        cond_key="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"
    else
        continue
    fi
    
    # Check if this condition should be processed
    if ! should_process_condition "$cond_key"; then
        echo "  Skipping $cond_key seed $seed: filtered out by INCLUDE/EXCLUDE"
        continue
    fi
    
    pattern=$(get_pattern_from_key "$cond_key")
    run_dir=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*${pattern}-seed${seed}*" 2>/dev/null | head -1)
    
    if [ -z "$run_dir" ]; then
        echo "  Skipping $cond_key seed $seed: run directory not found"
        continue
    fi
    
    # Find latest checkpoint (always use the most recent one)
    checkpoint=$(ls "$run_dir"/model_*.pt 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$checkpoint" ]; then
        echo "  Skipping $cond_key seed $seed: no checkpoint found"
        continue
    fi
    
    echo "  Evaluating: $cond_key seed $seed"
    echo "    Checkpoint: $checkpoint (latest)"
    
    python scripts/analysis/evaluate_noise_robustness.py \
        --checkpoint "$checkpoint" \
        --trajectory_data "$traj_file" \
        --noise_levels 0.0 0.1 0.2 0.4 0.5 0.8 \
        --output_dir "$OUTPUT_DIR/noise_robustness" \
        --no_plot \
        --device cuda || echo "    WARNING: Noise robustness evaluation failed"
done

echo ""
echo "=== Error vs Uncertainty Evaluation ==="

mkdir -p "$OUTPUT_DIR/error_vs_uncertainty"

for traj_file in $TRAJ_FILES; do
    # Parse trajectory filename: traj_<cond_key>_seed<N>.pt
    filename=$(basename "$traj_file")
    if [[ $filename =~ ^traj_(.+)_seed([0-9]+)\.pt$ ]]; then
        cond_key="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"
    else
        continue
    fi
    
    # Check if this condition should be processed
    if ! should_process_condition "$cond_key"; then
        echo "  Skipping $cond_key seed $seed: filtered out by INCLUDE/EXCLUDE"
        continue
    fi
    
    pattern=$(get_pattern_from_key "$cond_key")
    run_dir=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*${pattern}-seed${seed}*" 2>/dev/null | head -1)
    
    if [ -z "$run_dir" ]; then
        echo "  Skipping $cond_key seed $seed: run directory not found"
        continue
    fi
    
    # Find latest checkpoint (always use the most recent one)
    checkpoint=$(ls "$run_dir"/model_*.pt 2>/dev/null | sort -V | tail -1)
    
    if [ -z "$checkpoint" ]; then
        echo "  Skipping $cond_key seed $seed: no checkpoint found"
        continue
    fi
    
    echo "  Evaluating: $cond_key seed $seed"
    echo "    Checkpoint: $checkpoint (latest)"
    
    python scripts/analysis/evaluate_error_vs_uncertainty.py \
        --checkpoint "$checkpoint" \
        --trajectory_data "$traj_file" \
        --horizon "$HORIZON" \
        --velocity_only \
        --output_dir "$OUTPUT_DIR/error_vs_uncertainty" \
        --device cuda || echo "    WARNING: Error vs uncertainty evaluation failed"
done

echo ""
echo "========================================"
echo "Evaluations Complete"
echo "Results in: $OUTPUT_DIR"
echo "Exit Code: $?"
echo "End Time: $(date)"
echo "========================================"
