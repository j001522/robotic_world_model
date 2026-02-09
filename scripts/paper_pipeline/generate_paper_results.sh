#!/bin/bash
# =============================================================================
# Paper Results Generation Pipeline
# =============================================================================
# This script orchestrates the complete pipeline for generating publication
# figures. It automatically detects conditions with 3+ seeds from the log
# directory.
#
# Pipeline stages:
#   1. Extract TensorBoard data to CSV
#   2. Record trajectories for offline evaluation (requires Isaac Sim)
#   3. Evaluate hallucination horizon (offline, no Isaac Sim)
#   4. Evaluate noise robustness (offline, no Isaac Sim)
#   3b. Evaluate faithfulness gap (requires Isaac Sim)
#   5. Generate publication figures
#
# Usage:
#   ./generate_paper_results.sh [stage] [options]
#
# Stages:
#   all         - Run all stages (default)
#   extract     - Stage 1: Extract TensorBoard data
#   record      - Stage 2: Record trajectories (SLURM job)
#   evaluate    - Stage 3-4: Run offline evaluations (SLURM job)
#   faithfulness - Stage 3b: Evaluate faithfulness gap (SLURM job, Isaac Sim)
#   plot        - Stage 5: Generate figures
#
# Options (environment variables):
#   INCLUDE_RUNS   - Comma-separated list of conditions to include (e.g., "bs-pen025,rp-pen025-std")
#   EXCLUDE_RUNS   - Comma-separated list of conditions to exclude
#   HORIZON        - Prediction horizon for hallucination evaluation (default: 200)
#   MIN_SEEDS      - Minimum seeds required per condition (default: 3)
#
# Examples:
#   # Run with specific conditions only
#   INCLUDE_RUNS="bs-pen025,rp-pen025-std" ./generate_paper_results.sh plot
#
#   # Exclude certain conditions
#   EXCLUDE_RUNS="bs-nopen" ./generate_paper_results.sh evaluate
#
#   # Set custom horizon for hallucination evaluation
#   HORIZON=200 ./generate_paper_results.sh evaluate
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_DIR"

# Auto-detect device (use cuda if available, otherwise cpu)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
else
    DEVICE="cpu"
    echo "NOTE: CUDA not available, using CPU for evaluations"
fi

LOG_DIR="logs/rsl_rl/anymal_d_flat"
OUTPUT_DIR="results/paper"
TASK="Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0"

# Default conditions to include (must be defined before use)
DEFAULT_INCLUDE_RUNS="bs-nopen,bs-pen025,rp-pen006-std,rp-pen025-std"

# Configurable parameters (can be overridden via environment variables)
MIN_SEEDS="${MIN_SEEDS:-3}"
HORIZON="${HORIZON:-200}"
# Default to main paper conditions if not specified
INCLUDE_RUNS="${INCLUDE_RUNS:-$DEFAULT_INCLUDE_RUNS}"
EXCLUDE_RUNS="${EXCLUDE_RUNS:-}"

# Colors (colorblind-friendly) - indexed by short key
# Fixed color assignments:
# Green = baseline (bs-nopen)
# Orange = bootstrap penalty (bs-pen025)
# Blue = rp006 (rp-pen006-std)
# Light Blue = rp025 (rp-pen025-std)
declare -A COLORS
COLORS["bs-nopen"]="#228833"        # Green - Baseline
COLORS["bs-pen025"]="#EE7733"       # Orange - Bootstrap + Penalty (0.25)
COLORS["bs-pen006"]="#EE9955"       # Light Orange - Bootstrap + Penalty (0.06)
COLORS["rp-pen006-std"]="#0077BB"   # Blue - Rand. Priors + Penalty (0.06, std)
COLORS["rp-pen025-std"]="#33BBEE"   # Light Blue - Rand. Priors + Penalty (0.25, std)
COLORS["rp-pen025-var"]="#009988"   # Teal - Rand. Priors + Penalty (0.25, var)

# Labels for plots
LABELS["bs-nopen"]="Bootstrap (no penalty)"
LABELS["bs-pen025"]="Bootstrap + Penalty (0.25)"
LABELS["bs-pen006"]="Bootstrap + Penalty (0.06)"
LABELS["rp-pen025-std"]="Rand. Priors + Penalty (0.25, std)"
LABELS["rp-pen025-var"]="Rand. Priors + Penalty (0.25, var)"
LABELS["rp-pen006-std"]="Rand. Priors + Penalty (0.06, std)"

echo "=============================================="
echo "Paper Results Generation Pipeline"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Horizon: $HORIZON"
if [ -n "$INCLUDE_RUNS" ]; then
    echo "Include runs: $INCLUDE_RUNS"
fi
if [ -n "$EXCLUDE_RUNS" ]; then
    echo "Exclude runs: $EXCLUDE_RUNS"
fi
echo ""

# Parse stage argument
STAGE="${1:-all}"

# =============================================================================
# Helper functions for include/exclude filtering
# =============================================================================
should_include_condition() {
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

# =============================================================================
# Auto-detect conditions with 3+ seeds
# =============================================================================
detect_conditions() {
    echo ""
    echo "=== Detecting conditions with $MIN_SEEDS+ seeds ==="
    
    # Find all finetune runs and count seeds per condition
    declare -A seed_counts
    declare -A condition_patterns
    
    for run_dir in "$LOG_DIR"/*finetune*seed*; do
        if [ ! -d "$run_dir" ]; then
            continue
        fi
        
        dir_name=$(basename "$run_dir")
        
        # Extract condition pattern (everything except timestamp and seed)
        # Pattern: YYYY-MM-DD_HH-MM-SS_<condition>-seed<N>
        if [[ $dir_name =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_(.+)-seed([0-9]+)$ ]]; then
            condition="${BASH_REMATCH[1]}"
            seed="${BASH_REMATCH[2]}"
            
            # Normalize condition name to short key
            short_key=""
            if [[ $condition == "finetune-bs-nopen" ]]; then
                short_key="bs-nopen"
            elif [[ $condition == "finetune-bs-pen025" ]]; then
                short_key="bs-pen025"
            elif [[ $condition == "finetune-bs-pen006" ]]; then
                short_key="bs-pen006"
            elif [[ $condition == "finetune-rp-pen025-std" ]]; then
                short_key="rp-pen025-std"
            elif [[ $condition == "finetune-rp-pen025-var" ]]; then
                short_key="rp-pen025-var"
            elif [[ $condition == "finetune-rp-pen006-std" ]]; then
                short_key="rp-pen006-std"
            else
                # Try to auto-generate short key
                short_key=$(echo "$condition" | sed 's/finetune-//')
            fi
            
            if [ -n "$short_key" ]; then
                seed_counts[$short_key]=$((${seed_counts[$short_key]:-0} + 1))
                condition_patterns[$short_key]="$condition"
            fi
        fi
    done
    
    # Build CONDITIONS array with only those having MIN_SEEDS or more
    # and respecting INCLUDE_RUNS/EXCLUDE_RUNS filters
    CONDITIONS=()
    CONDITION_PATTERNS=()
    
    echo ""
    echo "Found conditions:"
    for key in "${!seed_counts[@]}"; do
        count=${seed_counts[$key]}
        pattern=${condition_patterns[$key]}
        
        # Check include/exclude filters
        if ! should_include_condition "$key"; then
            echo "  $key: $count seeds (pattern: $pattern) [FILTERED OUT]"
            continue
        fi
        
        if [ $count -ge $MIN_SEEDS ]; then
            echo "  $key: $count seeds (pattern: $pattern) [INCLUDED]"
            CONDITIONS+=("$key")
            CONDITION_PATTERNS+=("$pattern")
        else
            echo "  $key: $count seeds (pattern: $pattern) [SKIPPED - need $MIN_SEEDS+]"
        fi
    done
    
    if [ ${#CONDITIONS[@]} -eq 0 ]; then
        echo ""
        echo "ERROR: No conditions found with $MIN_SEEDS+ seeds!"
        exit 1
    fi
    
    echo ""
    echo "Will process ${#CONDITIONS[@]} conditions: ${CONDITIONS[*]}"
}

# Store detected conditions in arrays
declare -a CONDITIONS
declare -a CONDITION_PATTERNS

# Detect conditions
detect_conditions

# Find seeds for a given condition pattern
find_seeds() {
    local pattern="$1"
    local seeds=()
    
    for run_dir in "$LOG_DIR"/*${pattern}-seed*; do
        if [ ! -d "$run_dir" ]; then
            continue
        fi
        
        dir_name=$(basename "$run_dir")
        if [[ $dir_name =~ seed([0-9]+) ]]; then
            seeds+=("${BASH_REMATCH[1]}")
        fi
    done
    
    echo "${seeds[@]}"
}

# =============================================================================
# Stage 1: Extract TensorBoard Data
# =============================================================================
extract_tensorboard() {
    echo ""
    echo "=== Stage 1: Extracting TensorBoard Data ==="
    
    mkdir -p "$OUTPUT_DIR/tensorboard"
    
    # Extract all finetune runs at once - the script handles grouping by condition
    echo "Extracting all finetune runs..."
    python scripts/analysis/extract_tensorboard_data.py \
        --log_dir "$LOG_DIR" \
        --output_dir "$OUTPUT_DIR/tensorboard" \
        --filter "finetune"
    
    echo ""
    echo "TensorBoard extraction complete."
    echo "All data: $OUTPUT_DIR/tensorboard/"
    echo "Aggregated by condition: $OUTPUT_DIR/tensorboard/aggregated/"
}

# =============================================================================
# Stage 2: Record Trajectories (SLURM Jobs)
# =============================================================================
record_trajectories() {
    echo ""
    echo "=== Stage 2: Recording Trajectories ==="
    echo "This will submit SLURM jobs for each run."
    echo "Using HORIZON: $HORIZON for trajectory length"
    
    mkdir -p "$OUTPUT_DIR/trajectories"
    mkdir -p "$OUTPUT_DIR/logs"
    
    job_ids=""
    
    for i in "${!CONDITIONS[@]}"; do
        cond_key="${CONDITIONS[$i]}"
        pattern="${CONDITION_PATTERNS[$i]}"
        
        # Find all seeds for this condition
        seeds=($(find_seeds "$pattern"))
        
        for seed in "${seeds[@]}"; do
            run_dir=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*${pattern}-seed${seed}*" 2>/dev/null | head -1)
            
            if [ -z "$run_dir" ]; then
                echo "  Skipping $cond_key seed $seed: run not found"
                continue
            fi
            
            # Find latest checkpoint (always use the most recent one)
            checkpoint=$(ls "$run_dir"/model_*.pt 2>/dev/null | sort -V | tail -1)
            
            if [ -z "$checkpoint" ]; then
                echo "  Skipping $cond_key seed $seed: no checkpoint found"
                continue
            fi
            
            output_file="$OUTPUT_DIR/trajectories/traj_${cond_key}_seed${seed}.pt"
            
            if [ -f "$output_file" ]; then
                echo "  Skipping $cond_key seed $seed: trajectories already exist"
                continue
            fi
            
            echo "  Submitting job: $cond_key seed $seed"
            echo "    Checkpoint: $checkpoint (latest)"
            
            # Submit SLURM job with horizon parameter
            # Trajectory length = HORIZON + history_horizon (32) to allow full evaluation
            TRAJ_LEN=$((HORIZON + 32))
            job_id=$(sbatch --parsable \
                --job-name="rec-${cond_key}-s${seed}" \
                --output="$OUTPUT_DIR/logs/record-${cond_key}-seed${seed}-%j.out" \
                --error="$OUTPUT_DIR/logs/record-${cond_key}-seed${seed}-%j.err" \
                slurm/slurm_record_trajectories_parametric.sh \
                "$checkpoint" "$TASK" "$output_file" 100 "$TRAJ_LEN" 64 $seed)
            
            job_ids="$job_ids $job_id"
            echo "    Submitted job: $job_id"
        done
    done
    
    if [ -n "$job_ids" ]; then
        echo ""
        echo "Submitted jobs:$job_ids"
        echo "Monitor with: squeue -u $USER"
        echo "Wait for completion before running 'evaluate' stage."
    else
        echo "No jobs submitted (trajectories may already exist)."
    fi
}

# =============================================================================
# Stage 3-4: Offline Evaluations (SLURM Job)
# =============================================================================
run_evaluations() {
    echo ""
    echo "=== Stage 3-4: Submitting Offline Evaluations SLURM Job ==="
    
    mkdir -p "$OUTPUT_DIR/hallucination_horizon"
    mkdir -p "$OUTPUT_DIR/noise_robustness"
    mkdir -p "$OUTPUT_DIR/logs"
    
    # Check if trajectory files exist
    traj_count=$(find "$OUTPUT_DIR/trajectories" -name "traj_*.pt" 2>/dev/null | wc -l)
    
    if [ "$traj_count" -eq 0 ]; then
        echo "ERROR: No trajectory files found in $OUTPUT_DIR/trajectories/"
        echo "Run './generate_paper_results.sh record' first and wait for jobs to complete."
        exit 1
    fi
    
    echo "Found $traj_count trajectory files."
    echo "Using horizon: $HORIZON"
    
    # Build conditions filter for SLURM job
    local conditions_filter=""
    if [ -n "$INCLUDE_RUNS" ]; then
        conditions_filter="INCLUDE_RUNS=$INCLUDE_RUNS"
    fi
    if [ -n "$EXCLUDE_RUNS" ]; then
        conditions_filter="$conditions_filter EXCLUDE_RUNS=$EXCLUDE_RUNS"
    fi
    
    # Submit SLURM job for evaluations
    # Export variables separately to avoid issues with comma-separated lists
    export HORIZON INCLUDE_RUNS EXCLUDE_RUNS
    job_id=$(sbatch --parsable \
        --output="$OUTPUT_DIR/logs/evaluate-%j.out" \
        --error="$OUTPUT_DIR/logs/evaluate-%j.err" \
        --export=ALL \
        slurm/slurm_evaluate_paper.sh "$OUTPUT_DIR")
    
    echo ""
    echo "Submitted evaluation job: $job_id"
    echo "Monitor with: squeue -u $USER"
    echo "View logs: tail -f $OUTPUT_DIR/logs/evaluate-${job_id}.out"
    echo ""
    echo "After job completes, run './generate_paper_results.sh plot'."
}

# =============================================================================
# Stage 3b: Faithfulness Gap Evaluation (SLURM Job with Isaac Sim)
# =============================================================================
run_faithfulness() {
    echo ""
    echo "=== Stage 3b: Submitting Faithfulness Gap Evaluation SLURM Job ==="
    
    mkdir -p "$OUTPUT_DIR/faithfulness_gap"
    mkdir -p "$OUTPUT_DIR/logs"
    
    # Check if TensorBoard data exists (needed for imagination rewards)
    if [ ! -d "$OUTPUT_DIR/tensorboard/aggregated" ]; then
        echo "WARNING: TensorBoard data not found at $OUTPUT_DIR/tensorboard/aggregated"
        echo "Run './generate_paper_results.sh extract' first for comparison with imagination rewards."
    fi
    
    # Build conditions list for the SLURM job
    local conditions_arg=""
    if [ ${#CONDITIONS[@]} -gt 0 ]; then
        conditions_arg=$(IFS=,; echo "${CONDITIONS[*]}")
    fi
    
    # Submit SLURM job for faithfulness gap evaluation
    job_id=$(sbatch --parsable \
        --output="$OUTPUT_DIR/logs/faithfulness-%j.out" \
        --error="$OUTPUT_DIR/logs/faithfulness-%j.err" \
        --export="ALL,INCLUDE_RUNS=$INCLUDE_RUNS,EXCLUDE_RUNS=$EXCLUDE_RUNS,CONDITIONS=$conditions_arg" \
        slurm/slurm_evaluate_faithfulness.sh "$OUTPUT_DIR" "$OUTPUT_DIR/tensorboard/aggregated")
    
    echo ""
    echo "Submitted faithfulness gap evaluation job: $job_id"
    echo "Monitor with: squeue -u $USER"
    echo "View logs: tail -f $OUTPUT_DIR/logs/faithfulness-${job_id}.out"
    echo ""
    echo "NOTE: This job requires Isaac Sim and may take longer than offline evaluations."
    echo "After job completes, run './generate_paper_results.sh plot' to generate figures."
}

# =============================================================================
# Stage 5: Generate Publication Figures
# =============================================================================
generate_figures() {
    echo ""
    echo "=== Stage 5: Generating Publication Figures ==="
    
    mkdir -p "$OUTPUT_DIR/figures"
    
    # Build conditions filter arguments for Python scripts
    local conditions_filter=""
    if [ -n "$INCLUDE_RUNS" ]; then
        conditions_filter="--conditions $INCLUDE_RUNS"
    fi
    
    # Generate training curves from TensorBoard data
    echo ""
    echo "--- Training Curves ---"
    python scripts/analysis/generate_paper_figures.py \
        --from_tensorboard \
        --data_dir "$OUTPUT_DIR/tensorboard/aggregated" \
        --output_dir "$OUTPUT_DIR/figures" \
        --format pdf \
        --plots training_curves \
        $conditions_filter
    
    # Generate hallucination horizon plots (from offline evaluation CSVs)
    echo ""
    echo "--- Hallucination Horizon ---"
    if [ -d "$OUTPUT_DIR/hallucination_horizon" ] && [ "$(ls -A $OUTPUT_DIR/hallucination_horizon/*.csv 2>/dev/null)" ]; then
        python scripts/analysis/plot_hallucination_horizon.py \
            --data_dir "$OUTPUT_DIR/hallucination_horizon" \
            --output_dir "$OUTPUT_DIR/figures" \
            --format pdf \
            $conditions_filter
    else
        echo "  Skipping: No hallucination horizon data found."
        echo "  Run './generate_paper_results.sh evaluate' first."
    fi
    
    # Generate noise robustness plots (from offline evaluation CSVs)
    echo ""
    echo "--- Noise Robustness ---"
    if [ -d "$OUTPUT_DIR/noise_robustness" ] && [ "$(ls -A $OUTPUT_DIR/noise_robustness/*.csv 2>/dev/null)" ]; then
        python scripts/analysis/evaluate_noise_robustness.py \
            --from_tensorboard "$OUTPUT_DIR/tensorboard/aggregated" \
            --output_dir "$OUTPUT_DIR/figures" \
            --format pdf \
            $conditions_filter
    else
        echo "  Using TensorBoard data for noise robustness (note: may have metric bug)."
        python scripts/analysis/generate_paper_figures.py \
            --from_tensorboard \
            --data_dir "$OUTPUT_DIR/tensorboard/aggregated" \
            --output_dir "$OUTPUT_DIR/figures" \
            --format pdf \
            --plots noise_robustness \
            $conditions_filter
    fi
    
    # Generate faithfulness gap plots (from Isaac Sim evaluation)
    echo ""
    echo "--- Faithfulness Gap ---"
    local faithfulness_dir="$OUTPUT_DIR/faithfulness_gap"
    local has_faithfulness_data=false
    
    if [ -d "$faithfulness_dir" ]; then
        # Check for any CSV files
        if ls "$faithfulness_dir"/*.csv &>/dev/null; then
            has_faithfulness_data=true
        fi
    fi
    
    if [ "$has_faithfulness_data" = true ]; then
        echo "  Found faithfulness data in: $faithfulness_dir"
        python scripts/analysis/plot_faithfulness_gap.py \
            --data_dir "$faithfulness_dir" \
            --output_dir "$OUTPUT_DIR/figures" \
            --format pdf
        echo "  Faithfulness gap plots generated."
    else
        echo "  WARNING: No faithfulness gap data found in $faithfulness_dir"
        echo "  To generate faithfulness data, run:"
        echo "    ./generate_paper_results.sh faithfulness"
        echo "  This requires Isaac Sim and evaluates real policy rewards."
        echo ""
        echo "  Expected files after running faithfulness stage:"
        echo "    - $faithfulness_dir/real_rewards_all.csv"
        echo "    - $faithfulness_dir/faithfulness_comparison.csv"
    fi
    
    # Generate error vs uncertainty plots (from offline evaluation)
    echo ""
    echo "--- Error vs Uncertainty ---"
    local error_unc_dir="$OUTPUT_DIR/error_vs_uncertainty"
    local has_error_unc_data=false
    
    if [ -d "$error_unc_dir" ]; then
        # Check for any CSV files
        if ls "$error_unc_dir"/*.csv &>/dev/null; then
            has_error_unc_data=true
        fi
    fi
    
    if [ "$has_error_unc_data" = true ]; then
        echo "  Found error vs uncertainty data in: $error_unc_dir"
        python scripts/analysis/plot_error_vs_uncertainty.py \
            --data_dir "$error_unc_dir" \
            --output_dir "$OUTPUT_DIR/figures" \
            --format pdf \
            $conditions_filter
        echo "  Error vs uncertainty plots generated."
    else
        echo "  WARNING: No error vs uncertainty data found in $error_unc_dir"
        echo "  This is generated automatically during the 'evaluate' stage."
        echo "  Run './generate_paper_results.sh evaluate' to generate this data."
    fi
    
    echo ""
    echo "Figures saved to: $OUTPUT_DIR/figures/"
    ls -la "$OUTPUT_DIR/figures/"
}

# =============================================================================
# Main Execution
# =============================================================================
mkdir -p "$OUTPUT_DIR/logs"

case "$STAGE" in
    extract)
        extract_tensorboard
        ;;
    record)
        record_trajectories
        ;;
    evaluate)
        run_evaluations
        ;;
    faithfulness)
        run_faithfulness
        ;;
    plot)
        generate_figures
        ;;
    all)
        extract_tensorboard
        echo ""
        echo "NOTE: Trajectory recording and evaluation require SLURM jobs."
        echo "Run './generate_paper_results.sh record' to submit trajectory recording jobs."
        echo "After jobs complete, run './generate_paper_results.sh evaluate' to submit evaluation job."
        echo "Optionally run './generate_paper_results.sh faithfulness' for faithfulness gap (requires Isaac Sim)."
        echo "Finally, run './generate_paper_results.sh plot' to generate figures."
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: all, extract, record, evaluate, faithfulness, plot"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "Pipeline stage '$STAGE' complete."
echo "=============================================="
