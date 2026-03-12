# Analysis Pipeline Workflow

This document describes the workflow for generating paper figures from trained models.

## Overview

The pipeline is split into two phases:

1. **Data Generation** (requires SLURM / Isaac Sim) - Scripts that evaluate models and output CSVs
2. **Plotting** (local, no GPU) - Scripts that read CSVs and generate figures

This separation allows:
- Running expensive evaluations once on the cluster
- Iterating on plot styling locally without re-running evaluations
- Filtering conditions at plot time (not during data generation)

## Directory Structure

```
results/paper/
├── tensorboard/           # From extract_tensorboard_data.py
│   ├── aggregated/        # Per-condition aggregated CSVs
│   ├── individual/        # Per-run CSVs
│   └── all_runs.csv       # Master CSV with all data
├── trajectories/          # From batch_record_trajectories.py (SLURM)
│   ├── finetune-bs-nopen/       # Organized by condition
│   │   ├── seed42_step5000.pt
│   │   └── seed43_step5000.pt
│   └── finetune-rp-pen025-std/
│       └── ...
├── hallucination_horizon/ # From batch_evaluate.py (SLURM)
│   └── *.csv
├── noise_robustness/      # From batch_evaluate.py (SLURM)
│   └── *.csv
├── error_vs_uncertainty/  # From batch_evaluate.py (SLURM)
│   └── *.csv
├── correlation_by_horizon/ # From batch_evaluate.py (SLURM)
│   └── *.csv
├── faithfulness_gap/      # From evaluate_real_rewards.py (SLURM+Isaac)
│   └── *.csv
└── figures/               # From plot_*.py scripts
    └── *.pdf
```

---

## Phase 1: Data Generation

### Prerequisites

- Trained model checkpoints in `logs/rsl_rl/anymal_d_flat/`
- SLURM cluster access for GPU evaluations
- Isaac Sim environment for real reward evaluations

### Simplified Workflow (Recommended)

The batch scripts automate processing of all runs. Steps 2-3 require SLURM jobs.

```bash
cd robotic_world_model

# Step 1: Extract TensorBoard data (login node - no GPU needed)
python scripts/analysis/extract_tensorboard_data.py \
    --log_dir logs/rsl_rl/anymal_d_flat \
    --output_dir results/paper/tensorboard

# Step 2: Record trajectories for all runs (SLURM - requires Isaac Sim)
sbatch scripts/analysis/slurm_batch_record.sh
# Wait for job to complete before proceeding

# Step 3: Run all evaluations on trajectories (SLURM - requires GPU)
sbatch scripts/analysis/slurm_batch_evaluate.sh
# Wait for job to complete before proceeding

# Step 4: Generate all plots (login node - no GPU needed)
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --output_dir results/paper/figures
```

**Customizing SLURM jobs:**

```bash
# Custom trajectory recording parameters
OUTPUT_DIR=results/paper/trajectories \
NUM_TRAJECTORIES=200 \
TRAJECTORY_LENGTH=300 \
sbatch scripts/analysis/slurm_batch_record.sh

# Custom evaluation parameters
TRAJECTORY_DIR=results/paper/trajectories \
OUTPUT_DIR=results/paper \
HORIZON=100 \
EVALUATIONS="hallucination_horizon noise_robustness correlation_by_horizon" \
sbatch scripts/analysis/slurm_batch_evaluate.sh
```

**Monitoring jobs:**

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/slurm/batch-record-<jobid>.out
tail -f logs/slurm/batch-eval-<jobid>.out
```

---

### Detailed Steps

### Step 1: Extract TensorBoard Data (Local)

This extracts training metrics from TensorBoard logs into CSVs. By default, only `finetune` runs are extracted (pretrain runs are excluded).

```bash
cd scripts/analysis

# Extract finetune runs (default)
python extract_tensorboard_data.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --output_dir ../../results/paper/tensorboard

# Include all runs (including pretrain)
python extract_tensorboard_data.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --output_dir ../../results/paper/tensorboard \
    --filter ""

# Extract specific metrics only
python extract_tensorboard_data.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --output_dir ../../results/paper/tensorboard \
    --metrics "Train/mean_reward" "System_Dynamics/autoregressive_error"

# List available metrics
python extract_tensorboard_data.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --list_metrics

# For runs with non-standard names, use --runs with explicit condition name
python extract_tensorboard_data.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --output_dir ../../results/paper/tensorboard \
    --runs "2026-01-25_run1" "2026-01-25_run2" \
    --condition_name "baseline"
```

Output:
- `results/paper/tensorboard/all_runs.csv` - Master CSV with all data
- `results/paper/tensorboard/aggregated/*.csv` - Per-condition aggregated data
- `results/paper/tensorboard/individual/*.csv` - Per-run data

### Step 2: Record Trajectories (SLURM)

Records policy trajectories for offline evaluation. Requires Isaac Sim container.

**Batch Mode via SLURM (Recommended):**

```bash
# Submit SLURM job to record all trajectories
sbatch scripts/analysis/slurm_batch_record.sh

# With custom parameters
OUTPUT_DIR=results/paper/trajectories \
NUM_TRAJECTORIES=200 \
TRAJECTORY_LENGTH=300 \
SKIP_EXISTING=true \
sbatch scripts/analysis/slurm_batch_record.sh
```

The job runs `batch_record_trajectories.py` inside the Isaac Sim container, which:
- Auto-discovers all finetune runs in `logs/rsl_rl/anymal_d_flat/`
- Records trajectories for each run's latest checkpoint
- Saves to `results/paper/trajectories/<condition>/seed<N>_step<M>.pt`

**Dry run (preview on login node):**

```bash
# See what would be recorded without running
python scripts/analysis/batch_record_trajectories.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --log_dir logs/rsl_rl/anymal_d_flat \
    --output_dir results/paper/trajectories \
    --dry_run
```

Output structure:
```
results/paper/trajectories/
├── finetune-bs-nopen/
│   ├── seed42_step5000.pt
│   └── seed43_step5000.pt
├── finetune-rp-pen025-std/
│   └── ...
```

**Single Run Mode (for individual SLURM jobs):**

```bash
# Submit single trajectory recording
sbatch slurm/slurm_record_trajectories_parametric.sh \
    logs/rsl_rl/anymal_d_flat/run/model_5000.pt \
    Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    results/paper/trajectories/condition/seed42_step5000.pt \
    100 200 64 42
```

### Steps 3-5: Run Evaluations (SLURM)

The `batch_evaluate.py` script runs all evaluations on recorded trajectories. Requires GPU but NOT Isaac Sim.

**Batch Mode via SLURM (Recommended):**

```bash
# Submit SLURM job for all evaluations
sbatch scripts/analysis/slurm_batch_evaluate.sh

# With custom parameters
TRAJECTORY_DIR=results/paper/trajectories \
OUTPUT_DIR=results/paper \
HORIZON=100 \
EVALUATIONS="hallucination_horizon noise_robustness error_vs_uncertainty" \
sbatch scripts/analysis/slurm_batch_evaluate.sh
```

The job runs `batch_evaluate.py` which:
- Auto-discovers trajectory files in `results/paper/trajectories/`
- Matches each trajectory to its corresponding checkpoint
- Runs all requested evaluations (hallucination horizon, noise robustness, error vs uncertainty, correlation by horizon)
- Saves combined CSVs per evaluation type

**Dry run (preview on login node):**

```bash
# See what would be evaluated without running
python scripts/analysis/batch_evaluate.py \
    --log_dir logs/rsl_rl/anymal_d_flat \
    --trajectory_dir results/paper/trajectories \
    --output_dir results/paper \
    --dry_run
```

Output:
- `results/paper/hallucination_horizon/hallucination_horizon_all.csv`
- `results/paper/noise_robustness/noise_robustness_all.csv`
- `results/paper/error_vs_uncertainty/error_vs_uncertainty_all.csv`
- `results/paper/correlation_by_horizon/correlation_by_horizon_all.csv`
- Per-condition CSVs: `*_finetune-bs-nopen.csv`, etc.

---

**Single Run Mode (for advanced use):**

For running individual evaluations, use the single-run scripts directly:

### Evaluate Hallucination Horizon (Single Run)

Evaluates per-step prediction error over rollout horizon. By default, only `finetune` runs are evaluated.

```bash
python evaluate_hallucination_horizon.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --trajectory_data ../../results/paper/trajectories/traj.pt \
    --output_dir ../../results/paper/hallucination_horizon \
    --horizon 100
```

Output:
- `results/paper/hallucination_horizon/hallucination_horizon_all.csv`
- `results/paper/hallucination_horizon/hallucination_horizon_aggregated.csv`

### Evaluate Noise Robustness (Single Run)

Evaluates prediction error under input noise (OOD). By default, only `finetune` runs are evaluated.

```bash
python evaluate_noise_robustness.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --trajectory_data ../../results/paper/trajectories/traj.pt \
    --output_dir ../../results/paper/noise_robustness \
    --noise_levels 0.0 0.1 0.2 0.4 0.5 0.8
```

Or extract from TensorBoard (note: has known bug comparing to noised targets):

```bash
python evaluate_noise_robustness.py \
    --from_tensorboard ../../results/paper/tensorboard/aggregated \
    --output_dir ../../results/paper/noise_robustness
```

Output:
- `results/paper/noise_robustness/noise_robustness_all.csv`
- `results/paper/noise_robustness/noise_robustness_aggregated.csv`

### Evaluate Error vs Uncertainty (Single Run)

Evaluates correlation between prediction error and epistemic uncertainty. By default, only `finetune` runs are evaluated.

```bash
python evaluate_error_vs_uncertainty.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --trajectory_data ../../results/paper/trajectories/traj.pt \
    --output_dir ../../results/paper/error_vs_uncertainty \
    --horizon 100 \
    --uncertainty_metric std
```

Output:
- `results/paper/error_vs_uncertainty/error_vs_uncertainty_all.csv`
- `results/paper/error_vs_uncertainty/error_vs_uncertainty_aggregated.csv`

### Evaluate Correlation By Horizon (Single Run)

Evaluates how the correlation between prediction error and epistemic uncertainty changes across prediction steps. By default, only `finetune` runs are evaluated.

```bash
python evaluate_correlation_by_horizon.py \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --trajectory_data ../../results/paper/trajectories/traj.pt \
    --output_dir ../../results/paper/correlation_by_horizon \
    --horizon 100
```

Output:
- `results/paper/correlation_by_horizon/correlation_by_horizon_all.csv`
- `results/paper/correlation_by_horizon/correlation_by_horizon_aggregated.csv`

### Evaluate Real Rewards (Single Run - Isaac Sim)

Evaluates policy in actual simulator to compute faithfulness gap. By default, only `finetune` runs are evaluated.

```bash
python evaluate_real_rewards.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --log_dir ../../logs/rsl_rl/anymal_d_flat \
    --imagination_data_dir ../../results/paper/tensorboard/aggregated \
    --output_dir ../../results/paper/faithfulness_gap \
    --num_envs 64 \
    --num_episodes 100
```

Output:
- `results/paper/faithfulness_gap/real_rewards_all.csv`
- `results/paper/faithfulness_gap/faithfulness_comparison.csv`

---

## Phase 2: Plotting

All plotting scripts can be run from the login node (no GPU required).

### Quick Start: Generate All Figures

Use `plot_all.py` to generate all figures with a single command:

```bash
cd robotic_world_model

# Generate all figures with default settings
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --output_dir results/paper/figures

# Generate specific plots only
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --output_dir results/paper/figures \
    --training_curves --hallucination_horizon

# Filter to specific conditions
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --output_dir results/paper/figures \
    --conditions bs-nopen,rp-pen025-std

# Exclude conditions
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --output_dir results/paper/figures \
    --exclude bs-pen025

# List available conditions
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --list_conditions

# Dry run (show what would be done)
python scripts/analysis/plot_all.py \
    --results_dir results/paper \
    --dry_run
```

### Using a Config File

For complex configurations, use a YAML config file:

```bash
# Copy the example config
cp plot_config.example.yaml plot_config.yaml

# Edit to customize colors, conditions, smoothing, etc.
vim plot_config.yaml

# Run with config
python plot_all.py --config plot_config.yaml
```

See `plot_config.example.yaml` for all available options including:
- Custom colors per condition
- Custom labels per condition
- Condition include/exclude lists
- Smoothing parameters
- Per-plot settings

---

### Individual Plot Scripts

All plot scripts follow a consistent interface:

```bash
python scripts/analysis/plot_<analysis>.py \
    --data_dir <path_to_csvs> \
    --output_dir <path_for_figures> \
    --format pdf \
    --conditions <comma-separated list>  # optional filter
```

### Plot Training Curves

```bash
python scripts/analysis/plot_training_curves.py \
    --data_dir results/paper/tensorboard/aggregated \
    --output_dir results/paper/figures/training_curves \
    --format pdf \
    --smooth_window 20 \
    --autoregressive_smooth_window 5
```

Outputs:
- `mean_reward.pdf` - Policy learning curves
- `epistemic_uncertainty.pdf` - Model uncertainty over training
- `autoregressive_error.pdf` - Prediction error over training
- `training_curves_combined.pdf` - All three in one figure

### Plot Hallucination Horizon

```bash
python scripts/analysis/plot_hallucination_horizon.py \
    --data_dir results/paper/hallucination_horizon \
    --output_dir results/paper/figures \
    --format pdf
```

### Plot Noise Robustness

```bash
python scripts/analysis/plot_noise_robustness.py \
    --data_dir results/paper/noise_robustness \
    --output_dir results/paper/figures \
    --format pdf
```

### Plot Error vs Uncertainty

```bash
python scripts/analysis/plot_error_vs_uncertainty.py \
    --data_dir results/paper/error_vs_uncertainty \
    --output_dir results/paper/figures \
    --format pdf
```

### Plot Correlation By Horizon

```bash
python scripts/analysis/plot_correlation_by_horizon.py \
    --data_dir results/paper/correlation_by_horizon \
    --output_dir results/paper/figures \
    --format pdf
```

### Plot Faithfulness Gap

```bash
python scripts/analysis/plot_faithfulness_gap.py \
    --data_dir results/paper/faithfulness_gap \
    --output_dir results/paper/figures \
    --format pdf
```

---

## Filtering Conditions

All plot scripts support filtering via the `--conditions` flag:

```bash
# Plot only specific conditions
python scripts/analysis/plot_hallucination_horizon.py \
    --data_dir results/paper/hallucination_horizon \
    --output_dir results/paper/figures \
    --conditions bs-nopen,rp-pen025-std

# Partial matching works too
python scripts/analysis/plot_noise_robustness.py \
    --data_dir results/paper/noise_robustness \
    --output_dir results/paper/figures \
    --conditions rp-pen  # matches all conditions containing "rp-pen"
```

---

## Shared Utilities

### utils.py

Contains shared data loading and parsing utilities:
- `parse_run_name()` - Parse run directory names to extract metadata
- `load_checkpoint_and_config()` - Load model checkpoint and config
- `discover_conditions()` - Find all conditions with N+ seeds
- `STATE_COMPONENTS` / `VELOCITY_INDICES` - AnymalD state structure

### plot_utils.py

Contains shared plotting utilities:
- `setup_publication_style()` - Configure matplotlib for paper figures
- `ColorManager` - Consistent color assignment across plots
- `get_color()` / `get_label()` - Get color and label for a condition
- `PAPER_CONDITION_COLORS` - Fixed color assignments for paper conditions

---

## Quick Reference

| Script | Phase | Requires | Output |
|--------|-------|----------|--------|
| `extract_tensorboard_data.py` | Data | Login node | CSVs |
| `slurm_batch_record.sh` | Data | SLURM + Isaac Sim | .pt files (all runs) |
| `slurm_batch_evaluate.sh` | Data | SLURM + GPU | CSVs (all evaluations) |
| `batch_record_trajectories.py` | Data | Isaac Sim container | Called by slurm_batch_record.sh |
| `batch_evaluate.py` | Data | GPU | Called by slurm_batch_evaluate.sh |
| `record_trajectories.py` | Data | Isaac Sim | .pt files (single run) |
| `evaluate_hallucination_horizon.py` | Data | GPU | CSVs (single run) |
| `evaluate_noise_robustness.py` | Data | GPU | CSVs (single run) |
| `evaluate_error_vs_uncertainty.py` | Data | GPU | CSVs (single run) |
| `evaluate_correlation_by_horizon.py` | Data | GPU | CSVs (single run) |
| `evaluate_real_rewards.py` | Data | Isaac Sim | CSVs |
| `plot_all.py` | Plot | Login node | All Figures |
| `plot_training_curves.py` | Plot | Login node | Figures |
| `plot_hallucination_horizon.py` | Plot | Login node | Figures |
| `plot_noise_robustness.py` | Plot | Login node | Figures |
| `plot_error_vs_uncertainty.py` | Plot | Login node | Figures |
| `plot_correlation_by_horizon.py` | Plot | Login node | Figures |
| `plot_faithfulness_gap.py` | Plot | Login node | Figures |

---

## Condition Naming Convention

Run directories are parsed to extract experimental conditions. By default, only `finetune` runs are processed. The expected format is:

```
<timestamp>_finetune-<method>-<options>-seed<N>
```

Examples:
- `2026-02-02_23-56-23_finetune-bs-pen025-seed42`
- `2026-02-03_15-16-13_finetune-rp-pen025-std-seed42`
- `2026-02-03_15-16-13_finetune-bs-nopen-seed42`

This generates condition keys like:
- `finetune-bs-pen0.25` - Bootstrap with 0.25 penalty
- `finetune-bs-nopen` - Bootstrap without penalty (baseline)
- `finetune-rp-pen0.25-std` - Random priors with 0.25 penalty, std uncertainty

Note: The `finetune-` prefix is included in condition keys to distinguish from any pretrain runs if they were to be included.
