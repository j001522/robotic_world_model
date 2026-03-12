# Metrics and Plots Documentation

This document describes all metrics computed and plots generated in the `robotic_world_model/scripts/analysis` pipeline.

---

## Table of Contents

1. [Evaluation Pipeline Overview](#evaluation-pipeline-overview)
2. [Metrics Definitions](#metrics-definitions)
3. [Plots Generated](#plots-generated)
4. [Configuration](#configuration)

---

## Evaluation Pipeline Overview

The evaluation pipeline consists of the following stages:

### 1. Trajectory Recording (`record_trajectories.py`, `batch_record_trajectories.sh`)
- Records real trajectories from Isaac Sim
- Saves to `results/paper/trajectories/<condition>/seed<N>_step<M>.pt`

### 2. Offline Evaluation (`batch_evaluate.py`, `slurm_batch_evaluate.sh`)
- Runs all evaluations on pre-recorded trajectories
- Generates CSV files for each metric type

### 3. Aggregation (`compute_*.py` scripts)
- Processes raw evaluation data into aggregated statistics
- Computes derived metrics (AUROC, growth rates, etc.)

### 4. Plotting (`plot_*.py` scripts, `plot_all.py`)
- Generates publication-quality figures
- Supports multiple output formats (PDF, PNG, SVG)

---

## Metrics Definitions

### Primary Metrics (Computed per evaluation step)

#### 1. Relative Prediction Error
**Definition:**
```
error_rel = sum(|pred - gt|) / (sum(|gt|) + epsilon)
```
- Computed on selected state components (full-state or velocity-only)
- Used for training optimization
- **Unitless** (normalized)

**Variants:**
- `error_rel_mean`: Mean across all trajectories at this horizon
- `error_rel_std`: Standard deviation across trajectories
- `error_vel_rel_mean`: Velocity-only relative error (v, w, q_dot)
- `error_vel_rel_std`: Std dev of velocity-only error

#### 2. Absolute Prediction Error (MSE)
**Definition:**
```
error_abs = sum((pred - gt)^2)
```
- Squared error, summed across selected dimensions
- **Units:** State space squared

**Variants:**
- `error_abs_mean`: Mean MSE across trajectories
- `error_abs_std`: Std dev across trajectories

#### 3. Epistemic Uncertainty
**Definition:**
```
epistemic_uncertainty = std(ensemble_predictions)
```
- Computed as standard deviation across ensemble members
- Alternative metric: variance (`ensemble_var`)
- **Units:** State space (same scale as error)

**Variants:**
- `uncertainty_mean`: Mean uncertainty across trajectories
- `uncertainty_std`: Std dev of uncertainty across trajectories

---

### Derived Metrics (Computed from primary metrics)

#### 4. Hallucination Horizon
**Definition:**
```
H_halluc = min{h | error(h) > threshold}
```
- First horizon step where error exceeds threshold
- If never exceeds: H_halluc = max_horizon
- **Units:** Steps (1-H)

**Threshold Options:**
- **Percentile-based with shared baseline (recommended for IROS):** 95th percentile of short-horizon error (first 5 steps) computed from a **baseline condition** (e.g., B0)
  - Applies the **same threshold** to all conditions
  - Fair comparison: measures reliability against shared baseline
- **Percentile-based per condition (default):** 95th percentile of each condition's own short-horizon error
  - Not comparable across conditions due to different error scales
- **Fixed threshold (deprecated):** Absolute value (e.g., 1.0)

**Interpretation:**
- "When does model become worse than normal short-horizon error?"
- Larger values = better long-horizon reliability
- **Important:** Using a shared baseline condition ensures hallucination horizons are comparable across methods

**Output Columns:**
- `hallucination_horizon`: First exceedance step
- `threshold`: Used error threshold
- `threshold_method`: 'percentile_95' or 'fixed'
- `max_error`: Maximum error observed
- `error_at_hallucination`: Error at H_halluc

---

#### 5. Growth Rates
**Definition:**
```
error_growth_rate = slope(error vs horizon)
uncertainty_growth_rate = slope(uncertainty vs horizon)
```
- Linear regression: `y = slope * x + intercept`
- Computed per rollout (condition, seed, checkpoint)
- **Units:** Error/step or Uncertainty/step

**Fitting Range:**
- Optional: `--horizon_min` and `--horizon_max`
- Default: All horizons

**Output Columns:**
- `error_growth_rate`: Linear slope of error vs horizon
- `error_growth_rate_stderr`: Standard error of slope
- `error_growth_rate_p_value`: Statistical significance
- `error_growth_rate_r_squared`: R² of fit
- `uncertainty_growth_rate`: Linear slope of uncertainty vs horizon
- `uncertainty_growth_rate_stderr`: Standard error
- `uncertainty_growth_rate_p_value`: Statistical significance
- `uncertainty_growth_rate_r_squared`: R² of fit
- `n_samples`: Number of data points used

---

#### 6. AUROC (Area Under ROC Curve)
**Definition:**
```
AUROC = area_under(TPR vs FPR curve)
```
- Measures ability of uncertainty to predict high-error events
- **Range:** [0, 1]
- AUROC = 0.5: Random (uncertainty uninformative)
- AUROC = 1.0: Perfect prediction

**Labeling (Per-Horizon, recommended):**
```
label[h] = 1 if error[h] >= percentile_top_q(errors[h])
         = 0 otherwise
```
- Top q% = positive class (high error)
- Threshold computed **within each horizon** (prevents leakage)

**Labeling (Pooled, deprecated):**
- Threshold computed across all horizons pooled
- Can leak information if error scales with horizon

**Output Columns (per horizon):**
- `auroc`: AUROC score
- `optimal_threshold`: Uncertainty threshold maximizing Youden's J (TPR - FPR)
- `optimal_f1`: F1 score at optimal threshold
- `tpr_at_0.1_fpr`: True positive rate at 10% FPR
- `error_threshold`: Error threshold used for labeling
- `n_positive`: Number of positive samples
- `n_samples`: Total samples

---

#### 7. Risk-Coverage
**Definition:**
```
risk(cov) = mean_error(uncertainty <= quantile(1 - cov))
```
- Risk = Mean prediction error on low-uncertainty samples
- Coverage = Fraction of samples retained
- Lower curve = better uncertainty as reliability signal

**Computation:**
For each uncertainty percentile `p` (0-100):
1. Compute uncertainty threshold: `τ_p = percentile(uncertainties, p)`
2. Filter samples: `{i | uncertainty[i] <= τ_p}`
3. Compute mean error on filtered samples
4. Coverage = `(n_filtered / n_total)`

**Important:**
- **Uncertainty percentiles are computed per condition** (within each method's own uncertainty distribution)
- This evaluates **ranking quality**: "Does each method's uncertainty rank predictions by error?"
- NOT a test of a global calibration policy (would require shared reference distribution)
- **Horizon filtering:** Uses `horizon > 30` by default (same as AUROC) for consistency

**Output Columns:**
- `uncertainty_percentile`: Uncertainty percentile (0-100)
- `uncertainty_threshold`: Uncertainty value at percentile
- `mean_error_filtered`: Mean error of filtered samples
- `n_filtered`: Number of samples retained
- `coverage_fraction`: Fraction of samples retained
- `risk_at_X_cov`: Mean error at X% coverage (e.g., `risk_at_80_coverage`)

---

#### 8. Correlation Metrics
**Definition:**
```
correlation = Pearson_r(error, uncertainty)
```
- Measures linear relationship between error and uncertainty
- Computed per horizon and per rollout
- **Range:** [-1, 1]

**Variants:**
- `correlation_by_horizon`: Correlation at each horizon step
- `error_uncertainty_growth_correlation`: Correlation of growth rates

---

## Plots Generated

### Figure 2: Long-Horizon Behavior (IROS Paper)

#### 2a. Error vs Horizon
**Script:** `plot_fig2_long_horizon.py` → `plot_long_horizon_error()`
**File:** `fig2a_error_vs_horizon.{format}`

**Metrics Used:**
- X-axis: `horizon_t` (or `prediction_step`)
- Y-axis: `error_rel_mean` (or `error_vel_rel_mean`)
- Shading: `error_rel_std` (or `error_vel_rel_std`)

**Features:**
- Line plot with confidence intervals
- Supports velocity-only or full-state
- Starts from horizon=1 (not history_horizon)

**Flags:**
- `--velocity_only` (default: True): Use velocity-only errors
- `--full_state`: Use full-state errors

---

#### 2b. Epistemic Uncertainty vs Horizon
**Script:** `plot_fig2_long_horizon.py` → `plot_long_horizon_uncertainty()`
**File:** `fig2b_uncertainty_vs_horizon.{format}`

**Metrics Used:**
- X-axis: `horizon_t` (or `prediction_step`)
- Y-axis: `uncertainty_mean` (normalized per condition)
- Shading: `uncertainty_std`

**Features:**
- Line plot with confidence intervals
- Normalized uncertainty per condition (robust z-score)
- Enables fair comparison across mechanisms

**Normalization:**
```
uncertainty_norm = (uncertainty - median) / IQR
```

**Important Notes:**
- Per-condition normalization is used to enable **trend comparison** despite different uncertainty scales
- This shows **growth pattern** and **relative change** across horizons
- Absolute uncertainty magnitudes differ between methods (see raw statistics in console output)
- For claims about reliability, use **hallucination horizon** and **AUROC** which are absolute metrics

---

### Figure 3: Hallucination Horizon & Growth Rates (IROS Paper)

#### 3a. Hallucination Horizon Distribution
**Script:** `plot_fig3_hallucination.py` → `plot_hallucination_distribution()`
**File:** `fig3a_hallucination_distribution.{format}`

**Metrics Used:**
- Per-episode: `hallucination_horizon` from `aggregate_hallucination.py`

**Plot Types:**
- **Box plot** (default): Median, quartiles, outliers
- **Violin plot**: Distribution density

**Features:**
- Short labels: B0, Bp, R0, Rp (instead of full names)
- Uses percentile-based threshold (95th percentile)
- **For IROS paper:** Uses shared baseline condition (e.g., bs-nopen) for fair comparison across methods
- **Important:** Run `batch_evaluate.py` with IROS aggregations to get proper per-episode hallucination data

**Flags:**
- `--plot_type`: 'box' (default) or 'violin'
- `--use_short_labels` (default: True): Use B0/Bp/R0/Rp
- `--skip_growth_rates` (default: True): Skip scatter plot

---

#### 3b. Growth Rates Correlation (Optional)
**Script:** `plot_fig3_hallucination.py` → `plot_growth_rates_correlation()`
**File:** `fig3b_growth_rates_correlation.{format}`

**Metrics Used:**
- X-axis: `uncertainty_growth_rate`
- Y-axis: `error_growth_rate`
- Per-point: One rollout

**Features:**
- Scatter plot colored by condition
- Shows correlation between error and uncertainty growth
- Often **not meaningful** for publication

**Note:** This plot is optional and skipped by default. Feedback indicates it makes little sense when pooled across conditions.

---

### Figure 5: Decision Utility - Risk Filtering & AUROC (IROS Paper)

#### 5a. Risk-Coverage Curve
**Script:** `plot_fig5_decision_utility.py` → `plot_risk_coverage()`
**File:** `fig5a_risk_coverage.{format}`

**Metrics Used:**
- X-axis: `uncertainty_percentile` (plotted as "Coverage (kept fraction of lowest-uncertainty predictions)")
- Y-axis: `mean_error_filtered`
- Per-condition curves (one per method)
- Reference lines: No filtering (100%), 80% coverage

**Features:**
- Multiple curves (one per condition)
- Reference lines: No filtering (100%), 80% coverage
- X-axis inverted: High coverage (low filtering) → Low coverage (high filtering)
- Supports velocity-only or full-state
- Horizon filtered: Uses `h > 30` for consistency with AUROC

**Important Notes:**
- **Uncertainty percentiles are computed per condition** (within each method's uncertainty distribution)
- This evaluates **ranking quality**: "Does each method's uncertainty reliably rank higher-error predictions?"
- Different from testing a global calibration policy (which would use a shared reference distribution)
- Lower curve = better uncertainty as a signal of reliability

**Flags:**
- `--velocity_only` (default: True): Use velocity-only errors
- `--horizon_filter` (default: 30): Filter horizons (uses h > 30 for consistency with AUROC)

---

#### 5b. AUROC vs Horizon
**Script:** `plot_fig5_decision_utility.py` → `plot_auroc_vs_horizon()`
**File:** `fig5b_auroc_vs_horizon.{format}`

**Plot Types:**

**Line Plot (recommended):**
- X-axis: `horizon_t`
- Y-axis: `auroc`
- Shading: `auroc_std`
- Per-condition lines

**Bar Plot (aggregated):**
- X-axis: Condition
- Y-axis: `auroc` (mean over h > 30)
- Error bars: `auroc_std`

**Features:**
- Per-horizon AUROC (recommended)
- Shows how uncertainty predictive power varies with horizon
- Filters by horizon (>30 by default)

**Flags:**
- `--auroc_plot_type`: 'line' (default, recommended) or 'bar'
- `--horizon_filter` (default: 30)

---

### Other Plots (Not IROS Paper)

#### Training Curves
**Script:** `plot_training_curves.py`
**File:** `training_curves/.*`

**Metrics:**
- Reward over iterations
- Autoregressive error over iterations

#### Noise Robustness
**Script:** `plot_noise_robustness.py`
**File:** `noise_robustness/.*`

**Metrics:**
- Error vs noise level
- Multiple noise levels: 0.0, 0.1, 0.2, 0.4, 0.5, 0.8

#### Error vs Uncertainty Scatter
**Script:** `plot_error_vs_uncertainty.py`
**File:** `error_vs_uncertainty/.*`

**Metrics:**
- X-axis: `uncertainty_mean`
- Y-axis: `error_rel_mean`
- Colored by horizon

#### Correlation by Horizon
**Script:** `plot_correlation_by_horizon.py`
**File:** `correlation_by_horizon/.*`

**Metrics:**
- X-axis: `horizon_t`
- Y-axis: `correlation`
- Per-condition lines

#### Faithfulness Gap
**Script:** `plot_faithfulness_gap.py`
**File:** `faithfulness_gap/.*`

**Metrics:**
- Uncertainty calibration error

---

## Configuration

### Changing Trajectory Directory and Output Directory

#### Method 1: Command Line Arguments

**In `slurm_batch_evaluate.sh`:**

The script uses environment variables with defaults:

```bash
# Lines 30-31
TRAJECTORY_DIR="${TRAJECTORY_DIR:-results/paper/trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper}"
```

**To change directories, set environment variables before submitting:**

```bash
# Option A: Set inline
TRAJECTORY_DIR=/path/to/your/trajectories \
OUTPUT_DIR=/path/to/your/output \
sbatch scripts/analysis/slurm_batch_evaluate.sh

# Option B: Export first
export TRAJECTORY_DIR=/path/to/your/trajectories
export OUTPUT_DIR=/path/to/your/output
sbatch scripts/analysis/slurm_batch_evaluate.sh
```

**For local execution (without SLURM):**

```bash
python scripts/analysis/batch_evaluate.py \
    --log_dir logs/rsl_rl/anymal_d_flat \
    --trajectory_dir /path/to/your/trajectories \
    --output_dir /path/to/your/output \
    --horizon 200
```

---

#### Method 2: Edit the SLURM Script

Edit `scripts/analysis/slurm_batch_evaluate.sh` lines 30-31:

```bash
# Before:
TRAJECTORY_DIR="${TRAJECTORY_DIR:-results/paper/trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper}"

# After (your custom paths):
TRAJECTORY_DIR="${TRAJECTORY_DIR:-/custom/path/to/trajectories}"
OUTPUT_DIR="${OUTPUT_DIR:-/custom/path/to/output}"
```

---

#### Method 3: Use a Config File (Recommended)

Create a config file (e.g., `batch_eval_config.sh`):

```bash
#!/bin/bash
# Custom evaluation configuration
TRAJECTORY_DIR=/custom/path/to/trajectories
OUTPUT_DIR=/custom/path/to/output
HORIZON=200
EVALUATIONS="hallucination_horizon error_vs_uncertainty"

# Source the main script with your config
sbatch --export=TRAJECTORY_DIR,OUTPUT_DIR,HORIZON,EVALUATIONS \
    scripts/analysis/slurm_batch_evaluate.sh
```

Then run:
```bash
bash batch_eval_config.sh
```

---

### Configuration Files for Plotting

The `plot_all.py` script supports configuration via YAML or command line:

**YAML Config (`plot_config.yaml`):**

```yaml
# Directory configuration
results_dir: /path/to/your/results
output_dir: /path/to/your/figures

# Output format
format: pdf

# Condition filtering
conditions:
  - bs-nopen
  - rp-pen025-std
exclude:
  - bs-pen025

# Per-plot settings
long_horizon:
  velocity_only: true

hallucination_distribution:
  plot_type: box
  use_short_labels: true
  skip_growth_rates: true

decision_utility:
  horizon_filter: 30
  velocity_only: true
  auroc_plot_type: line

# Select which plots to generate
plots:
  long_horizon: true
  hallucination_distribution: true
  decision_utility: true
```

**Then run:**
```bash
python scripts/analysis/plot_all.py --config plot_config.yaml
```

---

### Key Parameters

#### For Evaluations (`batch_evaluate.py`, `slurm_batch_evaluate.sh`)

| Parameter | Flag | Default | Description |
|-----------|-------|----------|-------------|
| Trajectory directory | `--trajectory_dir` | `results/paper/trajectories` | Where trajectory .pt files are stored |
| Output directory | `--output_dir` | `results/paper` | Where to save evaluation results |
| Log directory | `--log_dir` | `logs/rsl_rl/anymal_d_flat` | Where to find checkpoints |
| Horizon | `--horizon` | `200` | Maximum prediction horizon |
| Evaluations | `--evaluations` | All | Which evaluations to run |
| IROS aggregations | `--iros_aggregations` | All | Which aggregations to compute |
| Noise levels | `--noise_levels` | `[0.0, 0.1, ...]` | Noise levels for robustness |

#### For Aggregation Scripts

**`aggregate_hallucination.py`:**
| Parameter | Flag | Default | Description |
|-----------|-------|----------|-------------|
| Threshold percentile | `--threshold_percentile` | `95.0` | Percentile for threshold (recommended) |
| Fixed threshold | `--error_threshold` | `None` | Fixed threshold (deprecated) |
| Baseline horizon | `--baseline_horizon` | `5` | Steps for baseline distribution |
| Baseline condition | `--baseline_condition` | `None` | Condition for shared threshold (e.g., "bs-nopen") |
| Velocity-only | `--velocity_only` | `True` | Use velocity-only errors |

**`compute_auroc.py`:**
| Parameter | Flag | Default | Description |
|-----------|-------|----------|-------------|
| Top percentile | `--top_percentile` | `10.0` | Top q% for positive class |
| Horizon filter | `--horizon_filter` | `30` | Minimum horizon for aggregation |
| Per-horizon | `--per_horizon` | `True` | Compute per-horizon (recommended) |
| Pooled | `--pooled` | `False` | Pool all horizons (deprecated) |

**`compute_risk_coverage.py`:**
| Parameter | Flag | Default | Description |
|-----------|-------|----------|-------------|
| Coverage levels | `--coverage_levels` | `[0.5, 0.6, 0.7, 0.8, 0.9]` | Coverage levels to report |
| Horizon filter | `--horizon_filter` | `30` | Minimum horizon to include (for consistency with AUROC) |

#### For Plotting Scripts (`plot_*.py`, `plot_all.py`)

| Parameter | Flag | Default | Description |
|-----------|-------|----------|-------------|
| Results directory | `--results_dir` | `results/paper` | Where to find evaluation data |
| Output directory | `--output_dir` | `results/paper/figures` | Where to save plots |
| Format | `--format` | `pdf` | Output format (pdf, png, svg) |
| Conditions | `--conditions` | `All` | Comma-separated conditions to include |
| Exclude | `--exclude` | `None` | Comma-separated conditions to exclude |
| Velocity-only | `--velocity_only` | `True` | Use velocity-only errors |

---

## File Structure

### Input (Trajectories)
```
results/paper/trajectories/
├── bs-nopen/
│   ├── seed42_step5000.pt
│   └── seed43_step5000.pt
├── bs-pen025/
│   └── ...
├── rp-nopen/
│   └── ...
└── rp-pen025-std/
    └── ...
```

### Output (Evaluations)
```
results/paper/
├── hallucination_horizon/
│   ├── hallucination_horizon_bs-nopen.csv
│   └── hallucination_horizon_rp-pen025-std.csv
├── error_vs_uncertainty/
│   ├── error_vs_uncertainty_bs-nopen.csv
│   └── error_vs_uncertainty_rp-pen025-std.csv
├── growth_rates/
│   ├── growth_rates_bs-nopen.csv
│   └── growth_rates_all.csv
├── auroc/
│   ├── auroc_bs-nopen.csv
│   └── auroc_all.csv
├── risk_coverage/
│   ├── risk_coverage_bs-nopen.csv
│   └── risk_coverage_all.csv
└── hallucination_stats/
    ├── hallucination_horizon_bs-nopen.csv
    └── hallucination_distribution_all.csv
```

### Output (Plots)
```
results/paper/figures/
├── fig2/
│   ├── fig2a_error_vs_horizon.pdf
│   ├── fig2b_uncertainty_vs_horizon.pdf
│   └── fig2_long_horizon_combined.pdf
├── fig3/
│   ├── fig3a_hallucination_distribution.pdf
│   └── fig3b_growth_rates_correlation.pdf
└── fig5/
    ├── fig5a_risk_coverage.pdf
    └── fig5b_auroc_vs_horizon.pdf
```

---

## Quick Reference: State Components

For Anymal (state_dim = 45):

| Index | Component | Range | Description |
|-------|-----------|--------|-------------|
| 0-2 | v | 0-2 | Base linear velocity |
| 3-5 | w | 3-5 | Base angular velocity |
| 6-8 | gravity | 6-8 | Projected gravity |
| 9-20 | q | 9-20 | Joint positions (12) |
| 21-32 | q_dot | 21-32 | Joint velocities (12) |
| 33-44 | torque | 33-44 | Joint torques (12) |

**Velocity-only indices:** `VELOCITY_INDICES = [0,1,2, 3,4,5, 21,22,23,24,25,26,27,28,29,30,31,32]`

- v: 0-2 (3 dims)
- w: 3-5 (3 dims)
- q_dot: 21-32 (12 dims)
- **Total: 18 dimensions**

---

## Best Practices

### For Anymal Locomotion

1. **Use velocity-only errors** (default)
   - More control-relevant
   - Scale-consistent
   - Better interpretability

2. **Use percentile-based thresholds with shared baseline** (for IROS paper)
   - Prevents arbitrary cutoffs
   - Computes threshold from a **baseline condition** (e.g., B0/bs-nopen)
   - Applies **same threshold** to all conditions for fair comparison
   - Use: `--baseline_condition bs-nopen` in `aggregate_hallucination.py`
   - For offline evaluation without baseline: compute threshold per condition

3. **Normalize uncertainty for trend visualization**
   - Prevents mechanism scale differences from dominating
   - Use robust z-score (median/IQR)
   - **Important:** This shows growth pattern, not absolute scale
   - For absolute comparisons, use hallucination horizon and AUROC

4. **Filter by horizon for AUROC/Risk-Coverage**
   - Use h > 30 (default)
   - Uncertainty becomes more informative at longer horizons

5. **Compute AUROC per-horizon**
   - Prevents information leakage
   - Enables AUROC vs horizon line plots
   - More honest evaluation

### For Publication

1. **Start horizon at 1** in plots (not history_horizon)
   - Avoids confusion about missing steps
   - Makes x-axis interpretable

2. **Use short labels** (B0, Bp, R0, Rp)
   - Prevents cramped x-axis labels
   - Better for multi-panel figures

3. **Drop growth rate scatter plot** (optional)
   - Often not meaningful when pooled
   - AUROC vs horizon and risk-coverage are stronger

### Understanding Metric Comparisons

| Metric | Purpose | Normalization | Fair for Comparison? |
|--------|---------|---------------|---------------------|
| Error vs horizon (Fig 2a) | Shows prediction degradation | Absolute | Yes |
| Uncertainty vs horizon (Fig 2b) | Shows uncertainty growth | Per-condition (for trends) | No (for absolute) |
| Hallucination horizon | Reliability vs baseline | Shared baseline threshold | Yes |
| AUROC vs horizon (Fig 5b) | Uncertainty calibration at each horizon | Per-horizon percentile | Yes |
| Risk-coverage (Fig 5a) | Uncertainty ranking quality | Per-condition percentile | Yes (for ranking) |

**Key Insight:**
- For **trend comparison** (growth, uncertainty scale): Per-condition normalization is acceptable and explicitly labeled
- For **fair reliability claims** (hallucination horizon, AUROC): Use shared baselines and per-horizon metrics

---

## Troubleshooting

### "No trajectory files found"

Check:
1. Trajectory directory path is correct
2. Files exist: `ls -la results/paper/trajectories/`
3. Filenames match pattern: `seed<N>_step<M>.pt`

### "No matched trajectory/checkpoint pairs"

Check:
1. Condition names match between trajectories and checkpoints
2. Checkpoint steps match trajectory steps
3. Both directories have data: `find logs/` and `find results/paper/trajectories/`

### LSP Errors (Type Checking)

- LSP errors in `.py` files are type checking issues
- These do NOT affect runtime execution
- Safe to ignore

### Plots Look Wrong

Check:
1. Data directories are correct: `--results_dir`
2. Velocity-only mode matches evaluation mode
3. Horizon filter is appropriate
4. Try re-running aggregation scripts if data changed

---

## References

- [Evaluation Scripts](scripts/analysis/)
- [Plotting Scripts](scripts/analysis/plot_*.py)
- [SLURM Batch Script](scripts/analysis/slurm_batch_evaluate.sh)
- [Main Plotting Pipeline](scripts/analysis/plot_all.py)
