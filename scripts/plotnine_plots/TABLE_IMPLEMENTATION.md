# Summary Table Implementation — Complete Summary

## Overview

This document summarizes all changes made to implement **summary table generation** in addition to the previous improvements (AUROC plots, risk-coverage, smoothing).

---

## 1. Summary Table Feature

### What Was Implemented

A **summary metrics table** generator that creates compact CSV files aggregating the most important evaluation metrics per condition.

**Output**: `tables/summary_metrics__task-{TASK}__{MODE}__hgt{FILTER}.csv`

**Rows**: One row per condition (B0, Bp, R0, Rp, etc.)

**Columns**: Metrics across 5 categories:
- **A) Long-horizon performance**: horizon_auc, hallucination_horizon
- **B) Robustness**: noise_auc
- **C) Uncertainty decision utility**: auroc_mean, risk_at_coverage
- **D) Correlation diagnostics**: pearson, spearman correlations
- **E) Training**: steady_state_reward, final errors

---

## 2. New Files Created

### `tables.py`
**Location**: `plotnine_plots/tables.py`

**Key Functions**:

#### `make_summary_table(inv, output_dir, summary, plot_config, ...)`
**Main entry point** for table generation.

**Parameters**:
- `inv`: Inventory object (from CSV discovery)
- `output_dir`: Output directory for tables
- `summary`: List to append progress messages
- `plot_config`: Configuration dict with all settings
- `hall_df`, `eu_df`, `corr_df`, `noise_df`, `auroc_df`, `risk_df`, `hall_per_ep`, `hall_distrib`, `tb_dict`, `faith_df`: DataFrames for each source

**Returns**: List of generated table file paths

**Process**:
1. Extracts configuration (task, horizon_filter, coverage_target, mode, etc.)
2. Computes metrics for each mode (vel/full or both)
3. Calls helper functions to compute specific metrics
4. Aggregates across seeds
5. Saves CSV files

---

### Helper Functions in `tables.py`

#### `compute_noise_auc(df, error_col)`
Computes AUC under error vs noise curve using trapezoidal rule.

**Returns**: `{'noise_auc_mean': float, 'noise_auc_std': float}`

---

#### `compute_horizon_auc(df, horizon_auc_file, inv)`
Extracts horizon AUC from `horizon_auc/` CSV files.

**Returns**: `{'horizon_auc_rel_mean': float, 'horizon_auc_rel_std': float}`

---

#### `select_checkpoint_data(df, policy, checkpoint_step, step_col)`
Selects data based on checkpoint policy:
- `'last'`: Last checkpoint per (condition, seed)
- `'specific'`: Specific checkpoint_step

**Returns**: Filtered DataFrame

---

#### `compute_training_metrics(tb_dict, tail_window)`
Computes training metrics from TensorBoard data:
- `steady_state_reward_mean`: Mean reward in last N steps
- `final_autoreg_error`: Mean AR error in last N steps
- `final_epistemic_uncertainty`: Mean uncertainty in last N steps

**Returns**: Dict of metrics per condition

---

#### `compute_auroc_metrics(auroc_df, horizon_filter, summary_list)`
Computes AUROC metrics filtered by horizon:
- `auroc_mean_hgt{X}_mean`: Mean AUROC over h > X
- `auroc_mean_hgt{X}_std`: Std of AUROC
- `auroc_min_hgt{X}_mean`: Minimum AUROC over h > X

**Fairness**: AUROC computed per-horizon, then averaged.

---

#### `compute_risk_metrics(risk_df, horizon_filter, coverage_target, summary_list)`
Computes risk metrics at different coverage levels:
- `risk_at_{80}cov_hgt{X}_mean`: Risk at 80% coverage
- `risk_at_{100}cov_hgt{X}_mean`: Risk at 100% coverage (no filtering)

**Process**:
1. Filter by horizon > X
2. For each condition, find risk at coverage_target
3. Compute mean and std across seeds

---

#### `compute_correlation_metrics(corr_df, horizon_filter, summary_list)`
Computes correlation metrics filtered by horizon:
- `pearson_mean_hgt{X}_mean`: Mean Pearson r over h > X
- `spearman_mean_hgt{X}_mean`: Mean Spearman r over h > X

---

#### `compute_hallucination_metrics(hall_distrib, hall_per_ep, summary_list)`
Computes hallucination horizon metrics.

**Fairness check**:
- Checks if `threshold_method` contains "shared"
- If not shared, returns NaN for hallucination metrics
- If shared, computes median and IQR

**Returns**: `{'hallucination_horizon_median': float, 'hallucination_horizon_iqr': float, 'hallucination_threshold_shared': bool}`

---

## 3. CLI Flags Added

### New Arguments in `__main__.py`

```bash
--make-tables              # Generate summary tables (default: True)
--no-make-tables           # Disable table generation
--table-checkpoint-policy {last,specific}  # Checkpoint selection
--checkpoint-step <int>     # Specific checkpoint (if policy=specific)
--training-tail-window <int>  # Trailing steps for training metrics (default: 500)
--coverage-target <float>    # Coverage target for risk metrics (default: 0.80)
--mode {vel,full,both}    # Error mode for tables (default: both)
```

### Updated Arguments

Previous arguments remain unchanged:
- `--task {anymal,franka}`
- `--horizon-filter <int>`
- `--smooth {none,rolling_mean,ewma}`
- `--smooth-window <int>`
- `--ewma-alpha <float>`
- `--top-q <float>`
- `--no-auroc-summary`

---

## 4. Configuration Integration

### `plot_config` Dict Extended

All plotting functions now receive additional table-related keys:

```python
{
    # Existing keys
    "smooth_mode": "none" | "rolling_mean" | "ewma",
    "smooth_window": int | None,
    "ewma_alpha": float,
    "task": "anymal" | "franka",
    "horizon_filter": int,
    "top_q": float | None,
    "no_auroc_summary": bool,

    # New keys for tables
    "table_mode": "vel" | "full" | "both",
    "table_checkpoint_policy": "last" | "specific",
    "checkpoint_step": int | None,
    "training_tail_window": int,
    "coverage_target": float,
}
```

---

## 5. Run Summary Updates

### New Section in `run_summary.txt`

```
--- CONFIGURATION ---
task:                anymal
horizon_filter:       h > 30
top_q:               default (from data)
smooth:              none
smooth_window:       auto
ewma_alpha:          0.1
auroc_summary:       enabled
make_tables:         True
  table_mode:        both
  checkpoint_policy: last
  checkpoint_step:   last
  coverage_target:   0.8
```

### Summary Tables Section

```
--- SUMMARY TABLES ---
  task: anymal
  horizon_filter: h > 30
  coverage_target: 0.8
  table_mode: both

  A) Long-horizon performance:
  B) Robustness:
  C) Uncertainty decision utility:
    AUROC metrics: filtered to h > 30
    Risk metrics: filtered to h > 30, coverage target = 0.8
  D) Correlation diagnostics:
    Correlation metrics: filtered to h > 30
  E) Training metrics:

  Found 6 conditions
  Saved: summary_metrics__task-anymal__hgt30__full.csv
  Saved: summary_metrics__task-anymal__hgt30__vel.csv

Generated 2 summary table(s)
```

---

## 6. Fairness Requirements Implementation

### 1. Shared Threshold for Hallucination Horizon

**Requirement**: Hallucination horizon metrics only valid if shared threshold is used.

**Implementation**:
```python
# Check if shared threshold was used
threshold_shared = False
if "threshold_method" in hall_distrib.columns:
    if hall_distrib["threshold_method"].str.contains("shared", na=False).any():
        threshold_shared = True

if not threshold_shared:
    # Set metrics to NaN and add boolean flag
    results["hallucination_horizon_median"] = np.nan
    results["hallucination_threshold_shared"] = False
```

**Table includes**: `hallucination_threshold_shared` boolean column

---

### 2. Per-Horizon AUROC

**Requirement**: AUROC computed per-horizon, not pooled.

**Implementation**:
```python
# Filter by horizon > X
filtered = auroc_df[auroc_df["horizon_t"] > horizon_filter].copy()

# Compute mean AUROC across horizons
grouped = filtered.groupby("condition")["auroc"].agg(["mean", "std"])
```

**Column naming**: `auroc_mean_hgt{X}_mean` explicitly includes filter

---

### 3. Horizon-Filtered Metrics

**Requirement**: Explicitly record horizon filter in column names and filenames.

**Implementation**:
- Column names: `auroc_mean_hgt30`, `risk_at_80cov_hgt30`, `pearson_mean_hgt30`
- Filename: `summary_metrics__task-anymal__hgt30__vel.csv`
- Summary logs: "filtered to h > 30"

---

## 7. Table Aggregation Rules

### Mean ± Standard Deviation
**Used for**: AUROC, risk, noise AUC, correlations, training metrics

```python
grouped = df.groupby("condition")["metric"].agg(["mean", "std"])
results["metric_mean"] = grouped["mean"]
results["metric_std"] = grouped["std"]
```

---

### Median + IQR
**Used for**: Hallucination horizon (skewed distribution)

```python
grouped = df.groupby("condition")["hallucination_horizon"].agg([
    "median",
    lambda x: np.percentile(x, 75) - np.percentile(x, 25)
])
grouped.columns = ["median", "iqr"]
```

---

## 8. Output Naming Convention

### Filename Format
```
summary_metrics__task-{TASK}__{MODE}__hgt{FILTER}__{MODE}.csv
```

### Components
- `task-{TASK}`: Task type (anymal or franka)
- `{MODE}`: Error mode (vel or full)
- `hgt{FILTER}`: Horizon filter (e.g., hgt30 = horizons > 30)

### Examples
```
tables/summary_metrics__task-anymal__hgt30__full.csv
tables/summary_metrics__task-anymal__hgt30__vel.csv
tables/summary_metrics__task-franka__hgt10__full.csv
```

---

## 9. Documentation

### New File: `TABLE_METRICS.md`

Comprehensive documentation of:
- **Metric definitions**: What each metric measures
- **Why included**: Why the metric is important
- **Interpretation**: How to read the values
- **Computation**: How the metric is calculated
- **Fairness requirements**: How they're enforced
- **Usage examples**: CLI commands for different scenarios
- **Troubleshooting**: Common issues and solutions
- **Paper writing notes**: Guidance for writing about these metrics

---

## 10. Usage Examples

### Generate tables with default settings
```bash
python -m plotnine_plots --results_dir results/paper --make-tables
```

**Output**: `tables/summary_metrics__task-anymal__hgt30__full.csv` and `summary_metrics__task-anymal__hgt30__vel.csv`

---

### Generate tables with custom horizon filter
```bash
python -m plotnine_plots --results_dir results/paper --horizon-filter 50 --make-tables
```

**Output**: `tables/summary_metrics__task-anymal__hgt50__full.csv`

---

### Generate tables for velocity mode only
```bash
python -m plotnine_plots --results_dir results/paper --mode vel --make-tables
```

**Output**: `tables/summary_metrics__task-anymal__hgt30__vel.csv`

---

### Generate tables with specific coverage target
```bash
python -m plotnine_plots --results_dir results/paper --coverage-target 0.90 --make-tables
```

**Metrics**: `risk_at_90cov_hgt30` instead of default `risk_at_80cov_hgt30`

---

### Disable table generation
```bash
python -m plotnine_plots --results_dir results/paper --no-make-tables
```

---

## 11. Integration with Existing Features

### Plots + Tables Together

Running with all features enabled:

```bash
python -m plotnine_plots \
    --results_dir results/paper \
    --smooth rolling_mean \
    --smooth-window 50 \
    --make-tables
```

**Generates**:
- All plots with smoothing applied
- Summary tables
- Descriptive filenames throughout

---

### Summary in `run_summary.txt`

The summary file now includes:

```
--- PLOTS ---
plot_training: 4 plots saved
plot_long_horizon: 3 plots saved
plot_noise: 1 plots saved
plot_decision: 2 plots saved
plot_correlations: 2 plots saved
plot_reliability: 3 plots saved

--- SUMMARY TABLES ---
  task: anymal
  horizon_filter: h > 30
  coverage_target: 0.8
  table_mode: both
  ...
  Generated 2 summary table(s)

--- SUMMARY ---
Total plots saved: 15
Elapsed: 125.4s
```

---

## 12. Metric Categories Explained

### A) Long-Horizon Performance

**What it measures**: How well the model predicts over extended horizons.

**Metrics**:
- `horizon_auc_rel`: Area under error vs horizon curve (lower = better)
- `hallucination_horizon_median`: Median steps before hallucination (higher = better)
- `hallucination_horizon_iqr`: Variability in hallucination horizons

**Why important**:
- Primary reliability metric for world models
- Shows if model can be used for long-term planning

---

### B) Robustness

**What it measures**: Sensitivity to input noise/perturbations.

**Metrics**:
- `noise_auc`: Area under error vs noise curve (lower = better)

**Why important**:
- Real-world inputs are noisy
- Robust models can handle imperfect observations

---

### C) Uncertainty Decision Utility

**What it measures**: How well epistemic uncertainty can be used for selective prediction.

**Metrics**:
- `auroc_mean_hgt{X}`: AUROC over horizons > X (higher = better)
- `auroc_min_hgt{X}`: Worst-case AUROC (higher = better)
- `risk_at_{80}cov_hgt{X}`: Error at 80% coverage (lower = better)
- `risk_at_{100}cov_hgt{X}`: Error with no filtering (lower = better)

**Why important**:
- Enables safe deployment via abstention
- High AUROC + low risk at 80% = useful uncertainty signal

---

### D) Correlation Diagnostics

**What it measures**: Statistical relationship between uncertainty and error.

**Metrics**:
- `pearson_mean_hgt{X}`: Linear correlation (higher positive = better)
- `spearman_mean_hgt{X}`: Monotonic correlation (higher positive = better)

**Why important**:
- Helps understand if uncertainty is calibrated
- Complements AUROC with continuous measure

---

### E) Training

**What it measures**: Final performance and convergence metrics.

**Metrics**:
- `steady_state_reward_mean`: Policy reward (higher = better)
- `final_autoreg_error`: World model error (lower = better)
- `final_epistemic_uncertainty`: Learned uncertainty level

**Why important**:
- Shows final model quality
- Helps diagnose training issues

---

## 13. Key Design Decisions

### Why Both vel and Full Modes?

Different error metrics capture different aspects:
- **Velocity-only**: Motion error (relevant for control)
- **Full-state**: Complete state error (including position/orientation)

Both can be useful depending on application.

---

### Why Horizon Filtering?

Early horizons (h < 30) may be easy and not representative.
Filtering focuses on **steady-state** prediction quality.

---

### Why Shared Threshold for Hallucination?

Per-condition thresholds would make cross-condition comparison meaningless:
- Different τ → different "hallucination" definitions
- Can't compare horizons fairly

Shared τ from baseline (B0) ensures fair comparison.

---

### Why Per-Horizon AUROC?

AUROC varies with horizon (early = easy, late = hard).
Per-horizon computation preserves this information.
Pooling would hide horizon-dependence.

---

## 14. Testing Recommendations

### Test Table Generation

```bash
# Test with real data
python -m plotnine_plots --results_dir results/paper --make-tables

# Verify output
ls tables/
# Should see: summary_metrics__task-anymal__hgt30__full.csv
```

### Check Table Contents

```python
import pandas as pd
df = pd.read_csv('tables/summary_metrics__task-anymal__hgt30__full.csv')
print(df.head())
print(df.columns)
```

### Verify Fairness

```bash
# Check that hallucination_threshold_shared is True
grep "hallucination_threshold_shared" tables/summary_metrics*.csv

# If False, check log for threshold_method
```

---

## 15. Backward Compatibility

### All Existing Features Preserved

- Plotting functions still work with original signatures
- Smoothing is opt-in (default: none)
- Table generation is opt-in (default: enabled)
- All previous CLI flags work unchanged

### No Breaking Changes

- Existing scripts calling plotting functions directly still work
- Optional parameters have sensible defaults
- Missing data is handled gracefully (NaN values)

---

## 16. Future Enhancements

### Possible Additions

1. **PDF rendering**: Render tables as publication-ready PDF images
2. **Multi-task support**: Aggregate across multiple tasks in one table
3. **Statistical tests**: Add significance test results
4. **Confidence intervals**: Bootstrap confidence intervals for metrics
5. **Additional metrics**: Add more specialized metrics as needed

### Current Limitations

1. **No PDF rendering**: CSV only (PDF would require LaTeX/Matplotlib)
2. **Single task per table**: Multiple tasks → multiple tables
3. **No statistical testing**: Only mean/std, no p-values

---

## Summary

### What Was Accomplished

✅ **Summary table generation** implemented
✅ **All 5 metric categories** covered
✅ **Fairness requirements** enforced (shared threshold, per-horizon AUROC)
✅ **CLI flags** added for configuration
✅ **Documentation** complete (TABLE_METRICS.md)
✅ **Integration** with existing features (plots, smoothing)
✅ **Descriptive filenames** (task, mode, horizon_filter)
✅ **Run summary** updated with table information
✅ **Backward compatible** (no breaking changes)

### Key Files Changed/Created

- **Created**: `plotnine_plots/tables.py` (table generation logic)
- **Created**: `plotnine_plots/TABLE_METRICS.md` (comprehensive documentation)
- **Modified**: `plotnine_plots/__main__.py` (CLI flags and table integration)
- **Modified**: All `plot_*.py` files (accept `plot_config` dict)
- **Modified**: `plotnine_plots/transforms.py` (added smoothing functions)

### Deliverables

1. ✅ **Code implementation**: Complete table generation with all metrics
2. ✅ **Documentation**: Detailed explanation of each metric
3. ✅ **CLI integration**: All flags wired into main entry point
4. ✅ **Fairness enforcement**: Shared threshold check, per-horizon AUROC
5. ✅ **Usage examples**: Common scenarios documented

---

## Quick Reference

### Generate Everything (Plots + Tables)
```bash
python -m plotnine_plots --results_dir results/paper --make-tables
```

### With Smoothing
```bash
python -m plotnine_plots --results_dir results/paper \
    --smooth rolling_mean --smooth-window 50 --make-tables
```

### Custom Configuration
```bash
python -m plotnine_plots --results_dir results/paper \
    --horizon-filter 50 --coverage-target 0.90 \
    --mode vel --make-tables
```

### Tables Only (No Plots)
```bash
python -m plotnine_plots --results_dir results/paper \
    --make-tables --inventory-only  # Discovery only, then check tables/
# Note: Currently tables are generated after plots
```

---

## Notes for Users

1. **Check run_summary.txt** for table generation status
2. **Verify hallucination_threshold_shared** is True before comparing conditions
3. **Use horizon-filtered metrics** for steady-state analysis (h > 30 for Anymal)
4. **Report both mean ± std** in papers
5. **Cite horizon filter** when reporting metrics (e.g., "over h > 30")
6. **Use TABLE_METRICS.md** as reference for writing about metrics
