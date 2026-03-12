# Summary of Plotting Pipeline Improvements

## Overview
This document summarizes the changes made to the plotting pipeline to improve AUROC plots, risk-coverage plots, and add optional smoothing for noisy curves.

---

## 1. AUROC Plots — Required Changes

### Problems Fixed
- **Single AUROC bar plot was uninformative**: Near-1.0 values hid the horizon-dependence of AUROC.
- **Labeling definition was hidden**: No explanation of how positives were defined.
- **No per-horizon analysis**: AUROC varies significantly with prediction horizon.

### Changes Implemented

#### A) AUROC-vs-Horizon Line Plot (Default)
- **x-axis**: Prediction horizon step (`horizon_t`)
- **y-axis**: AUROC at that horizon
- **One line per condition** with ± std ribbon (if `auroc_std` exists)
- **Horizontal reference line** at AUROC=0.5 (random classifier)
- **Subtitle** explicitly states labeling protocol:
  - "AUROC computed per-horizon (positives = top-q errors within each horizon)"
  - Includes top-q value if specified
  - Includes horizon filter if applied (e.g., "Horizon filter: h > 30")

#### B) AUROC Summary Bar Plot (Optional)
- **Enabled by default**, can be disabled with `--no-auroc-summary` flag
- **Configurable horizon filter**: default h>30 for Anymal, h>10 for Franka
- **Bars with errorbars** across seeds (aggregated over filtered horizons)
- **Descriptive filename**: `decision_utility_auroc_summary__hgt30.pdf`
- **Subtitle**: "Mean AUROC over horizons > {horizon_filter}"

#### C) Caption/Subtitle Requirements
- Explicitly states labeling protocol: positives are top-q errors computed **within each horizon**
- No pooling of horizons when computing labels
- Horizon filter clearly indicated in subtitle and filename

#### D) Sanity Checks
- **Warns if `n_positive < 5`** at any horizon used (AUROC unstable)
- **Warns if only aggregated bars possible** (missing per-horizon data)

---

## 2. Risk–Coverage Plots — Required Changes

### Problems Fixed
- **Unclear interpretation**: What does the x-axis represent?
- **Missing visual aids**: No reference points for common operating points.
- **No horizon filtering**: Can't focus on steady-state predictions.

### Changes Implemented

#### A) Axis Labels and Interpretation
- **x-axis**: `Coverage (fraction kept)` (0..1)
  - Coverage=1: keep all samples (no filtering)
  - Coverage<1: keep only lowest-uncertainty samples
- **y-axis**: `Risk (mean prediction error)`
  - Error measured on retained subset
  - Filtering by lowest epistemic uncertainty (ranking-based)

#### B) Clarifying Subtitle
Explicitly states:
- "Risk = mean prediction error on retained subset"
- "Filtering by lowest epistemic uncertainty (ranking-based)"
- "Horizon filter: h > {filter}" if applied

#### C) Visual Aids
- **Vertical dashed line at coverage=0.8**: Common operating point for selective prediction
- **No-filtering risk**: Risk at coverage=1.0 (baseline) is computed per-condition
  - Can be inferred from the plot (line endpoint at x=1.0)

#### D) Horizon Filtering
- **Supports filtering by horizon** (default: h>30 for Anymal, h>10 for Franka)
- **Included in subtitle and filename**:
  - `decision_utility_risk_coverage__hgt30.pdf`
  - `decision_utility_risk_coverage__hgt30__smooth_rollmean_w10.pdf`

---

## 3. Smoothing — Optional Feature

### Problems Fixed
- **Fragmented/noisy curves**: Training curves, uncertainty curves, correlation curves, horizon curves
- **Poor visualization**: Noise obscures trends

### Changes Implemented

#### A) CLI Flags for Smoothing
```bash
--smooth {none, rolling_mean, ewma}        # Smoothing mode (default: none)
--smooth_window <int>                      # Window size (default varies by plot type)
--ewma-alpha <float>                       # EWMA parameter (default: 0.1)
```

#### B) Smoothing Behavior
- **Applied only to line series (means)**, not to uncertainty ribbons
- **Does not alter raw data files** — only for plotting
- **Users can always turn off smoothing** with `--smooth none`

#### C) Recommended Defaults
- **Training curves**: `rolling_mean` with window ~100 (or auto-based on step spacing)
- **Horizon curves**: `rolling_mean` with window ~5–10
- **Correlation vs horizon**: `rolling_mean` with window ~5–10
- **Noise curves**: **No smoothing** (few points)

#### D) Implementation Details
- Creates smoothed columns per group (condition, seed, etc.)
- For `rolling_mean`: uses pandas `.rolling(window, center=True, min_periods=1).mean()`
- For `ewma`: uses pandas `.ewm(alpha, adjust=False).mean()`
- Boundary handling: NaNs introduced where window extends beyond data, handled by `min_periods=1`

---

## 4. Output Naming — Descriptive Filenames

### Naming Convention
Filenames now reflect:
- **Plot family**: `decision_utility`, `correlations`, `training`, `long_horizon`, `noise_robustness`
- **Mode**: `__vel` or `__full` if applicable
- **Horizon filter**: `__hgt30` if applied
- **Smoothing**: `__smooth_rollmean_w10` or `__smooth_ewma_a0.10` if used

### Examples
```
decision_utility/decision_utility_auroc_vs_horizon__vel__hgt30__smooth_rollmean_w5.pdf
decision_utility/decision_utility_risk_coverage__vel__hgt30.pdf
correlations/correlations_pearson_vs_horizon__smooth_rollmean_w10.pdf
training/reward_vs_step__smooth_rollmean_w100.pdf
long_horizon/error_vs_horizon__vel__smooth_rollmean_w10.pdf
```

---

## 5. Task Defaults

### Anymal (4-legged robot)
- **Horizon filter**: h > 30 (default)
- **Error metric**: Velocity-only relative error (if available)

### Franka (manipulator)
- **Horizon filter**: h > 10 (default)
- **Error metric**: MSE or task-appropriate (avoid unstable relative errors)

---

## 6. Implementation Details

### New CLI Arguments
- `--task {anymal,franka}`: Task type (affects horizon filter and error metric defaults)
- `--horizon-filter <int>`: Minimum horizon for decision utility plots
- `--top-q <float>`: Top-q percentile for AUROC labeling
- `--smooth {none,rolling_mean,ewma}`: Smoothing mode
- `--smooth-window <int>`: Window size for rolling mean
- `--ewma-alpha <float>`: Alpha for EWMA
- `--no-auroc-summary`: Disable AUROC summary bar plot

### Configuration Passed to Plotting Functions
All plotting functions now accept a `plot_config` dict containing:
```python
{
    "smooth_mode": "none" | "rolling_mean" | "ewma",
    "smooth_window": int | None,
    "ewma_alpha": float,
    "task": "anymal" | "franka",
    "horizon_filter": int,
    "top_q": float | None,
    "no_auroc_summary": bool,
}
```

### Updated Functions in `transforms.py`

#### `apply_rolling_smooth(df, value_col, group_cols, window)`
- Applies rolling mean smoothing
- Creates column `{value_col}_smooth`
- NaNs at boundaries handled by `min_periods=1`

#### `apply_ewma_smooth(df, value_col, group_cols, alpha)`
- Applies exponential weighted moving average
- Creates column `{value_col}_smooth`

#### `get_smoothed_column(df, value_col, group_cols, mode, window, alpha)`
- Main entry point for smoothing
- Returns `(df_with_smooth_col, col_name_to_use)`
- Handles "none", "rolling_mean", and "ewma" modes

### Summary Reporting
Run summary now includes:
```
--- CONFIGURATION ---
task:                anymal
horizon_filter:       h > 30
top_q:               default (from data)
smooth:              rolling_mean
smooth_window:       auto
ewma_alpha:          0.1
auroc_summary:       enabled
```

---

## Explanation of Key Concepts

### Why AUROC Must Be Shown vs Horizon

**AUROC is horizon-dependent** because:
1. **Prediction error increases with horizon**: World models accumulate errors over longer predictions
2. **Uncertainty signals change**: Epistemic uncertainty vs. prediction error correlation varies with horizon
3. **Different reliability regimes**: Early horizons may be easy (high AUROC), while later horizons are hard (low AUROC)

A single summary AUROC value hides this variation and can be misleading (e.g., averaging high early-horizon AUROC with low late-horizon AUROC). By plotting AUROC vs horizon, we can:
- Identify horizons where uncertainty is most informative
- Compare how different methods degrade with prediction length
- Make informed decisions about usable prediction horizons

### What Risk-Coverage Represents (Selective Prediction / Abstention)

**Risk-coverage plots visualize selective prediction**:
- **X-axis (coverage)**: What fraction of predictions we keep (after filtering)
- **Y-axis (risk)**: Average error on the kept predictions

**The mechanism**:
1. **Rank samples by uncertainty** (lowest uncertainty first)
2. **Keep top-k% samples** (where k = coverage)
3. **Compute mean error** on the kept subset

**Interpretation**:
- **Ideal**: Low error (risk) even at low coverage → uncertainty is well-calibrated
- **Poor**: High error (risk) even at high coverage → uncertainty doesn't distinguish good/bad predictions
- **Coverage=0.8 line**: Common operating point — we accept 20% abstentions to reduce risk

**Applications**:
- **Safety-critical systems**: Only use predictions when uncertainty is low
- **Resource allocation**: Expensive ground-truth verification only when needed
- **Active learning**: Query ground truth for uncertain predictions

### How Smoothing is Applied Without Changing Underlying Data

**Smoothing is for visualization only**:
1. **Copy of data**: Smoothing functions work on a copy of the DataFrame
2. **New columns created**: Smoothed values stored in `{col}_smooth` columns
3. **Original data preserved**: Raw columns remain unchanged
4. **Plotting uses smoothed**: Only the visual plot uses smoothed values
5. **No file modification**: Source CSV files are never altered

**Benefits**:
- **Reduces visual noise**: Makes trends easier to see
- **Improves comparability**: Easier to compare lines across conditions
- **Non-destructive**: Original data available for analysis
- **User control**: Always can disable smoothing with `--smooth none`

---

## File Changes Summary

### Modified Files
1. `__main__.py`: Added CLI arguments for smoothing, task config, horizon filter
2. `transforms.py`: Added smoothing functions (`apply_rolling_smooth`, `apply_ewma_smooth`, `get_smoothed_column`)
3. `plot_decision.py`: Complete rewrite of AUROC and risk-coverage plots with new requirements
4. `plot_correlations.py`: Added smoothing support, improved filename generation
5. `plot_long_horizon.py`: Added smoothing support to error and uncertainty plots
6. `plot_training.py`: Added smoothing support to all training curves
7. `plot_noise.py`: Updated to accept plot_config (no smoothing for noise plots)
8. `plot_faithfulness.py`: Added smoothing support to faithfulness gap plot
9. `plot_reliability.py`: Updated to accept plot_config

### No Breaking Changes
- All existing functionality preserved
- New CLI arguments have sensible defaults
- Smoothing is opt-in (`--smooth none` by default)
- Summary bar plot for AUROC is enabled by default (can disable)

---

## Testing Recommendations

1. **Test AUROC plots**:
   ```bash
   # Default: AUROC vs horizon line plot + summary bar
   python -m plotnine_plots --results_dir results/paper

   # Disable summary bar
   python -m plotnine_plots --results_dir results/paper --no-auroc-summary
   ```

2. **Test risk-coverage with horizon filter**:
   ```bash
   # Anymal with h>30 filter
   python -m plotnine_plots --results_dir results/paper --task anymal --horizon-filter 30

   # Franka with h>10 filter
   python -m plotnine_plots --results_dir results/franka --task franka --horizon-filter 10
   ```

3. **Test smoothing**:
   ```bash
   # Rolling mean smoothing with custom window
   python -m plotnine_plots --results_dir results/paper --smooth rolling_mean --smooth-window 50

   # EWMA smoothing
   python -m plotnine_plots --results_dir results/paper --smooth ewma --ewma-alpha 0.2
   ```

4. **Verify filenames**: Check output filenames follow the new descriptive convention
5. **Verify summary**: Check run_summary.txt includes new configuration section
