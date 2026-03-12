# Summary Table Metrics — Definitions and Interpretation

This document explains each metric in the `summary_metrics.csv` table, why it's included, and how to interpret it.

---

## Overview

The summary table aggregates the most important evaluation metrics **per condition** (e.g., B0, Bp, R0, Rp) to enable quick cross-condition comparison. All metrics are computed from the same set of evaluation results, ensuring consistency.

**Table location**: `tables/summary_metrics__task-{TASK}__{MODE}__hgt{FILTER}.csv`

---

## Metric Categories

### A) Long-Horizon Performance

#### `horizon_auc_rel_mean` / `horizon_auc_rel_std`
**Definition**: Area Under the Curve (AUC) of relative error vs prediction horizon.

**Source**: `horizon_auc/` family (horizon_auc_all.csv or individual files)

**Why included**:
- Captures overall prediction quality across all horizons in a single number
- Lower AUC = better predictions (error stays low for longer horizons)
- Enables quick comparison of which methods maintain low error across predictions

**Interpretation**:
- **Lower is better**: AUC near 0 means error remains low throughout
- **Higher is worse**: AUC increases means error grows quickly with horizon
- **Compare across conditions**: Smaller AUC indicates more reliable long-horizon predictions

**Computation**: Trapezoidal integration of `error_rel_mean` vs `horizon_t` curve

---

#### `hallucination_horizon_median` / `hallucination_horizon_iqr`
**Definition**: Median and interquartile range (IQR) of hallucination horizons.

**Source**: `hallucination_stats/` family (hallucination_per_episode_all.csv)

**Fairness requirement**: Only computed if a **shared threshold** τ is used (computed from a single baseline condition). If thresholds are per-condition, this metric is set to NaN.

**Why included**:
- Hallucination horizon is the **primary reliability metric** for world models
- Median gives typical behavior; IQR indicates variability across seeds/episodes
- Direct measure of how far ahead the model can predict before becoming unreliable

**Interpretation**:
- **Higher is better**: Model can predict more steps before hallucinating
- **Median**: Typical hallucination horizon across seeds/episodes
- **IQR**: Spread; small IQR = consistent behavior, large IQR = unstable
- **Compare across conditions**: Higher median + smaller IQR indicates more reliable model

**Computation**:
- First horizon where prediction error > shared threshold τ
- τ computed from baseline condition (e.g., B0) using p95 error in early horizons
- Median and IQR across all seeds/episodes per condition

---

#### `error_at_horizon_{H}` (optional)
**Definition**: Prediction error at a specific prediction horizon H (e.g., H=200).

**Source**: `hallucination_horizon/` family (error curves)

**Why included**:
- Provides concrete error numbers at a specific prediction length
- Useful for setting practical operating points (e.g., "we need 50-step predictions")
- Complements AUC with interpretable point estimates

**Interpretation**:
- **Lower is better**: Less error at that horizon
- **Compare across conditions**: Lower error = better predictions at that horizon

---

### B) Robustness

#### `noise_auc_mean` / `noise_auc_std`
**Definition**: Area Under the Curve (AUC) of error vs noise level.

**Source**: `noise_robustness/` family (noise_robustness_all.csv)

**Why included**:
- Measures **noise sensitivity** — how much predictions degrade under input noise
- Higher noise AUC = model is robust to perturbations
- Important for real-world deployment where inputs are imperfect

**Interpretation**:
- **Lower is better**: Error increases slowly with noise (robust model)
- **Higher is worse**: Error grows quickly with noise (fragile model)
- **Compare across conditions**: Lower AUC indicates better noise robustness

**Computation**: Trapezoidal integration of `error_mean` vs `noise_level` curve

---

#### `error_at_noise_max` (optional)
**Definition**: Prediction error at maximum noise level tested (e.g., noise=0.8).

**Source**: `noise_robustness/` family

**Why included**:
- Shows worst-case error under extreme noise
- Useful for failure mode analysis
- Complements noise AUC with point estimate at high stress

**Interpretation**:
- **Lower is better**: Model handles extreme noise well
- **Compare across conditions**: Lower error = better robustness

---

### C) Uncertainty Decision Utility (Horizon-Filtered)

These metrics evaluate how well **epistemic uncertainty** can be used for selective prediction (abstention).

#### `auroc_mean_hgt{X}_mean` / `auroc_mean_hgt{X}_std`
**Definition**: Mean AUROC over all prediction horizons > X (e.g., h > 30).

**Source**: `auroc/` family (auroc_all.csv)

**Filtering**: Only horizons > X are included (default X=30 for Anymal, X=10 for Franka).

**Why included**:
- Measures **how well uncertainty distinguishes high-error predictions**
- AUROC = 0.5: Uncertainty is useless (random)
- AUROC > 0.5: Higher uncertainty correlates with higher error (informative)
- AUROC ≈ 1.0: Perfect discrimination (uncertainty perfectly predicts error)

**Interpretation**:
- **Higher is better**: Uncertainty is informative about prediction quality
- **> 0.7**: Good uncertainty calibration (useful for selective prediction)
- **0.5-0.7**: Moderate uncertainty information
- **~ 0.5**: Uncertainty not useful (random)
- **Compare across conditions**: Higher AUROC = better uncertainty calibration

**Note**: AUROC is computed **per-horizon** (positives = top-q errors within each horizon), then averaged. This is fairer than pooling horizons.

---

#### `auroc_min_hgt{X}` (optional)
**Definition**: Minimum AUROC over all prediction horizons > X.

**Source**: `auroc/` family

**Why included**:
- Shows **worst-case** uncertainty calibration
- Even if mean AUROC is high, low minimum indicates some horizons are poorly calibrated
- Important for identifying operating points where uncertainty fails

**Interpretation**:
- **Higher is better**: Minimum AUROC across horizons
- **If min << mean**: Some horizons have poor uncertainty calibration
- **Compare across conditions**: Higher minimum = more robust uncertainty across horizons

---

#### `risk_at_{80}cov_hgt{X}_mean` / `risk_at_{80}cov_hgt{X}_std`
**Definition**: Mean prediction error (risk) when filtering to keep top 80% lowest-uncertainty samples, over horizons > X.

**Source**: `risk_coverage/` family (risk_coverage_all.csv)

**Filtering**:
1. Rank samples by epistemic uncertainty (lowest first)
2. Keep top 80% (coverage = 0.80)
3. Compute mean error on kept subset

**Why included**:
- Shows **practical benefit of uncertainty-based filtering**
- Risk at 80% coverage: "What error do we get if we accept 20% abstention?"
- Direct measure of selective prediction value

**Interpretation**:
- **Lower is better**: Filtering by uncertainty significantly reduces error
- **Compare with risk_at_100cov**: If risk_80 << risk_100, filtering is effective
- **Compare across conditions**: Lower risk at 80% = better uncertainty-based selection

---

#### `risk_at_{100}cov_hgt{X}_mean` / `risk_at_{100}cov_hgt{X}_std`
**Definition**: Mean prediction error (risk) when keeping all samples (coverage = 1.0), over horizons > X.

**Source**: `risk_coverage/` family

**Why included**:
- Baseline error **without any filtering**
- Enables comparison: "How much do we gain by using uncertainty for selection?"
- Also called "no-filtering risk"

**Interpretation**:
- **Lower is better**: Model's raw error without selective prediction
- **Compare with risk_at_80cov**: If risk_80 << risk_100, filtering provides value
- **Compare across conditions**: Lower risk = better baseline predictions

**Interpretation example**:
- If `risk_at_80cov = 0.10` and `risk_at_100cov = 0.20`
- → Filtering to 80% coverage reduces error by 50% (valuable uncertainty signal)

---

### D) Correlation Diagnostics (Optional)

#### `pearson_mean_hgt{X}_mean` / `pearson_mean_hgt{X}_std`
**Definition**: Mean Pearson correlation coefficient between epistemic uncertainty and prediction error, over horizons > X.

**Source**: `correlation_by_horizon/` family

**Why included**:
- Linear relationship measure between uncertainty and error
- Positive correlation: Higher uncertainty → Higher error (desirable)
- Near zero: Uncertainty doesn't predict error (not useful)

**Interpretation**:
- **Higher (positive) is better**: Uncertainty and error are positively correlated
- **~ 0.5-0.8**: Strong positive correlation (good)
- **~ 0**: No correlation (uncertainty useless for that metric)
- **< 0**: Inverse correlation (unexpected, check for bugs)
- **Compare across conditions**: Higher correlation = better uncertainty-error relationship

---

#### `spearman_mean_hgt{X}_mean` / `spearman_mean_hgt{X}_std`
**Definition**: Mean Spearman (rank) correlation between epistemic uncertainty and prediction error, over horizons > X.

**Source**: `correlation_by_horizon/` family

**Why included**:
- Monotonic relationship measure (non-linear)
- Robust to outliers compared to Pearson
- Shows if higher uncertainty consistently maps to higher error

**Interpretation**:
- **Higher (positive) is better**: Ranking of uncertainty aligns with ranking of error
- **~ 0.5-0.8**: Strong monotonic relationship (good)
- **~ 0**: No monotonic relationship
- **Compare across conditions**: Higher Spearman = better uncertainty calibration

---

### E) Training Metrics (Optional)

#### `steady_state_reward_mean`
**Definition**: Mean tracking reward during steady-state training (last N steps, default N=500).

**Source**: `tensorboard/` family (TensorBoard CSVs)

**Why included**:
- Final performance metric for the policy
- Steady-state removes initial transient behavior
- Shows convergence level

**Interpretation**:
- **Higher is better**: Better policy performance
- **Compare across conditions**: Higher reward = better final policy

**Column used**: `Episode_Reward_track_lin_vel_xy_exp_mean`

---

#### `final_autoreg_error`
**Definition**: Mean autoregressive prediction error during steady-state training (last N steps).

**Source**: `tensorboard/` family

**Why included**:
- Measures world model prediction quality during training
- Lower error = model learned dynamics well
- Complements hallucination_horizon (which measures out-of-distribution error)

**Interpretation**:
- **Lower is better**: Better dynamics prediction
- **Compare across conditions**: Lower error = better world model

**Column used**: `System_Dynamics_autoregressive_error_mean`

---

#### `final_epistemic_uncertainty`
**Definition**: Mean epistemic uncertainty during steady-state training (last N steps).

**Source**: `tensorboard/` family

**Why included**:
- Shows model's learned uncertainty level
- Can indicate overconfident vs. underconfident behavior
- Useful for calibrating uncertainty thresholds

**Interpretation**:
- **No inherent "better" direction**: Depends on calibration
- **Too low**: Model may be overconfident
- **Too high**: Model may be underconfident
- **Compare with AUROC**: If uncertainty is high but AUROC is low, calibration is poor
- **Compare across conditions**: Similar to AUROC/correlation metrics

**Column used**: `Model_Based_epistemic_uncertainty_mean`

---

## Fairness Requirements

### 1. Shared Threshold for Hallucination Horizon
**Rule**: Hallucination horizon metrics (`hallucination_horizon_median`, `hallucination_horizon_iqr`) are only valid if a **shared threshold** τ is used across all conditions.

**Implementation**:
- τ is computed from a **single baseline condition** (e.g., B0)
- Using early horizons (1..baseline_horizon) of baseline only
- τ = p95 error from baseline in that window
- All conditions use the **same τ** for crossing detection

**Check**: Table includes `hallucination_threshold_shared` boolean column.
- `True`: Valid shared threshold used
- `False`: Per-condition thresholds (horizon metrics = NaN, not comparable)

---

### 2. Per-Horizon AUROC
**Rule**: AUROC must be computed **within each horizon** separately, then averaged across horizons.

**Why**: Pooling horizons would mix easy early predictions with hard late predictions, obscuring horizon-dependence.

**Implementation**:
- For each horizon h: compute AUROC using top-q errors at that horizon only
- Positives are top-q errors **within that horizon** (not across all horizons)
- Average AUROC across all h > horizon_filter

**Check**: Filter is applied (`auroc_mean_hgt{X}` indicates h > X used).

---

### 3. Horizon-Filtered Metrics
**Rule**: Metrics filtered by horizon (AUROC, risk, correlations) must explicitly record the filter.

**Implementation**:
- All filtered metric names include `hgt{X}` suffix (e.g., `auroc_mean_hgt30`)
- Table filename includes `__hgt{X}`
- Run summary logs horizon filter used

---

## Table Aggregation Rules

### Mean ± Standard Deviation
- Used for: AUROC, risk, noise AUC, correlations, training metrics
- Computation: `mean` and `std` across seeds/rollouts per condition
- Columns: `{metric}_mean`, `{metric}_std`

### Median + Interquartile Range (IQR)
- Used for: Hallucination horizon (skewed distribution)
- Computation: `median` and `IQR = Q75 - Q25` across seeds/episodes per condition
- Columns: `{metric}_median`, `{metric}_iqr`

### N (Sample Count)
- Optional: Number of seeds/rollouts contributing to each metric
- Useful for assessing statistical significance

---

## Output Naming

### Filename Format
```
summary_metrics__task-{TASK}__{MODE}__hgt{FILTER}__{MODE}.csv
```

### Examples
```
tables/summary_metrics__task-anymal__hgt30__full.csv
tables/summary_metrics__task-anymal__hgt30__vel.csv
tables/summary_metrics__task-franka__hgt10__full.csv
```

### Components
- `task-{TASK}`: Task type (anymal or franka)
- `{MODE}`: Error mode (vel or full)
- `hgt{FILTER}`: Horizon filter (e.g., hgt30 = horizons > 30)

---

## Usage Examples

### 1. Generate tables with default settings
```bash
python -m plotnine_plots --results_dir results/paper --make-tables
```
Default: horizon_filter = 30 (anymal), mode = both (vel + full), coverage_target = 0.80

---

### 2. Generate tables with custom horizon filter
```bash
python -m plotnine_plots --results_dir results/paper --horizon-filter 50 --make-tables
```

---

### 3. Generate tables for velocity mode only
```bash
python -m plotnine_plots --results_dir results/paper --mode vel --make-tables
```

---

### 4. Generate tables with specific coverage target
```bash
python -m plotnine_plots --results_dir results/paper --coverage-target 0.90 --make-tables
```

---

### 5. Disable table generation
```bash
python -m plotnine_plots --results_dir results/paper --no-make-tables
```

---

## CLI Flags for Table Generation

| Flag | Description | Default |
|-------|-------------|----------|
| `--make-tables` | Generate summary tables | True |
| `--no-make-tables` | Disable table generation | False |
| `--table-checkpoint-policy {last,specific}` | Checkpoint selection | last |
| `--checkpoint-step <int>` | Specific checkpoint step | None |
| `--training-tail-window <int>` | Trailing steps for training metrics | 500 |
| `--coverage-target <float>` | Coverage target for risk metrics | 0.80 |
| `--mode {vel,full,both}` | Error mode for tables | both |
| `--horizon-filter <int>` | Minimum horizon for filtered metrics | task default |

---

## Troubleshooting

### Issue: All metrics are NaN
**Cause**: No evaluation data found for requested mode.

**Solution**:
- Check that CSV files exist in `results_dir/` subdirectories
- Verify mode (vel/full) matches available columns
- Check logs for "No data" or "skipped" messages

---

### Issue: Hallucination horizon metrics are NaN
**Cause**: Threshold not shared (per-condition thresholds used).

**Solution**:
- Check `hallucination_threshold_shared` column (should be True)
- Verify hallucination horizons were recomputed with shared threshold
- Look for log: "Using recomputed shared-threshold hallucination horizons"

---

### Issue: AUROC metrics missing
**Cause**: AUROC CSV not found or missing `horizon_t` column.

**Solution**:
- Check `auroc/auroc_all.csv` exists
- Verify AUROC evaluation was run with per-horizon computation
- Check logs for "AUROC metrics: skipped"

---

## Integration with Run Summary

The `run_summary.txt` file includes a **"--- SUMMARY TABLES ---"** section with:
- Task type (anymal/franka)
- Horizon filter used
- Coverage target
- Mode (vel/full/both)
- Number of tables generated
- List of generated table filenames

Example:
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

## Notes for Paper Writing

When writing about these metrics:

1. **Always specify horizon filter**: "Over horizons > 30, condition X achieved..."
2. **Mention shared threshold for hallucination**: "Using a shared threshold computed from B0's early-horizon errors..."
3. **Explain AUROC computation**: "AUROC computed per-horizon, then averaged"
4. **Clarify risk-coverage**: "At 80% coverage (keeping lowest-uncertainty samples), risk was reduced by..."
5. **Report both mean ± std**: "Hallucination horizon: 45 ± 12 steps (median ± IQR)"
6. **Compare across modes**: For both vel and full modes when applicable

---

## References

- **Hallucination horizon**: See `PLOTTING.md` section on reliability plots
- **AUROC computation**: Binary classification with top-q errors as positives per horizon
- **Risk-coverage**: Selective prediction / abstention framework
- **Noise robustness**: Error vs. input perturbation magnitude
- **Correlations**: Linear (Pearson) and monotonic (Spearman) uncertainty-error relationships
