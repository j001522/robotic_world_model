# Plotting — Metric Definitions and Normalization

This document specifies **what** each plot shows: the precise metric definitions,
data sources, normalization methods, and aggregation strategies. It covers all
plots produced by the `plotnine_plots` module.

---

## Table of Contents

1. [Condition Labels](#condition-labels)
2. [Seed Aggregation](#seed-aggregation)
3. [Normalization: Robust Z-Score](#normalization-robust-z-score)
4. [Error Modes (Full vs Velocity)](#error-modes-full-vs-velocity)
5. [Training Curves](#1-training-curves)
6. [Long-Horizon Error and Uncertainty](#2-long-horizon-error-and-uncertainty)
7. [Noise Robustness](#3-noise-robustness)
8. [Decision Utility](#4-decision-utility)
9. [Correlations](#5-correlations)
10. [Reliability (Hallucination Distributions)](#6-reliability-hallucination-distributions)
11. [Faithfulness](#7-faithfulness)
12. [Source-of-Truth Rules](#source-of-truth-rules)
13. [Sanity Checks](#sanity-checks)

---

## Condition Labels

Long condition names from the evaluation pipeline are shortened for plot legends:

| Long name           | Short label | Description                                      |
| ------------------- | ----------- | ------------------------------------------------ |
| `bs-nopen`          | **B0**      | Baseline, no penalty                             |
| `bs-pen025`         | **Bp**      | Baseline, penalty 0.25                           |
| `rp-nopen-std`      | **R0**      | Reward penalty, no penalty, std ensemble         |
| `rp-pen006-std`     | **R6**      | Reward penalty 0.06, std ensemble                |
| `rp-pen025-std`     | **Rp**      | Reward penalty 0.25, std ensemble                |
| `rp-pen025-var`     | **Rv**      | Reward penalty 0.25, variance ensemble           |
| `ensemble-nopen`    | **E0**      | Ensemble baseline, no penalty (Franka only)      |
| `rp-nopen`          | **R0**      | Reward penalty, no penalty (Franka)              |
| `rp-pen006`         | **R6**      | Reward penalty 0.06 (Franka)                     |

---

## Seed Aggregation

All evaluation CSVs contain per-seed rows (seeds: 42, 123, 1001). Before
plotting, values are aggregated across seeds within each
`(condition, x-axis-variable)` group:

- **Mean ± Std**: `mean(values)` and `std(values)` across seeds. The std
  is shown as a ribbon band around the mean line.
- The `seed` column is always excluded from the grouping keys.

For TensorBoard data, aggregation across seeds is already done in the
`tensorboard/aggregated/` CSVs. Each metric has `_mean` and `_std` suffixes
computed across seeds 42, 123, and 1001.

---

## Normalization: Robust Z-Score

Some trend metrics (epistemic uncertainty, prediction uncertainty) are
produced in **two versions**: raw and normalized.

**Robust z-score** is computed independently per condition:

```
z = (u - median(u)) / IQR(u)
```

where:
- `u` = the raw metric value for that condition
- `median(u)` = median of all values of `u` within that condition
- `IQR(u)` = Q75 - Q25 of `u` within that condition

Fallback: if IQR = 0, the denominator becomes `MAD = median(|u - median(u)|)`.
If MAD is also 0, the z-score is set to 0.

**Purpose**: conditions may operate at very different absolute uncertainty
scales. The z-score removes the per-condition baseline, making it possible to
compare *trends* (e.g. "does uncertainty grow faster than error?") on a
common axis. It is **not** used for decision-utility plots (AUROC,
risk-coverage), which require raw uncertainty to be meaningful.

---

## Error Modes (Full vs Velocity)

Prediction error comes in two flavours:

| Mode   | Columns used                                                                 | Aggregation                                           | Filename suffix |
| ------ | ---------------------------------------------------------------------------- | ----------------------------------------------------- | --------------- |
| `full` | `error_rel_mean`, `error_rel_std`                                            | Used directly                                         | `__full`        |
| `vel`  | `error_rel_v_mean`, `error_rel_w_mean`, `error_rel_q_dot_mean` (+ `_std`)   | Mean across the three components; std via `sqrt(mean(var_i))` | `__vel`         |

- **`error_rel_mean`** = relative L2 error of the full predicted state vs
  ground truth, averaged over episodes, at a given prediction step.
- **`error_rel_v_mean`** = relative error of the linear velocity component only.
- **`error_rel_w_mean`** = relative error of the angular velocity component only.
- **`error_rel_q_dot_mean`** = relative error of the joint velocity component only.

Velocity columns only exist in `paper_all/`; they are absent from `paper/`
and `paper_franka/`. When absent, `__vel` plots are skipped automatically.

---

## 1. Training Curves

**Subfolder**: `training/`  
**Data source**: `tensorboard/aggregated/<condition>.csv`

Each condition's TensorBoard CSV has ~473 columns (AnymalD) or ~361 columns
(Franka), with one row per training step. Metrics are pre-aggregated across
seeds (suffixes: `_mean`, `_std`, `_min`, `_max`, `_count`, `_seed42`,
`_seed123`, `_seed1001`).

### 1a. Reward vs Step

- **File**: `training/reward_vs_step.pdf`
- **Metric**: `Episode_Reward_track_lin_vel_xy_exp_mean`
- **Definition**: The exponential reward for tracking the commanded linear
  velocity in the XY plane, averaged across all episodes within a training
  epoch, then across seeds.
- **Y-axis**: Reward (lin vel xy)
- **Band**: ± `Episode_Reward_track_lin_vel_xy_exp_std` (cross-seed std)
- **Note**: This metric does not exist in Franka TensorBoard data (different
  robot, different reward terms). The plot is automatically skipped for Franka.

### 1b. Autoregressive Error vs Step

- **File**: `training/autoregressive_error_vs_step.pdf`
- **Metric**: `System_Dynamics_autoregressive_error_mean`
- **Definition**: The multi-step autoregressive prediction error of the
  learned world model. The model predicts `k` steps ahead using its own
  predictions as input (no ground-truth correction). The error measures how
  quickly the model's imagined trajectory diverges from reality.
- **Y-axis**: Error
- **Band**: ± `System_Dynamics_autoregressive_error_std`

### 1c. Epistemic Uncertainty vs Step (raw)

- **File**: `training/epistemic_uncertainty_vs_step.pdf`
- **Metric**: `Model_Based_epistemic_uncertainty_mean`
- **Definition**: The epistemic uncertainty of the world model ensemble.
  Computed as the disagreement (variance or std, depending on condition
  configuration) across the ensemble members' predictions. Higher values
  indicate more model disagreement about the predicted next state.
- **Y-axis**: Uncertainty
- **Band**: ± `Model_Based_epistemic_uncertainty_std`

### 1d. Epistemic Uncertainty vs Step (z-scored)

- **File**: `training/epistemic_uncertainty_vs_step__zscore.pdf`
- **Metric**: Robust z-score of `Model_Based_epistemic_uncertainty_mean`,
  computed per condition (see [Normalization](#normalization-robust-z-score)).
- **Y-axis**: Robust z-score
- **Band**: ± the original `_std` (not z-scored), which gives a sense of
  cross-seed spread at the normalized scale.
- **Purpose**: Compare uncertainty *trends* across conditions that have
  different absolute uncertainty magnitudes.

---

## 2. Long-Horizon Error and Uncertainty

**Subfolder**: `long_horizon/`  
**Data source (error)**: `hallucination_horizon/hallucination_horizon_all.csv`  
**Data source (uncertainty)**: `error_vs_uncertainty/error_vs_uncertainty_all.csv`

### 2a. Error vs Horizon

- **File**: `long_horizon/error_vs_horizon__full.pdf`
- **X-axis**: `horizon_t` — the prediction horizon in simulation steps
  (1, 2, ..., up to the maximum evaluated horizon, typically 168).
- **Y-axis**: `error_rel_mean` — the relative L2 prediction error, defined as:

  ```
  error_rel = ||predicted_state - true_state||_2 / ||true_state||_2
  ```

  averaged over all episodes and steps at that horizon, per seed.
- **Aggregation**: Mean ± std across seeds, per (condition, horizon).
- **Interpretation**: Shows how prediction accuracy degrades as the model
  rolls out further into the future. Steeper curves indicate faster
  accumulation of compounding errors.

### 2b. Uncertainty vs Horizon (raw)

- **File**: `long_horizon/uncertainty_vs_horizon.pdf`
- **X-axis**: `prediction_step` (same meaning as `horizon_t`, from the
  `error_vs_uncertainty` dataset which uses this column name).
- **Y-axis**: `uncertainty_mean` — the raw epistemic uncertainty at each
  prediction step, computed as ensemble disagreement (std or variance across
  ensemble members' predicted next states).
- **Aggregation**: Mean ± std across seeds, per (condition, prediction_step).
- **Interpretation**: Shows whether the model "knows" it is becoming less
  reliable at longer horizons.

### 2c. Uncertainty vs Horizon (z-scored)

- **File**: `long_horizon/uncertainty_vs_horizon__zscore.pdf`
- **Metric**: Robust z-score of `uncertainty_mean` per condition.
- **Y-axis**: Robust z-score
- **Purpose**: Same as 1d — compare uncertainty growth *trends* across
  conditions on a common scale.

---

## 3. Noise Robustness

**Subfolder**: `noise_robustness/`  
**Data source**: `noise_robustness/noise_robustness_all.csv`

### 3a. Error vs Noise Level

- **File**: `noise_robustness/error_vs_noise_level.pdf`
- **X-axis**: `noise_level` — the magnitude of Gaussian noise injected into
  the observations (action inputs) during evaluation. Values range from 0.0
  (clean) to some maximum (e.g. 0.3).
- **Y-axis**: `error_mean` — the mean prediction error (L2) across all
  episodes and horizons at that noise level. This is a scalar summary per
  (condition, seed, noise_level).
- **Aggregation**: Mean ± std across seeds per (condition, noise_level).
- **Representation**: Lines with points and error bars (not ribbons), since
  noise levels are discrete.
- **Interpretation**: Measures how gracefully each condition degrades under
  observation noise. Flatter curves indicate greater robustness.

---

## 4. Decision Utility

**Subfolder**: `decision_utility/`

These plots evaluate whether the uncertainty estimate is *useful for
decision-making*: can it reliably distinguish "good" predictions from "bad"
ones? Raw uncertainty is used (no z-scoring), since the absolute
threshold/ranking matters for binary classification.

### 4a. Risk-Coverage Curve

- **File**: `decision_utility/risk_coverage.pdf`
- **Data source**: `risk_coverage/risk_coverage_all.csv`
- **X-axis**: `coverage_fraction` — the fraction of predictions retained
  after filtering out those with uncertainty above a given percentile
  threshold. Ranges from near 0 (only the most certain predictions) to 1.0
  (all predictions).
- **Y-axis**: `mean_error_filtered` — the mean prediction error of the
  retained predictions.
- **Curve construction**: For each `uncertainty_percentile` (0, 1, 2, ...,
  99), predictions with uncertainty above that percentile are removed. The
  remaining predictions' mean error and coverage fraction are recorded.
- **Aggregation**: Mean ± std across seeds per (condition,
  uncertainty_percentile).
- **Interpretation**: A good uncertainty estimator produces a curve that
  rises sharply — low error at low coverage (the model correctly identifies
  its best predictions) and higher error only when forced to cover more data.
  Lower curves = better selective prediction.

### 4b. AUROC vs Horizon (line plot)

- **File**: `decision_utility/auroc_vs_horizon.pdf`
- **Condition for this variant**: Produced when the AUROC CSV contains a
  `horizon_t` column (as in `paper_franka/`).
- **Data source**: `auroc/<files>.csv` (5 columns: `condition, seed,
  checkpoint_step, horizon_t, auroc`).
- **X-axis**: `horizon_t` — prediction horizon.
- **Y-axis**: `auroc` — Area Under the ROC Curve for the binary
  classification task: "is this prediction a hallucination?" using the
  epistemic uncertainty as the score.
- **Definition of the classification task**: A prediction at horizon `t` is
  labelled positive (hallucination) if `error_rel > error_threshold`, and the
  uncertainty is used as the classifier score. AUROC measures how well
  uncertainty separates hallucinated from non-hallucinated predictions.
- **Aggregation**: Mean ± std across seeds per (condition, horizon_t).
- **Interpretation**: AUROC = 0.5 means uncertainty is no better than random
  at detecting hallucinations. AUROC = 1.0 means perfect separation.

### 4c. AUROC Bar Chart

- **File**: `decision_utility/auroc_bar.pdf`
- **Condition for this variant**: Produced when the AUROC CSV does NOT
  contain `horizon_t` (as in `paper/`), meaning there is a single AUROC per
  (condition, seed).
- **Data source**: `auroc/auroc_all.csv` (11 columns including `auroc`,
  `optimal_threshold`, `optimal_f1`, `tpr_at_0.1_fpr`, `n_positive`,
  `n_samples`).
- **Y-axis**: `auroc` — same definition as 4b, but collapsed across all
  horizons.
- **Additional columns available** (not plotted, but in the data):
  - `optimal_threshold`: uncertainty threshold that maximizes F1
  - `optimal_f1`: F1 at that threshold
  - `tpr_at_0.1_fpr`: true positive rate at 10% false positive rate
  - `n_positive`: number of hallucinated predictions (must be >= 5 for
    AUROC to be reliable)
  - `error_threshold`: the relative error threshold used to define
    hallucinations.
- **Aggregation**: Mean ± std across seeds per condition (error bars).

---

## 5. Correlations

**Subfolder**: `correlations/`  
**Data source**: `correlation_by_horizon/correlation_by_horizon_all.csv`

### 5a. Pearson Correlation vs Horizon

- **File**: `correlations/pearson_vs_horizon.pdf`
- **X-axis**: `prediction_step` — prediction horizon.
- **Y-axis**: `pearson_r` — Pearson product-moment correlation coefficient
  between the raw uncertainty and the raw prediction error, computed across
  all episodes at that prediction step.
- **Range**: [-1, 1]. A dashed grey reference line is drawn at r=0.
- **Aggregation**: Mean ± std across seeds per (condition, prediction_step).
- **Interpretation**: Measures the **linear** relationship between
  uncertainty and error. High positive values mean that when uncertainty is
  high, error also tends to be high — the model's confidence is well
  calibrated in a linear sense.

### 5b. Spearman Correlation vs Horizon

- **File**: `correlations/spearman_vs_horizon.pdf`
- **X-axis**: `prediction_step`
- **Y-axis**: `spearman_r` — Spearman rank correlation coefficient between
  uncertainty and error.
- **Interpretation**: Measures the **monotonic** (rank-order) relationship.
  More robust to non-linear relationships than Pearson. If Spearman is high
  but Pearson is lower, the relationship exists but is non-linear.
- **Additional data** (not plotted): `pearson_p` and `spearman_p` (p-values),
  `n_samples` (number of samples in the correlation), `history_horizon`,
  `uncertainty_metric`.

---

## 6. Reliability (Hallucination Distributions)

**Subfolder**: `reliability/`  
**Data source**: `hallucination_stats/hallucination_per_episode_all.csv`
(preferred) or `hallucination_stats/hallucination_distribution_all.csv`
(fallback)

### 6a. Hallucination Horizon Distribution (Histogram)

- **File**: `reliability/hallucination_horizon_distribution.pdf`
- **Metric**: `hallucination_horizon` — the number of simulation steps the
  model can predict before its relative error exceeds a defined threshold.
- **Definition**: For each evaluation episode, the model rolls out
  autoregressively from step 1. The hallucination horizon is the first step
  `t` where `error_rel(t) > threshold`. If the error never exceeds the
  threshold within the evaluation window, the hallucination horizon equals
  the maximum horizon.
- **Threshold**: defined in the `threshold` column (per-episode data) or
  `threshold_method` column (distribution data). May be a fixed global
  value or a per-condition value.
- **Representation**: Per-condition faceted histograms (30 bins).
- **Interpretation**: Right-shifted distributions indicate models that
  stay accurate for more steps. Very peaked distributions at low values
  suggest the model hallucinates almost immediately.

### 6b. Hallucination Horizon Boxplot

- **File**: `reliability/hallucination_horizon_boxplot.pdf`
- **Metric**: Same `hallucination_horizon` as 6a.
- **Representation**: One boxplot per condition (horizontal), showing
  median, IQR, whiskers, and outliers.
- **Interpretation**: Compact summary for cross-condition comparison.

**Diagnostic flag**: If the hallucination threshold is per-condition
(rather than a single global threshold), these plots are labelled
`diagnostic_only` in the run summary. Per-condition thresholds make
cross-condition comparison of absolute horizon values unreliable,
though within-condition spread is still informative.

---

## 7. Faithfulness

**Subfolder**: `faithfulness/`  
**Data source**: `faithfulness_gap/faithfulness_comparison.csv`

This data exists only in `paper_all/` (legacy). It is automatically skipped
for `paper/` and `paper_franka/`.

### 7a. Faithfulness Gap vs Step

- **File**: `faithfulness/faithfulness_gap_vs_step.pdf`
- **X-axis**: `step` — training step.
- **Y-axis**: `faithfulness_gap` — defined as:

  ```
  faithfulness_gap = imagination_reward_mean - real_reward_mean
  ```

  The difference between the reward the model *predicts it would get* (by
  rolling out imagined trajectories through the learned world model) and the
  reward actually achieved in the real environment.
- **Reference line**: Dashed grey line at y=0 (perfect faithfulness).
- **Interpretation**: Positive gap means the model is optimistic (imagined
  reward > real reward). Negative gap means pessimistic. A gap converging
  to zero over training indicates improving model faithfulness.

### 7b. Real vs Imagined Reward

- **File**: `faithfulness/real_vs_imagined_reward.pdf`
- **X-axis**: `step` — training step.
- **Y-axis**: `reward` — two lines per condition:
  - **Solid**: `real_reward_mean` — mean reward achieved in the real
    environment across `n_seeds_real` seeds.
  - **Dashed**: `imagination_reward_mean` — mean reward predicted by rolling
    out the learned world model.
- **Band** (solid line only): ± `real_reward_std` when available.
- **Interpretation**: Shows how the real and imagined rewards evolve during
  training and whether they converge.

---

## Source-of-Truth Rules

When multiple CSV families contain overlapping metrics, the module enforces
precedence rules to avoid plotting the same metric twice from different
sources:

| Metric              | Primary source                         | Fallback source           |
| ------------------- | -------------------------------------- | ------------------------- |
| Error vs horizon    | `hallucination_horizon/`               | `error_vs_uncertainty/`   |
| Uncertainty vs horizon | `error_vs_uncertainty/`             | (none)                    |
| Noise robustness    | `noise_robustness/`                    | (none)                    |
| AUROC               | `auroc/`                               | (none)                    |
| Risk-coverage       | `risk_coverage/`                       | (none)                    |
| Correlations        | `correlation_by_horizon/`              | (none)                    |
| Hallucination dists | `hallucination_stats/` (per_episode)   | `hallucination_stats/` (distribution) |
| Training curves     | `tensorboard/aggregated/`              | (none; `all_runs.csv` is ignored) |
| Faithfulness        | `faithfulness_gap/` (comparison file)  | (none)                    |

Within each family, `*_all.csv` files (containing all conditions) are
preferred over per-condition files.

---

## Sanity Checks

Before plotting, the following data quality checks are run. Warnings are
logged and written to `run_summary.txt`:

| Check                    | Applies to                | Condition                                           |
| ------------------------ | ------------------------- | --------------------------------------------------- |
| Contiguous horizons      | hallucination_horizon, error_vs_uncertainty | Horizons must be integers 1, 2, ..., N with no gaps |
| AUROC n_positive         | auroc                     | Warns if fewer than 5 positive (hallucinated) samples — AUROC unreliable |
| Coverage monotonicity    | risk_coverage             | `coverage_fraction` must be monotonically non-decreasing and within (0, 1] |
| Hallucination median     | hallucination_stats       | Warns if median hallucination horizon < 10 steps    |
| Hallucination saturation | hallucination_stats       | Warns if all episodes have the same hallucination horizon (= max), suggesting the threshold is too lenient |
| Correlation n_samples    | correlation_by_horizon    | Warns if any row has fewer than 10 samples          |
