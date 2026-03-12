"""
transforms.py — Data transformations for plotting.

Seed aggregation, robust z-score normalization, velocity/full column
selection, and condition label shortening.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Velocity / full-state column sets
# ---------------------------------------------------------------------------
_VEL_MEAN_COLS = ["error_rel_v_mean", "error_rel_w_mean", "error_rel_q_dot_mean"]
_VEL_STD_COLS = ["error_rel_v_std", "error_rel_w_std", "error_rel_q_dot_std"]
_FULL_MEAN_COL = "error_rel_mean"
_FULL_STD_COL = "error_rel_std"


def has_velocity_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame contains velocity-decomposed error columns."""
    return all(c in df.columns for c in _VEL_MEAN_COLS)


def select_error_columns(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return DataFrame with unified ``error_mean`` / ``error_std`` columns.

    Parameters
    ----------
    mode : str
        ``"vel"`` — average the three velocity components.
        ``"full"`` — use ``error_rel_mean`` / ``error_rel_std``.
    """
    df = df.copy()
    if mode == "vel":
        if not has_velocity_columns(df):
            raise ValueError("Velocity columns not present in DataFrame")
        df["error_mean"] = df[_VEL_MEAN_COLS].mean(axis=1)
        # Propagate std: sqrt(mean of variances)
        df["error_std"] = np.sqrt((df[_VEL_STD_COLS] ** 2).mean(axis=1))
    elif mode == "full":
        if _FULL_MEAN_COL not in df.columns:
            raise ValueError(f"Column {_FULL_MEAN_COL!r} not in DataFrame")
        df["error_mean"] = df[_FULL_MEAN_COL]
        df["error_std"] = df.get(_FULL_STD_COL, 0.0)
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'vel' or 'full'")
    return df


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------
def aggregate_seeds(
    df: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
    agg: str = "mean_std",
) -> pd.DataFrame:
    """Aggregate across seeds within each group.

    Parameters
    ----------
    group_cols : list[str]
        Columns that define a group (e.g. ``["condition", "horizon_t"]``).
        ``"seed"`` is excluded automatically if present.
    value_cols : list[str]
        Numeric columns to aggregate.
    agg : str
        ``"mean_std"`` — compute mean and std across seeds.
        ``"median_iqr"`` — compute median, q25, q75 across seeds.

    Returns a DataFrame with ``{col}_mean`` and ``{col}_std`` (or
    ``{col}_median``, ``{col}_q25``, ``{col}_q75``).
    """
    group_cols = [c for c in group_cols if c != "seed" and c in df.columns]
    value_cols = [c for c in value_cols if c in df.columns]
    if not group_cols or not value_cols:
        return df

    if agg == "mean_std":
        agg_mean = df.groupby(group_cols, sort=True)[value_cols].mean()
        agg_std = df.groupby(group_cols, sort=True)[value_cols].std()
        # Always append _mean/_std, even if input already has that suffix
        agg_mean.columns = [f"{c}_mean" for c in agg_mean.columns]
        agg_std.columns = [f"{c}_std" for c in agg_std.columns]
        result = agg_mean.join(agg_std).reset_index()
    elif agg == "median_iqr":
        agg_med = df.groupby(group_cols, sort=True)[value_cols].median()
        agg_q25 = df.groupby(group_cols, sort=True)[value_cols].quantile(0.25)
        agg_q75 = df.groupby(group_cols, sort=True)[value_cols].quantile(0.75)
        agg_med.columns = [f"{c}_median" for c in agg_med.columns]
        agg_q25.columns = [f"{c}_q25" for c in agg_q25.columns]
        agg_q75.columns = [f"{c}_q75" for c in agg_q75.columns]
        result = agg_med.join(agg_q25).join(agg_q75).reset_index()
    else:
        raise ValueError(f"Unknown agg={agg!r}")
    return result


# ---------------------------------------------------------------------------
# Robust z-score normalisation  (u - median) / IQR  per condition
# ---------------------------------------------------------------------------
def robust_zscore(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "condition",
    out_col: Optional[str] = None,
) -> pd.DataFrame:
    """Add a robust z-score column ``(value - median) / IQR`` per group.

    If IQR == 0 for a group, falls back to ``(value - median) / mad``
    where ``mad = median(|value - median|)``.  If both are zero the
    z-score is set to 0.
    """
    df = df.copy()
    if out_col is None:
        out_col = f"{value_col}_zscore"

    df[out_col] = np.nan
    for name, grp in df.groupby(group_col):
        vals = grp[value_col].dropna()
        if vals.empty:
            continue
        med = vals.median()
        iqr = vals.quantile(0.75) - vals.quantile(0.25)
        if iqr == 0:
            mad = np.median(np.abs(vals - med))
            scale = mad if mad > 0 else 1.0
        else:
            scale = iqr
        df.loc[grp.index, out_col] = (grp[value_col] - med) / scale
    return df


# ---------------------------------------------------------------------------
# Condition label shortening
# ---------------------------------------------------------------------------
# Maps long condition names → short paper labels.
# Users can extend this; unknown conditions keep their original name.
_LABEL_MAP: dict[str, str] = {
    # paper/ conditions
    "bs-nopen": "B0",
    "bs-pen025": "Bp",
    "rp-nopen-std": "R0",
    "rp-pen006-std": "Rp",
    "rp-pen025-std": "R25",
    "rp-pen025-var": "Rv",
    # paper_all/ conditions (finetune- prefix)
    "finetune-bs": "B0",
    "finetune-bs-pen0.25": "Bp",
    "finetune-bs-nopen": "B0",
    "finetune-rp-nopen-std": "R0",
    "finetune-rp-pen0.06-std": "Rp",
    "finetune-rp-pen0.25-std": "R25",
    "finetune-rp-pen0.25-var": "Rv",
    # paper_franka/ conditions
    "bs-pen025": "Bp",
    "ensemble-nopen": "E0",
    "rp-nopen": "R0",
    "rp-pen006": "Rp",
}


def shorten_condition(name: str) -> str:
    """Return a short label for *name*, or *name* itself if unknown."""
    return _LABEL_MAP.get(name, name)


def add_short_labels(df: pd.DataFrame, col: str = "condition") -> pd.DataFrame:
    """Add a ``condition_short`` column with shortened labels."""
    if col not in df.columns:
        return df
    df = df.copy()
    df["condition_short"] = df[col].map(shorten_condition)
    return df


# ---------------------------------------------------------------------------
# Smoothing helpers (for plotting only, does not alter raw data)
# ---------------------------------------------------------------------------
def apply_rolling_smooth(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    window: int,
) -> pd.DataFrame:
    """Apply rolling mean smoothing to value_col, grouped by group_cols.

    Creates a new column {value_col}_smooth with rolling mean values.
    Original data is preserved. NaNs are introduced at boundaries where
    the window extends beyond available data.
    """
    df = df.copy()
    smooth_col = f"{value_col}_smooth"
    df[smooth_col] = np.nan

    for _, grp in df.groupby(group_cols, sort=True):
        grp = grp.sort_values(group_cols[-1] if len(group_cols) == 1 else group_cols)
        values = grp[value_col]
        rolling = values.rolling(window=window, center=True, min_periods=1).mean()
        df.loc[grp.index, smooth_col] = rolling.values

    return df


def apply_ewma_smooth(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Apply exponential weighted moving average smoothing to value_col.

    Creates a new column {value_col}_smooth with EWMA values.
    Original data is preserved.
    """
    df = df.copy()
    smooth_col = f"{value_col}_smooth"
    df[smooth_col] = np.nan

    for _, grp in df.groupby(group_cols, sort=True):
        grp = grp.sort_values(group_cols[-1] if len(group_cols) == 1 else group_cols)
        values = grp[value_col]
        ewma = values.ewm(alpha=alpha, adjust=False).mean()
        df.loc[grp.index, smooth_col] = ewma.values

    return df


def get_smoothed_column(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    mode: str = "none",
    window: int = 10,
    alpha: float = 0.1,
) -> tuple[pd.DataFrame, str]:
    """Apply smoothing to value_col based on mode and return smoothed column name.

    Parameters
    ----------
    df : DataFrame
        Input data.
    value_col : str
        Column to smooth.
    group_cols : list[str]
        Columns to group by for smoothing (typically condition, seed, etc.).
    mode : str
        "none", "rolling_mean", or "ewma".
    window : int
        Window size for rolling mean.
    alpha : float
        Alpha parameter for EWMA.

    Returns
    -------
    (df, col_name) where col_name is the column to use for plotting
    (either value_col or value_col_smooth).
    """
    if mode == "none":
        return df, value_col
    elif mode == "rolling_mean":
        df_smooth = apply_rolling_smooth(df, value_col, group_cols, window)
        return df_smooth, f"{value_col}_smooth"
    elif mode == "ewma":
        df_smooth = apply_ewma_smooth(df, value_col, group_cols, alpha)
        return df_smooth, f"{value_col}_smooth"
    else:
        log.warning("Unknown smoothing mode '%s'; using no smoothing", mode)
        return df, value_col


# ---------------------------------------------------------------------------
# Per-horizon AUROC computation (offline, from error_vs_uncertainty data)
# ---------------------------------------------------------------------------

def compute_auroc_per_horizon(
    eu_df: pd.DataFrame,
    top_q: float = 10.0,
    error_col: str = "error_rel_mean",
    uncertainty_col: str = "uncertainty_mean",
    horizon_col: str = "prediction_step",
    window: int = 10,
) -> pd.DataFrame:
    """Compute AUROC as a function of prediction horizon per condition.

    Because each (condition, seed, horizon) typically has only 1 row in
    aggregated evaluation data, we use a **sliding window** approach:

    For each condition, at each horizon *h*, collect all (error, uncertainty)
    pairs from horizons in ``[h - window//2, h + window//2]`` across **all
    seeds**.  Within that pooled set, label the top ``top_q`` percent of
    errors as positives and compute AUROC.

    This yields one AUROC value per (condition, horizon_t), which can then
    be aggregated across seeds if desired.

    Labeling protocol:
        Positives = top ``top_q`` % errors computed **within the horizon
        window** (per-horizon labeling — horizons outside the window are
        NOT included when computing the error threshold).

    Parameters
    ----------
    eu_df : DataFrame
        ``error_vs_uncertainty`` data.  Required columns:
        ``[horizon_col, error_col, uncertainty_col, 'condition', 'seed']``.
    top_q : float
        Percentile for labeling (default 10 → top 10 % errors are positive).
    error_col, uncertainty_col, horizon_col : str
        Column names.
    window : int
        Sliding window half-width in horizon steps (default 10).

    Returns
    -------
    DataFrame with columns:
        ``horizon_t, auroc, n_positive, n_negative, n_samples, condition``
    sorted by ``(condition, horizon_t)``.
    """
    required = {horizon_col, error_col, uncertainty_col, "condition", "seed"}
    missing = required - set(eu_df.columns)
    if missing:
        log.warning("compute_auroc_per_horizon: missing columns %s", missing)
        return pd.DataFrame()

    half_w = window // 2
    records: list[dict] = []

    for cond, cond_df in eu_df.groupby("condition", sort=True):
        horizons = sorted(cond_df[horizon_col].unique())

        for h in horizons:
            # Collect all samples in the horizon window, across all seeds
            mask = (
                (cond_df[horizon_col] >= h - half_w)
                & (cond_df[horizon_col] <= h + half_w)
            )
            window_df = cond_df.loc[mask]

            errors = window_df[error_col].values
            uncertainties = window_df[uncertainty_col].values

            # Remove NaN/Inf
            valid = np.isfinite(errors) & np.isfinite(uncertainties)
            errors = errors[valid]
            uncertainties = uncertainties[valid]

            if len(errors) < 10:
                continue

            # Per-window labeling: top q% of errors
            threshold = np.percentile(errors, 100.0 - top_q)
            labels = (errors >= threshold).astype(int)
            n_pos = int(labels.sum())
            n_neg = int(len(labels) - n_pos)

            if n_pos < 2 or n_neg < 2:
                continue

            try:
                auroc_val = float(roc_auc_score(labels, uncertainties))
            except ValueError:
                continue

            records.append({
                "horizon_t": int(h),
                "auroc": auroc_val,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_samples": n_pos + n_neg,
                "condition": cond,
            })

    if not records:
        log.warning("compute_auroc_per_horizon: produced 0 rows")
        return pd.DataFrame()

    result = pd.DataFrame(records).sort_values(
        ["condition", "horizon_t"]
    ).reset_index(drop=True)

    # Sanity warnings
    min_pos = result["n_positive"].min()
    if min_pos < 5:
        log.warning(
            "AUROC per-horizon: minimum n_positive = %d (< 5) at some "
            "horizons; estimates may be unstable there",
            min_pos,
        )

    log.info(
        "Computed per-horizon AUROC: %d rows, %d horizons, %d conditions "
        "(window=%d, top_q=%.0f%%)",
        len(result),
        result["horizon_t"].nunique(),
        result["condition"].nunique(),
        window,
        top_q,
    )
    return result


# ---------------------------------------------------------------------------
# Tensorboard metric extraction helpers
# ---------------------------------------------------------------------------
def extract_tb_metric(
    tb_dict: dict[str, pd.DataFrame],
    metric_stem: str,
    stat: str = "mean",
) -> pd.DataFrame:
    """Extract a single metric from the per-condition TensorBoard dict.

    Parameters
    ----------
    tb_dict : dict[str, DataFrame]
        Mapping from condition name to wide TensorBoard DataFrame.
    metric_stem : str
        Full column prefix, e.g. ``"Episode_Reward_track_lin_vel_xy_exp"``.
    stat : str
        Stat suffix — ``"mean"`` or ``"std"``.

    Returns
    -------
    Long-format DataFrame with columns ``[step, condition, value, std]``.
    """
    mean_col = f"{metric_stem}_mean"
    std_col = f"{metric_stem}_std"
    frames = []
    for cond, df in tb_dict.items():
        if mean_col not in df.columns:
            log.warning("Metric %s not found for condition %s", mean_col, cond)
            continue
        sub = pd.DataFrame({
            "step": df["step"],
            "condition": cond,
            "value": df[mean_col],
        })
        if std_col in df.columns:
            sub["std"] = df[std_col]
        else:
            sub["std"] = 0.0
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    # Drop rows with inf or NaN in value (e.g. autoregressive error at early steps)
    n_before = len(result)
    result = result[np.isfinite(result["value"]) & result["value"].notna()]
    result = result.reset_index(drop=True)
    n_dropped = n_before - len(result)
    if n_dropped > 0:
        log.info("extract_tb_metric(%s): dropped %d inf/NaN rows", metric_stem, n_dropped)
    return result


# ---------------------------------------------------------------------------
# Shared-threshold hallucination horizon recomputation
# ---------------------------------------------------------------------------
# Preferred reference conditions for computing the shared threshold.
# Tried in order; the first one found in the data is used.
_PREFERRED_BASELINES = ["bs-nopen", "bs-pen025"]


def recompute_hallucination_horizons(
    raw_error_df: pd.DataFrame,
    baseline_condition: Optional[str] = None,
    threshold_percentile: float = 95.0,
    baseline_horizon: int = 30,
    error_col: str = "error_rel_mean",
    step_col: str = "horizon_t",
) -> pd.DataFrame:
    """Recompute hallucination horizons using a **shared** threshold.

    Instead of per-condition thresholds (which make cross-condition
    comparison meaningless), this function:

    1. Selects a reference condition (``baseline_condition`` or
       auto-detected from ``_PREFERRED_BASELINES``).
    2. Collects error values from horizons 1..baseline_horizon
       across all seeds of that reference condition.
    3. Computes the ``threshold_percentile``-th percentile as the shared
       threshold.
    4. For every episode/run, finds the first horizon where
       ``error_col > threshold``.

    Parameters
    ----------
    raw_error_df : DataFrame
        Raw per-step error curves.  Expected columns:
        ``[step_col, error_col, 'condition', 'seed', 'checkpoint_step']``.
        If ``episode_id`` column exists, computes per-episode horizons;
        otherwise computes per-run horizons (grouped by condition, seed,
        checkpoint_step) and warns.
    baseline_condition : str, optional
        Condition name to derive the threshold from.  If *None*, the
        first condition found in ``_PREFERRED_BASELINES`` is used.
    threshold_percentile : float
        Percentile of baseline early-step errors (default 95).
    baseline_horizon : int
        Maximum horizon step to include in baseline window (default 30).
        Selects all horizons where 1 <= horizon <= baseline_horizon.
    error_col : str
        Column with per-step error values (default ``error_rel_mean``).
    step_col : str
        Column with horizon step values (default ``horizon_t``).

    Returns
    -------
    DataFrame with columns:
        condition, seed, checkpoint_step, [episode_id], hallucination_horizon,
        threshold, threshold_method, max_error, error_at_hallucination,
        n_steps.
    """
    required = {step_col, error_col, "condition", "seed", "checkpoint_step"}
    missing = required - set(raw_error_df.columns)
    if missing:
        log.warning("recompute_hallucination_horizons: missing columns %s", missing)
        return pd.DataFrame()

    df = raw_error_df.copy()
    has_episode_id = "episode_id" in df.columns

    # ── 1. Auto-detect baseline condition ──────────────────────────────
    conditions = df["condition"].unique().tolist()
    if baseline_condition is None:
        for pref in _PREFERRED_BASELINES:
            if pref in conditions:
                baseline_condition = pref
                break
    if baseline_condition is None:
        baseline_condition = sorted(conditions)[0]
        log.warning(
            "No preferred baseline found; falling back to '%s'",
            baseline_condition,
        )

    # ── 2. Compute shared threshold from baseline early horizons ───────
    baseline_mask = df["condition"] == baseline_condition
    early_mask = baseline_mask & (df[step_col] >= 1) & (df[step_col] <= baseline_horizon)
    baseline_errors = df.loc[early_mask, error_col].dropna()

    if baseline_errors.empty:
        log.warning(
            "No baseline errors for condition='%s' in window 1..%d; returning empty DataFrame",
            baseline_condition, baseline_horizon,
        )
        return pd.DataFrame()

    threshold = float(np.percentile(baseline_errors, threshold_percentile))
    baseline_stats = {
        "median": float(baseline_errors.median()),
        "p90": float(np.percentile(baseline_errors, 90)),
        "p95": float(np.percentile(baseline_errors, 95)),
    }

    threshold_method = f"shared_percentile_{threshold_percentile}_baseline={baseline_condition}_h=1..{baseline_horizon}"
    log.info(
        "Shared threshold = %.6f (from %d baseline observations in 1..%d: median=%.6f, p90=%.6f, p95=%.6f)",
        threshold, len(baseline_errors), baseline_horizon,
        baseline_stats["median"], baseline_stats["p90"], baseline_stats["p95"],
    )

    # ── 3. Determine granularity (per-episode vs per-run) ────────────────
    if not has_episode_id:
        log.warning(
            "No 'episode_id' column found; computing per-run horizons "
            "(grouped by condition, seed, checkpoint_step). "
            "Interpretation differs from per-episode horizons."
        )
        group_keys = ["condition", "seed", "checkpoint_step"]
        granularity = "per-run"
    else:
        group_keys = ["condition", "seed", "checkpoint_step", "episode_id"]
        granularity = "per-episode"

    # ── 4. Find first threshold crossing for each episode/run ───────────
    records = []
    max_horizon_observed = 0
    for keys, grp in df.groupby(group_keys, sort=True):
        grp = grp.sort_values(step_col)
        steps = grp[step_col].values
        errors = grp[error_col].values

        valid = np.isfinite(errors)
        steps = steps[valid]
        errors = errors[valid]
        n_steps = len(steps)

        if n_steps == 0:
            continue

        max_horizon_observed = max(max_horizon_observed, steps[-1])

        exceeds = errors > threshold
        if exceeds.any():
            first_idx = np.argmax(exceeds)
            h_halluc = int(steps[first_idx])
            err_at_h = float(errors[first_idx])
        else:
            h_halluc = int(steps[-1])
            err_at_h = float(errors[-1])

        record = {
            "condition": keys[0],
            "seed": keys[1],
            "checkpoint_step": keys[2],
            "hallucination_horizon": h_halluc,
            "threshold": threshold,
            "threshold_method": threshold_method,
            "max_error": float(errors.max()),
            "error_at_hallucination": err_at_h,
            "n_steps": n_steps,
        }
        if has_episode_id:
            record["episode_id"] = keys[3]
        records.append(record)

    result = pd.DataFrame(records)
    if result.empty:
        log.warning("recompute_hallucination_horizons produced 0 rows")
        return result

    # ── 5. Sanity checks ────────────────────────────────────────────────
    median_horizon = result["hallucination_horizon"].median()
    if median_horizon < 10 and max_horizon_observed >= 50:
        log.warning(
            "Sanity: median hallucination horizon (%.1f) is low while "
            "max horizon observed (%d) is large; threshold may be too low "
            "or wrong error column used (expected ~10-50+ steps for meaningful horizons)",
            median_horizon, max_horizon_observed,
        )

    log.info(
        "Recomputed hallucination horizons: %d %s, median horizon = %.1f steps",
        len(result), granularity, median_horizon,
    )
    return result
