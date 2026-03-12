"""
tables.py — Generate summary metrics tables per condition.

Creates tables/summary_metrics__task-{TASK}__{MODE}__hgt{FILTER}.csv
with aggregated metrics from all evaluation sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .discovery import Inventory
from .transforms import add_short_labels, _LABEL_MAP, compute_auroc_per_horizon

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions for computing metrics
# ---------------------------------------------------------------------------

def _safe_agg(series: pd.Series, func: str) -> float:
    """Aggregate a series safely, returning NaN on failure."""
    try:
        if func == "mean":
            return float(series.mean())
        elif func == "std":
            return float(series.std())
        elif func == "min":
            return float(series.min())
        elif func == "median":
            return float(series.median())
    except (ValueError, TypeError):
        pass
    return np.nan


def compute_auroc_metrics_per_condition(
    auroc_df: Optional[pd.DataFrame],
    eu_df: Optional[pd.DataFrame],
    horizon_filter: int,
    top_q: float,
    summary_list: list,
) -> pd.DataFrame:
    """Compute per-condition AUROC metrics.

    Returns DataFrame with columns: condition, auroc_mean, auroc_std, auroc_min
    """
    # Try CSV-based per-horizon AUROC first
    if auroc_df is not None and not auroc_df.empty and "horizon_t" in auroc_df.columns:
        summary_list.append("  AUROC metrics: using CSV data with horizon_t")
        filtered = auroc_df[auroc_df["horizon_t"] > horizon_filter].copy()
    elif eu_df is not None and not eu_df.empty:
        # Compute per-horizon AUROC from error_vs_uncertainty
        summary_list.append(f"  AUROC metrics: computing from error_vs_uncertainty (top_q={top_q}%)")
        computed = compute_auroc_per_horizon(eu_df, top_q=top_q)
        if computed.empty:
            summary_list.append("  AUROC metrics: computation returned empty")
            return pd.DataFrame()
        filtered = computed[computed["horizon_t"] > horizon_filter].copy()
    else:
        # Aggregated CSV only — no per-horizon data
        if auroc_df is not None and not auroc_df.empty:
            summary_list.append("  AUROC metrics: using aggregated CSV (no horizon_t)")
            grouped = auroc_df.groupby("condition")["auroc"].agg(["mean", "std", "min"]).reset_index()
            grouped.columns = ["condition", "auroc_mean", "auroc_std", "auroc_min"]
            return grouped
        summary_list.append("  AUROC metrics: no data available")
        return pd.DataFrame()

    if filtered.empty:
        summary_list.append(f"  AUROC metrics: no data after horizon filter h > {horizon_filter}")
        return pd.DataFrame()

    # Aggregate: group by condition (and seed if available), then across seeds
    if "seed" in filtered.columns:
        per_seed = filtered.groupby(["condition", "seed"])["auroc"].agg(["mean", "min"]).reset_index()
        per_seed.columns = ["condition", "seed", "auroc_seed_mean", "auroc_seed_min"]
        result = per_seed.groupby("condition").agg(
            auroc_mean=("auroc_seed_mean", "mean"),
            auroc_std=("auroc_seed_mean", "std"),
            auroc_min=("auroc_seed_min", "min"),
        ).reset_index()
    else:
        # Computed AUROC (pooled across seeds per horizon) — just aggregate across horizons
        result = filtered.groupby("condition")["auroc"].agg(["mean", "std", "min"]).reset_index()
        result.columns = ["condition", "auroc_mean", "auroc_std", "auroc_min"]

    n_conditions = len(result)
    summary_list.append(f"  AUROC metrics: {n_conditions} conditions, h > {horizon_filter}")
    return result


def compute_risk_metrics_per_condition(
    risk_df: Optional[pd.DataFrame],
    horizon_filter: int,
    coverage_target: float,
    summary_list: list,
) -> pd.DataFrame:
    """Compute per-condition risk metrics. Returns DataFrame with condition column."""
    if risk_df is None or risk_df.empty or "coverage_fraction" not in risk_df.columns:
        summary_list.append("  Risk metrics: skipped (no data)")
        return pd.DataFrame()

    df = risk_df.copy()

    # Find risk at coverage_target for each (condition, seed)
    records = []
    for (cond, seed), grp in df.groupby(["condition", "seed"]):
        grp = grp.sort_values("coverage_fraction")

        # Risk at coverage_target (closest)
        cov_diff = (grp["coverage_fraction"] - coverage_target).abs()
        idx_target = cov_diff.idxmin()
        risk_at_target = grp.loc[idx_target, "mean_error_filtered"]

        # Risk at coverage=1.0 (no filtering)
        idx_full = (grp["coverage_fraction"] - 1.0).abs().idxmin()
        risk_at_full = grp.loc[idx_full, "mean_error_filtered"]

        records.append({
            "condition": cond,
            "seed": seed,
            "risk_at_target": risk_at_target,
            "risk_at_full": risk_at_full,
        })

    if not records:
        summary_list.append("  Risk metrics: no valid groups")
        return pd.DataFrame()

    per_seed = pd.DataFrame(records)
    result = per_seed.groupby("condition").agg(
        risk_at_target_mean=("risk_at_target", "mean"),
        risk_at_target_std=("risk_at_target", "std"),
        risk_at_full_mean=("risk_at_full", "mean"),
        risk_at_full_std=("risk_at_full", "std"),
    ).reset_index()

    summary_list.append(f"  Risk metrics: {len(result)} conditions, coverage_target={coverage_target}")
    return result


def compute_correlation_metrics_per_condition(
    corr_df: Optional[pd.DataFrame],
    horizon_filter: int,
    summary_list: list,
) -> pd.DataFrame:
    """Compute per-condition correlation metrics."""
    if corr_df is None or corr_df.empty or "prediction_step" not in corr_df.columns:
        summary_list.append("  Correlation metrics: skipped (no data)")
        return pd.DataFrame()

    filtered = corr_df[corr_df["prediction_step"] > horizon_filter].copy()
    if filtered.empty:
        summary_list.append(f"  Correlation metrics: no data after h > {horizon_filter}")
        return pd.DataFrame()

    # Mean per (condition, seed), then across seeds
    records = []
    for (cond, seed), grp in filtered.groupby(["condition", "seed"]):
        rec = {"condition": cond, "seed": seed}
        for col in ["pearson_r", "spearman_r"]:
            if col in grp.columns:
                rec[col] = grp[col].mean()
        records.append(rec)

    if not records:
        return pd.DataFrame()

    per_seed = pd.DataFrame(records)
    agg_dict = {}
    for col in ["pearson_r", "spearman_r"]:
        if col in per_seed.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_std"] = (col, "std")

    result = per_seed.groupby("condition").agg(**agg_dict).reset_index()
    summary_list.append(f"  Correlation metrics: {len(result)} conditions, h > {horizon_filter}")
    return result


def compute_noise_metrics_per_condition(
    noise_df: Optional[pd.DataFrame],
    summary_list: list,
) -> pd.DataFrame:
    """Compute per-condition noise AUC."""
    if noise_df is None or noise_df.empty or "noise_level" not in noise_df.columns:
        summary_list.append("  Noise metrics: skipped (no data)")
        return pd.DataFrame()

    records = []
    for (cond, seed), grp in noise_df.groupby(["condition", "seed"]):
        grp = grp.sort_values("noise_level")
        if len(grp) < 2:
            continue
        x = grp["noise_level"].values
        y = grp["error_mean"].values if "error_mean" in grp.columns else grp.get("error_rel_mean", pd.Series()).values
        if len(y) == 0:
            continue
        noise_auc = float(np.trapz(y, x))
        records.append({"condition": cond, "seed": seed, "noise_auc": noise_auc})

    if not records:
        return pd.DataFrame()

    per_seed = pd.DataFrame(records)
    result = per_seed.groupby("condition").agg(
        noise_auc_mean=("noise_auc", "mean"),
        noise_auc_std=("noise_auc", "std"),
    ).reset_index()

    summary_list.append(f"  Noise metrics: {len(result)} conditions")
    return result


def compute_training_metrics_per_condition(
    tb_dict: Optional[dict[str, pd.DataFrame]],
    tail_window: int,
    summary_list: list,
) -> pd.DataFrame:
    """Compute per-condition training metrics from TensorBoard data."""
    if not tb_dict:
        summary_list.append("  Training metrics: no data")
        return pd.DataFrame()

    records = []
    for cond, df in tb_dict.items():
        df = df.sort_values("step")
        tail = df.tail(tail_window) if len(df) >= tail_window else df

        rec = {"condition": cond}

        # Reward
        reward_cols = [c for c in df.columns if "Episode_Reward_track_lin_vel_xy_exp_mean" in c]
        if reward_cols:
            rec["reward_mean"] = float(tail[reward_cols[0]].mean())

        # Autoregressive error
        ae_cols = [c for c in df.columns if "System_Dynamics_autoregressive_error_mean" in c]
        if ae_cols:
            rec["autoreg_error_mean"] = float(tail[ae_cols[0]].mean())

        # Epistemic uncertainty
        eu_cols = [c for c in df.columns if "Model_Based_epistemic_uncertainty_mean" in c]
        if eu_cols:
            rec["epistemic_unc_mean"] = float(tail[eu_cols[0]].mean())

        records.append(rec)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    summary_list.append(f"  Training metrics: {len(result)} conditions")
    return result


# ---------------------------------------------------------------------------
# Main table generation function
# ---------------------------------------------------------------------------

def make_summary_table(
    inv: Inventory,
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
    hall_df: Optional[pd.DataFrame],
    eu_df: Optional[pd.DataFrame],
    corr_df: Optional[pd.DataFrame],
    noise_df: Optional[pd.DataFrame],
    auroc_df: Optional[pd.DataFrame],
    risk_df: Optional[pd.DataFrame],
    hall_per_ep: Optional[pd.DataFrame],
    hall_distrib: Optional[pd.DataFrame],
    tb_dict: Optional[dict],
    faith_df: Optional[pd.DataFrame],
) -> list[Path]:
    """Generate summary metrics table per condition.

    Returns list of generated table file paths.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary.append("")
    summary.append("--- SUMMARY TABLES ---")

    # Extract config
    task = plot_config.get("task", "anymal")
    horizon_filter = plot_config.get("horizon_filter", 30)
    coverage_target = plot_config.get("coverage_target", 0.80)
    table_mode = plot_config.get("table_mode", "both")
    training_tail_window = plot_config.get("training_tail_window", 500)
    top_q = plot_config.get("top_q") or 10.0

    summary.append(f"  task: {task}")
    summary.append(f"  horizon_filter: h > {horizon_filter}")
    summary.append(f"  coverage_target: {coverage_target}")
    summary.append(f"  table_mode: {table_mode}")
    summary.append("")

    saved_paths: list[Path] = []
    modes = [table_mode] if table_mode != "both" else ["full", "vel"]

    for mode in modes:
        summary.append(f"Computing table for mode: {mode}")

        # Collect all conditions from all available data
        all_conditions: set[str] = set()
        for df in [auroc_df, hall_df, risk_df, eu_df, corr_df, noise_df]:
            if df is not None and not df.empty and "condition" in df.columns:
                all_conditions.update(df["condition"].unique())
        if tb_dict:
            all_conditions.update(tb_dict.keys())

        if not all_conditions:
            summary.append("  No conditions found, skipping table generation")
            continue

        conditions = sorted(all_conditions)
        summary.append(f"  Found {len(conditions)} conditions: {conditions}")

        # Start building the table as a dict of {condition: {metric: value}}
        table_data: dict[str, dict] = {c: {"condition": c} for c in conditions}
        for c in conditions:
            table_data[c]["condition_short"] = _LABEL_MAP.get(c, c)

        # A) AUROC metrics
        summary.append("  A) AUROC metrics:")
        auroc_metrics = compute_auroc_metrics_per_condition(
            auroc_df, eu_df, horizon_filter, top_q, summary,
        )
        if not auroc_metrics.empty:
            for _, row in auroc_metrics.iterrows():
                c = row["condition"]
                if c in table_data:
                    for col in auroc_metrics.columns:
                        if col != "condition":
                            table_data[c][f"auroc_{col}" if not col.startswith("auroc") else col] = row[col]

        # B) Risk metrics
        summary.append("  B) Risk metrics:")
        risk_metrics = compute_risk_metrics_per_condition(
            risk_df, horizon_filter, coverage_target, summary,
        )
        if not risk_metrics.empty:
            for _, row in risk_metrics.iterrows():
                c = row["condition"]
                if c in table_data:
                    for col in risk_metrics.columns:
                        if col != "condition":
                            table_data[c][col] = row[col]

        # C) Correlation metrics
        summary.append("  C) Correlation metrics:")
        corr_metrics = compute_correlation_metrics_per_condition(
            corr_df, horizon_filter, summary,
        )
        if not corr_metrics.empty:
            for _, row in corr_metrics.iterrows():
                c = row["condition"]
                if c in table_data:
                    for col in corr_metrics.columns:
                        if col != "condition":
                            table_data[c][col] = row[col]

        # D) Noise metrics
        summary.append("  D) Noise metrics:")
        noise_metrics = compute_noise_metrics_per_condition(noise_df, summary)
        if not noise_metrics.empty:
            for _, row in noise_metrics.iterrows():
                c = row["condition"]
                if c in table_data:
                    for col in noise_metrics.columns:
                        if col != "condition":
                            table_data[c][col] = row[col]

        # E) Training metrics
        summary.append("  E) Training metrics:")
        training_metrics = compute_training_metrics_per_condition(
            tb_dict, training_tail_window, summary,
        )
        if not training_metrics.empty:
            for _, row in training_metrics.iterrows():
                c = row["condition"]
                if c in table_data:
                    for col in training_metrics.columns:
                        if col != "condition":
                            table_data[c][col] = row[col]

        # Build DataFrame
        table_df = pd.DataFrame(list(table_data.values()))
        table_df = table_df.sort_values("condition").reset_index(drop=True)

        # Save CSV
        suffix_parts = [f"task-{task}", f"hgt{horizon_filter}"]
        if mode != "both":
            suffix_parts.append(mode)
        suffix = "__" + "__".join(suffix_parts)
        filename = f"summary_metrics{suffix}.csv"
        filepath = tables_dir / filename
        table_df.to_csv(filepath, index=False, float_format="%.6f")
        summary.append(f"  Saved: {filepath.name} ({len(table_df)} rows x {len(table_df.columns)} cols)")
        saved_paths.append(filepath)
        summary.append("")

    summary.append(f"Generated {len(saved_paths)} summary table(s)")
    return saved_paths
