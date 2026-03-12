#!/usr/bin/env python3
"""
Build IROS-ready paper summary tables from raw evaluation CSVs.

Enforces fairness rules:
  - Hallucination metrics ONLY from shared-threshold data (else NaN)
  - AUROC min only from per-horizon data (else NaN for aggregated)
  - Horizon-filtered uncertainty metrics where supported

Outputs:
  tables/paper_table_main.csv      (4 selected conditions)
  tables/paper_table_appendix.csv  (all conditions)
  tables/run_metadata.json

Usage:
  python make_paper_table.py \\
      --results_dir ../results/paper \\
      --output_dir  ../results/paper/figures_out \\
      --rp_main rp-pen025-std
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── logging setup ────────────────────────────────────────────────────────────
log = logging.getLogger("paper_table")

# numpy compat: trapezoid (>=2.0) vs trapz (older)
_integrate = getattr(np, "trapezoid", None) or np.trapz

# ── condition short-names for the paper ──────────────────────────────────────
CONDITION_SHORT: Dict[str, str] = {
    "bs-nopen": "B0",
    "bs-pen025": "Bp",
    "rp-nopen-std": "R0",
    "rp-pen006-std": "Rp006",
    "rp-pen025-std": "Rp025",
    "rp-pen025-var": "Rpvar",
}

# Main-table conditions (the 4th is chosen by --rp_main)
_MAIN_FIXED = ["bs-nopen", "bs-pen025", "rp-nopen-std"]

# Tensorboard filename → standard condition name
_TB_CONDITION_MAP: Dict[str, str] = {
    "finetune-bs-nopen": "bs-nopen",
    "finetune-bs-pen0.25": "bs-pen025",
    "finetune-rp-nopen-std": "rp-nopen-std",
    "finetune-rp-pen0.06-std": "rp-pen006-std",
    "finetune-rp-pen0.25-std": "rp-pen025-std",
    "finetune-rp-pen0.25-var": "rp-pen025-var",
}

# Family key → sub-directory under results_dir
_FAMILY_DIRS: Dict[str, str] = {
    "hallucination_horizon": "hallucination_horizon",
    "hallucination_stats": "hallucination_stats",
    "noise_robustness": "noise_robustness",
    "risk_coverage": "risk_coverage",
    "auroc": "auroc",
    "correlation_by_horizon": "correlation_by_horizon",
    "horizon_auc": "horizon_auc",
}


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Trapezoidal AUC; returns NaN if fewer than 2 points."""
    if len(x) < 2:
        return np.nan
    return float(_integrate(y, x))


def _to_float(series) -> np.ndarray:
    """Series/array → float64 numpy array."""
    return np.asarray(pd.to_numeric(pd.Series(series), errors="coerce"), dtype=float)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column name in *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _horizon_col(df: pd.DataFrame) -> Optional[str]:
    return _find_col(df, ["horizon_t", "prediction_step", "horizon", "t_horizon"])


def _run_cols(df: pd.DataFrame) -> List[str]:
    """Groupby columns defining a single 'run'."""
    return [c for c in ["condition", "seed", "checkpoint_step"] if c in df.columns]


def _closest_row(group: pd.DataFrame, col: str, target: float):
    """Index label of the row closest to *target* in *col*."""
    v = _to_float(group[col])
    mask = np.isfinite(v)
    if not mask.any():
        return None
    return group.index[int(np.argmin(np.abs(v[mask] - target)))]


def _max_row(group: pd.DataFrame, col: str):
    v = _to_float(group[col])
    mask = np.isfinite(v)
    if not mask.any():
        return None
    return group.index[int(np.argmax(v[mask]))]


def _per_run(df: pd.DataFrame, fn: Callable) -> pd.DataFrame:
    """Apply *fn* to each (condition, seed, checkpoint_step) group."""
    cols = _run_cols(df)
    rows: List[dict] = []
    for keys, g in df.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(cols, keys))
        row.update(fn(g))
        rows.append(row)
    return pd.DataFrame(rows)


def _agg_runs(run_df: pd.DataFrame, value_cols: List[str],
              std_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Aggregate per-run DataFrame → per-condition mean (±std for selected cols)."""
    present = [c for c in value_cols if c in run_df.columns]
    if not present:
        return pd.DataFrame(columns=["condition"])
    out = run_df.groupby("condition")[present].mean().reset_index()
    if std_cols:
        s_present = [c for c in std_cols if c in run_df.columns]
        if s_present:
            sdf = run_df.groupby("condition")[s_present].std(ddof=1).reset_index()
            sdf = sdf.rename(columns={c: f"{c}_std" for c in s_present})
            out = out.merge(sdf, on="condition", how="left")
    return out


def _merge(left: Optional[pd.DataFrame],
           right: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Left-merge on 'condition', handling None inputs and duplicate cols."""
    if right is None or (isinstance(right, pd.DataFrame) and right.empty):
        return left
    if left is None:
        return right
    dup = [c for c in right.columns if c in left.columns and c != "condition"]
    if dup:
        right = right.drop(columns=dup)
    return left.merge(right, on="condition", how="outer")


def _dedup_by_run(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate rows caused by loading both _all and per-condition CSVs."""
    cols = _run_cols(df)
    other_cols = [c for c in df.columns if c not in cols]
    return df.drop_duplicates(subset=cols + other_cols[:3], keep="first").reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
#  CSV discovery & loading
# ═════════════════════════════════════════════════════════════════════════════

def _condition_from_filename(filepath: str, family_key: str) -> Optional[str]:
    """Infer condition from a CSV filename (used for CSVs without a condition column)."""
    base = os.path.splitext(os.path.basename(filepath))[0]
    if family_key == "tensorboard_aggregated":
        return _TB_CONDITION_MAP.get(base, base)
    # strip known prefixes: 'auroc_rp-pen025-std' → 'rp-pen025-std'
    prefixes = [family_key + "_", "hallucination_horizon_",
                "hallucination_per_episode_", "hallucination_distribution_"]
    for pfx in prefixes:
        if base.startswith(pfx):
            return base[len(pfx):]
    return None


def _load_family(results_dir: str, family_key: str, subdir: str,
                 filename_filter: Optional[str] = None,
                 ) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load CSVs from results_dir/subdir/*.csv, concatenate, and return.

    Strategy: if an *_all.csv exists and has a 'condition' column with
    per-condition rows, use it alone (avoids duplication from per-condition
    files).  Otherwise load all matching files and deduplicate.
    """
    family_dir = os.path.join(results_dir, subdir)
    if not os.path.isdir(family_dir):
        return None, []
    files = sorted(glob(os.path.join(family_dir, "*.csv")))
    if filename_filter:
        rx = re.compile(filename_filter)
        files = [f for f in files if rx.search(os.path.basename(f))]
    if not files:
        return None, []

    # Try the _all file first — it typically contains all conditions already
    all_files = [f for f in files if "_all" in os.path.basename(f)]
    if all_files:
        try:
            df_all = pd.read_csv(all_files[0])
            if "condition" in df_all.columns and df_all["condition"].nunique() > 1:
                df_all = df_all[df_all["condition"] != "all"].reset_index(drop=True)
                df_all = df_all.drop_duplicates().reset_index(drop=True)
                return df_all, all_files
        except Exception:
            pass  # fall through to loading all files

    # Fallback: load every file
    frames: List[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
            if "condition" not in df.columns:
                cond = _condition_from_filename(fp, family_key)
                if cond:
                    if len(df.columns) > 50:
                        df = df.copy()  # defragment wide DFs
                    df["condition"] = cond
            frames.append(df)
        except Exception as exc:
            log.warning("Failed to read %s: %s", fp, exc)
    if not frames:
        return None, files
    merged = pd.concat(frames, ignore_index=True)
    if "condition" not in merged.columns:
        merged["condition"] = "unknown"
    # Exclude 'all' rows and deduplicate
    if "all" in merged["condition"].values:
        merged = merged[merged["condition"] != "all"].reset_index(drop=True)
    merged = _dedup_by_run(merged)
    return merged, files


def _log_load(name: str, files: List[str], df: Optional[pd.DataFrame]) -> None:
    n = len(files)
    if df is None:
        log.info("  %-30s  %d files, no rows", name, n)
    else:
        conds = sorted(df["condition"].dropna().unique().tolist())
        log.info("  %-30s  %d files, %d rows, conditions=%s", name, n, len(df), conds)


# ═════════════════════════════════════════════════════════════════════════════
#  Metric computation — one function per family
# ═════════════════════════════════════════════════════════════════════════════

# ── A) Horizon reliability ───────────────────────────────────────────────────

def metric_error_at_horizon(df: pd.DataFrame, target_horizon: int) -> pd.DataFrame:
    """error_at_h{target}_mean from hallucination_horizon data."""
    hcol = _horizon_col(df)
    ecol = _find_col(df, ["error_rel_mean"])
    if not hcol or not ecol:
        log.warning("hallucination_horizon: missing horizon (%s) or error (%s) column", hcol, ecol)
        return pd.DataFrame(columns=["condition"])

    def fn(g: pd.DataFrame) -> dict:
        g = g.copy()
        g[hcol] = pd.to_numeric(g[hcol], errors="coerce")
        g[ecol] = pd.to_numeric(g[ecol], errors="coerce")
        idx = _closest_row(g, hcol, target_horizon)
        if idx is None:
            idx = _max_row(g, hcol)
        err = float(g.loc[idx, ecol]) if idx is not None else np.nan
        return {"error_at_target": err}

    run_df = _per_run(df, fn)
    agg = _agg_runs(run_df, ["error_at_target"])
    return agg.rename(columns={"error_at_target": f"error_at_h{target_horizon}_mean"})


def metric_horizon_auc(df: pd.DataFrame) -> pd.DataFrame:
    """horizon_auc_rel_mean from pre-computed horizon_auc folder."""
    if "horizon_auc_rel" not in df.columns:
        log.warning("horizon_auc: missing horizon_auc_rel column")
        return pd.DataFrame(columns=["condition"])

    def fn(g: pd.DataFrame) -> dict:
        v = _to_float(g["horizon_auc_rel"])
        v = v[np.isfinite(v)]
        return {"horizon_auc_rel": float(np.mean(v)) if len(v) else np.nan}

    run_df = _per_run(df, fn)
    agg = _agg_runs(run_df, ["horizon_auc_rel"])
    return agg.rename(columns={"horizon_auc_rel": "horizon_auc_rel_mean"})


def metric_horizon_auc_from_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: compute horizon AUC from error-vs-horizon curve if horizon_auc/ missing."""
    hcol = _horizon_col(df)
    ecol = _find_col(df, ["error_rel_mean"])
    if not hcol or not ecol:
        return pd.DataFrame(columns=["condition"])

    def fn(g: pd.DataFrame) -> dict:
        g = g.copy()
        g[hcol] = pd.to_numeric(g[hcol], errors="coerce")
        g[ecol] = pd.to_numeric(g[ecol], errors="coerce")
        s = g.dropna(subset=[hcol, ecol]).sort_values(hcol)
        auc = _trapz_auc(s[hcol].to_numpy(), s[ecol].to_numpy())
        return {"horizon_auc_rel": auc}

    run_df = _per_run(df, fn)
    agg = _agg_runs(run_df, ["horizon_auc_rel"])
    return agg.rename(columns={"horizon_auc_rel": "horizon_auc_rel_mean"})


# ── B) Hallucination (shared-threshold enforcement) ──────────────────────────

def metric_hallucination(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Compute hallucination metrics ONLY if threshold is shared.

    Returns:
        (metric_df, threshold_is_shared)
    """
    if "hallucination_horizon" not in df.columns:
        log.warning("hallucination_stats: missing hallucination_horizon column")
        return pd.DataFrame(columns=["condition"]), False

    # ── Check threshold method across all rows ──
    methods = set()
    if "threshold_method" in df.columns:
        methods = set(df["threshold_method"].dropna().unique())

    has_shared = any("shared" in str(m).lower() for m in methods)

    if not has_shared:
        log.warning(
            "hallucination_stats: NO shared threshold found. "
            "threshold_methods present: %s. "
            "Hallucination metrics will be NaN (diagnostic_only=True).",
            methods,
        )
        # Return skeleton with NaN values + flags
        conds = sorted(df["condition"].dropna().unique())
        skeleton = pd.DataFrame({
            "condition": conds,
            "halluc_h_median": np.nan,
            "halluc_h_iqr": np.nan,
            "halluc_threshold_value": np.nan,
            "halluc_threshold_method": [str(next(iter(methods))) if methods else ""] * len(conds),
            "halluc_threshold_shared": False,
        })
        return skeleton, False

    # ── Shared threshold exists: filter to shared-only rows ──
    shared_mask = df["threshold_method"].str.lower().str.contains("shared", na=False)
    df_shared = df[shared_mask].copy()
    if df_shared.empty:
        log.warning("hallucination_stats: shared mask matched 0 rows despite methods=%s", methods)
        return pd.DataFrame(columns=["condition"]), False

    df_shared["hallucination_horizon"] = pd.to_numeric(
        df_shared["hallucination_horizon"], errors="coerce"
    )

    rows: List[dict] = []
    for cond, g in df_shared.groupby("condition"):
        vals = g["hallucination_horizon"].dropna().to_numpy()
        if len(vals) == 0:
            median = iqr = np.nan
        else:
            median = float(np.nanmedian(vals))
            iqr = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))

        thr_vals = g["threshold"].dropna().unique() if "threshold" in g.columns else []
        meth_vals = g["threshold_method"].dropna().unique() if "threshold_method" in g.columns else []
        rows.append({
            "condition": cond,
            "halluc_h_median": median,
            "halluc_h_iqr": iqr,
            "halluc_threshold_value": float(thr_vals[0]) if len(thr_vals) else np.nan,
            "halluc_threshold_method": str(meth_vals[0]) if len(meth_vals) else "",
            "halluc_threshold_shared": True,
        })

    return pd.DataFrame(rows), True


# ── C) Noise robustness ─────────────────────────────────────────────────────

def metric_noise(df: pd.DataFrame) -> pd.DataFrame:
    """noise_auc_mean and error_at_noise_max_mean."""
    noise_col = _find_col(df, ["noise_level", "noise", "sigma"])
    err_col = _find_col(df, ["error_mean", "error_rel_mean", "error"])
    if not noise_col or not err_col:
        log.warning("noise_robustness: missing noise (%s) or error (%s) column", noise_col, err_col)
        return pd.DataFrame(columns=["condition"])

    def fn(g: pd.DataFrame) -> dict:
        g = g.copy()
        g[noise_col] = pd.to_numeric(g[noise_col], errors="coerce")
        g[err_col] = pd.to_numeric(g[err_col], errors="coerce")
        s = g.dropna(subset=[noise_col, err_col]).sort_values(noise_col)
        auc = _trapz_auc(s[noise_col].to_numpy(), s[err_col].to_numpy())
        idx = _max_row(g, noise_col)
        err_max = float(g.loc[idx, err_col]) if idx is not None else np.nan
        return {"noise_auc": auc, "error_at_noise_max": err_max}

    run_df = _per_run(df, fn)
    agg = _agg_runs(run_df, ["noise_auc", "error_at_noise_max"])
    return agg.rename(columns={
        "noise_auc": "noise_auc_mean",
        "error_at_noise_max": "error_at_noise_max_mean",
    })


# ── D) Risk-coverage ────────────────────────────────────────────────────────

def metric_risk_coverage(df: pd.DataFrame, target_coverage: float,
                         horizon_filter: int) -> pd.DataFrame:
    """risk_at_80cov_mean/std and risk_at_100cov_mean."""
    cov_col = _find_col(df, ["coverage_fraction", "coverage"])
    risk_col = _find_col(df, ["mean_error_filtered", "mean_error", "risk"])
    if not cov_col or not risk_col:
        log.warning("risk_coverage: missing coverage (%s) or risk (%s) column", cov_col, risk_col)
        return pd.DataFrame(columns=["condition"])

    # Check for horizon metadata
    hcol = _horizon_col(df)
    if hcol:
        log.info("risk_coverage: horizon column '%s' found; filtering > %d", hcol, horizon_filter)
    else:
        log.warning(
            "risk_coverage: no horizon column — data was pre-aggregated over all horizons. "
            "Cannot enforce horizon_filter=%d. Proceeding with unfiltered data.",
            horizon_filter,
        )

    def fn(g: pd.DataFrame) -> dict:
        g = g.copy()
        if hcol:
            g[hcol] = pd.to_numeric(g[hcol], errors="coerce")
            g = g[g[hcol] > horizon_filter]
        g[cov_col] = pd.to_numeric(g[cov_col], errors="coerce")
        g[risk_col] = pd.to_numeric(g[risk_col], errors="coerce")
        if g.empty:
            return {"risk_at_cov": np.nan, "risk_at_100cov": np.nan}
        i_cov = _closest_row(g, cov_col, target_coverage)
        i_100 = _closest_row(g, cov_col, 1.0)
        return {
            "risk_at_cov": float(g.loc[i_cov, risk_col]) if i_cov is not None else np.nan,
            "risk_at_100cov": float(g.loc[i_100, risk_col]) if i_100 is not None else np.nan,
        }

    run_df = _per_run(df, fn)
    agg = _agg_runs(run_df, ["risk_at_cov", "risk_at_100cov"], std_cols=["risk_at_cov"])
    return agg.rename(columns={
        "risk_at_cov": "risk_at_80cov_mean",
        "risk_at_cov_std": "risk_at_80cov_std",
        "risk_at_100cov": "risk_at_100cov_mean",
    })


# ── D cont.) AUROC ──────────────────────────────────────────────────────────

def metric_auroc(df: pd.DataFrame, horizon_filter: int) -> pd.DataFrame:
    """
    AUROC computation.

    If per-horizon AUROC exists (horizon_t column):
        per-run: mean & min of AUROC for horizons > horizon_filter
        then aggregate across runs → mean±std.
    If only aggregated AUROC (no horizon_t):
        per-run scalar → aggregate across runs → mean±std.
        auroc_min_hgt set to NaN (cannot fairly compute min from aggregated).
    """
    if "auroc" not in df.columns:
        log.warning("auroc: missing 'auroc' column")
        return pd.DataFrame(columns=["condition"])

    hcol = _horizon_col(df)

    if hcol is not None:
        # ── Per-horizon AUROC ──
        log.info("auroc: per-horizon column '%s' found — computing mean & min for h > %d",
                 hcol, horizon_filter)

        def fn(g: pd.DataFrame) -> dict:
            g = g.copy()
            g[hcol] = pd.to_numeric(g[hcol], errors="coerce")
            g["auroc"] = pd.to_numeric(g["auroc"], errors="coerce")
            g = g[g[hcol] > horizon_filter]
            vals = g["auroc"].dropna().to_numpy()
            return {
                "auroc_mean_hgt": float(np.mean(vals)) if len(vals) else np.nan,
                "auroc_min_hgt": float(np.min(vals)) if len(vals) else np.nan,
            }

        run_df = _per_run(df, fn)
        agg = _agg_runs(run_df, ["auroc_mean_hgt", "auroc_min_hgt"],
                         std_cols=["auroc_mean_hgt"])
        return agg.rename(columns={
            "auroc_mean_hgt": "auroc_mean_hgt_mean",
            "auroc_mean_hgt_std": "auroc_mean_hgt_std",
            "auroc_min_hgt": "auroc_min_hgt_mean",
        })
    else:
        # ── Aggregated AUROC only ──
        log.warning(
            "auroc: NO horizon column — only aggregated AUROC available. "
            "auroc_min_hgt will be NaN (cannot compute min from pre-aggregated data)."
        )

        def fn_agg(g: pd.DataFrame) -> dict:
            vals = _to_float(g["auroc"])
            vals = vals[np.isfinite(vals)]
            return {
                "auroc_mean_hgt": float(np.mean(vals)) if len(vals) else np.nan,
                "auroc_min_hgt": np.nan,  # explicitly NaN: not fair to compute
            }

        run_df = _per_run(df, fn_agg)
        agg = _agg_runs(run_df, ["auroc_mean_hgt", "auroc_min_hgt"],
                         std_cols=["auroc_mean_hgt"])
        return agg.rename(columns={
            "auroc_mean_hgt": "auroc_mean_hgt_mean",
            "auroc_mean_hgt_std": "auroc_mean_hgt_std",
            "auroc_min_hgt": "auroc_min_hgt_mean",
        })


# ═════════════════════════════════════════════════════════════════════════════
#  Main processing
# ═════════════════════════════════════════════════════════════════════════════

# Columns in the final table (order matters for the paper)
_OUTPUT_COLS = [
    "condition",
    "condition_short",
    "error_at_h{target}_mean",        # placeholder — filled at runtime
    "horizon_auc_rel_mean",
    "noise_auc_mean",
    "error_at_noise_max_mean",
    "risk_at_80cov_mean",
    "risk_at_80cov_std",
    "risk_at_100cov_mean",
    "auroc_mean_hgt_mean",
    "auroc_mean_hgt_std",
    "auroc_min_hgt_mean",
    "halluc_h_median",
    "halluc_h_iqr",
    "halluc_threshold_value",
    "halluc_threshold_method",
    "halluc_threshold_shared",
]


def build_table(results_dir: str, args) -> pd.DataFrame:
    """Load all families, compute metrics, merge into one table."""
    log.info("=" * 70)
    log.info("Results root: %s", results_dir)
    log.info("=" * 70)

    # ── Load every family ────────────────────────────────────────────────
    hh_df, hh_f = _load_family(results_dir, "hallucination_horizon",
                                "hallucination_horizon")
    _log_load("hallucination_horizon", hh_f, hh_df)

    hs_df, hs_f = _load_family(results_dir, "hallucination_stats",
                                "hallucination_stats",
                                filename_filter=r"hallucination_per_episode")
    _log_load("hallucination_stats (per_episode)", hs_f, hs_df)

    nr_df, nr_f = _load_family(results_dir, "noise_robustness", "noise_robustness")
    _log_load("noise_robustness", nr_f, nr_df)

    rc_df, rc_f = _load_family(results_dir, "risk_coverage", "risk_coverage")
    _log_load("risk_coverage", rc_f, rc_df)

    au_df, au_f = _load_family(results_dir, "auroc", "auroc")
    _log_load("auroc", au_f, au_df)

    ha_df, ha_f = _load_family(results_dir, "horizon_auc", "horizon_auc")
    _log_load("horizon_auc", ha_f, ha_df)

    # ── Discover all conditions ──────────────────────────────────────────
    all_dfs = [hh_df, hs_df, nr_df, rc_df, au_df, ha_df]
    conditions: set = set()
    for d in all_dfs:
        if d is not None and "condition" in d.columns:
            conditions.update(d["condition"].dropna().unique().tolist())
    conditions.discard("all")
    conditions.discard("unknown")
    if not conditions:
        log.error("No conditions found in any CSV. Aborting.")
        sys.exit(1)

    table = pd.DataFrame({"condition": sorted(conditions)})
    log.info("Conditions found: %s", sorted(conditions))

    # ── A) Horizon reliability ───────────────────────────────────────────
    h_target = args.target_horizon
    err_col_name = f"error_at_h{h_target}_mean"

    if hh_df is not None:
        table = _merge(table, metric_error_at_horizon(hh_df, h_target))

    # Horizon AUC: prefer pre-computed; fallback to curve integration
    if ha_df is not None:
        table = _merge(table, metric_horizon_auc(ha_df))
    elif hh_df is not None:
        log.info("horizon_auc/ not found; computing AUC from error-vs-horizon curve")
        table = _merge(table, metric_horizon_auc_from_curve(hh_df))
    else:
        log.warning("No data for horizon AUC")

    # ── B) Hallucination (shared threshold enforcement) ──────────────────
    if hs_df is not None:
        halluc_df, is_shared = metric_hallucination(hs_df)
        table = _merge(table, halluc_df)
    else:
        log.warning("hallucination_stats: no per-episode data found")

    # ── C) Noise robustness ──────────────────────────────────────────────
    if nr_df is not None:
        table = _merge(table, metric_noise(nr_df))
    else:
        log.warning("noise_robustness: no data")

    # ── D) Risk-coverage ─────────────────────────────────────────────────
    if rc_df is not None:
        table = _merge(table, metric_risk_coverage(rc_df, args.target_coverage,
                                                    args.horizon_filter))
    else:
        log.warning("risk_coverage: no data")

    # ── D cont.) AUROC ───────────────────────────────────────────────────
    if au_df is not None:
        table = _merge(table, metric_auroc(au_df, args.horizon_filter))
    else:
        log.warning("auroc: no data")

    # ── Add condition_short ──────────────────────────────────────────────
    table["condition_short"] = table["condition"].map(
        lambda c: CONDITION_SHORT.get(c, c)
    )

    # ── Ensure all output columns exist ──────────────────────────────────
    expected = [c.replace("{target}", str(h_target)) for c in _OUTPUT_COLS]
    for c in expected:
        if c not in table.columns:
            table[c] = np.nan

    # Reorder: put expected columns first, then any extras
    extras = [c for c in table.columns if c not in expected]
    table = table[[c for c in expected if c in table.columns] + extras]

    return table


def write_outputs(table: pd.DataFrame, output_dir: str, args) -> None:
    """Write main (4-condition) and appendix (all-condition) tables + metadata."""
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # ── Appendix table (all conditions) ──────────────────────────────────
    appendix_path = os.path.join(tables_dir, "paper_table_appendix.csv")
    table.to_csv(appendix_path, index=False)
    log.info("Wrote appendix table: %s  (%d conditions × %d cols)",
             appendix_path, len(table), len(table.columns))

    # ── Main table (4 conditions) ────────────────────────────────────────
    main_conditions = _MAIN_FIXED + [args.rp_main]
    present = [c for c in main_conditions if c in table["condition"].values]
    missing = [c for c in main_conditions if c not in table["condition"].values]
    if missing:
        log.warning("Main table: conditions not found in data: %s", missing)

    main_table = table[table["condition"].isin(present)].copy()
    # Enforce ordering
    order_map = {c: i for i, c in enumerate(main_conditions)}
    main_table["_order"] = main_table["condition"].map(order_map)
    main_table = main_table.sort_values("_order").drop(columns=["_order"])

    main_path = os.path.join(tables_dir, "paper_table_main.csv")
    main_table.to_csv(main_path, index=False)
    log.info("Wrote main table: %s  (%d conditions × %d cols)",
             main_path, len(main_table), len(main_table.columns))

    # ── Metadata JSON ────────────────────────────────────────────────────
    meta = {
        "script": "make_paper_table.py",
        "results_dir": os.path.abspath(args.results_dir),
        "horizon_filter": args.horizon_filter,
        "target_coverage": args.target_coverage,
        "target_horizon": args.target_horizon,
        "rp_main": args.rp_main,
        "main_conditions": main_conditions,
        "all_conditions": sorted(table["condition"].unique().tolist()),
        "halluc_threshold_shared": bool(
            table["halluc_threshold_shared"].any()
            if "halluc_threshold_shared" in table.columns else False
        ),
    }
    meta_path = os.path.join(tables_dir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote metadata: %s", meta_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build IROS-ready paper summary tables from evaluation CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results_dir", required=True,
                        help="Root dir containing family subdirs (e.g. results/paper)")
    parser.add_argument("--output_dir", default=None,
                        help="Output base dir (default: results_dir/figures_out)")
    parser.add_argument("--rp_main", required=True,
                        help="Which Rp condition to include in the main table "
                             "(e.g. rp-pen025-std)")
    parser.add_argument("--horizon_filter", type=int, default=30,
                        help="Horizon filter (>) for AUROC / risk-coverage / correlation")
    parser.add_argument("--target_horizon", type=int, default=200,
                        help="Target horizon step for error_at_h metric")
    parser.add_argument("--target_coverage", type=float, default=0.8,
                        help="Target coverage fraction for risk metric")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    results_dir = os.path.abspath(args.results_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else \
        os.path.join(results_dir, "figures_out")

    # Validate rp_main
    all_known = set(CONDITION_SHORT.keys())
    if args.rp_main not in all_known:
        log.warning("--rp_main='%s' is not in known conditions %s. Will proceed anyway.",
                     args.rp_main, sorted(all_known))

    table = build_table(results_dir, args)

    # ── Print summary to console ─────────────────────────────────────────
    print()
    print(table.to_string(index=False))
    print()

    write_outputs(table, output_dir, args)


if __name__ == "__main__":
    main()
