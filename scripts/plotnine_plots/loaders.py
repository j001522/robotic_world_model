"""
loaders.py — Load DataFrames with source-of-truth precedence rules.

Each public ``load_*`` function returns a DataFrame (or None) and appends
provenance info to *summary_lines*.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .discovery import Inventory, CSVEntry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Condition filtering
# ---------------------------------------------------------------------------
def _filter_conditions(
    df: pd.DataFrame,
    include: Optional[list[str]],
    exclude: Optional[list[str]],
) -> pd.DataFrame:
    if "condition" not in df.columns:
        return df
    if include:
        df = df[df["condition"].isin(include)]
    if exclude:
        df = df[~df["condition"].isin(exclude)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _read_entry(
    entry: Optional[CSVEntry],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    if entry is None:
        return None
    try:
        df = pd.read_csv(entry.path)
        df = _filter_conditions(df, include, exclude)
        if df.empty:
            return None
        return df
    except Exception as exc:
        log.warning("Failed to read %s: %s", entry.path, exc)
        return None


def _pick_all_csv(inv: Inventory, family: str) -> Optional[CSVEntry]:
    """Prefer *_all.csv inside the family; fall back to first entry."""
    return inv.best_for_family(family)


# ---------------------------------------------------------------------------
# Public loaders (one per logical dataset)
# ---------------------------------------------------------------------------
def load_hallucination_horizon(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """Long-horizon error curves.  Source: hallucination_horizon/ first,
    then error_vs_uncertainty/ as fallback."""
    entry = _pick_all_csv(inv, "hallucination_horizon")
    source = "hallucination_horizon"
    if entry is None:
        entry = _pick_all_csv(inv, "error_vs_uncertainty")
        source = "error_vs_uncertainty (fallback)"
    df = _read_entry(entry, include, exclude)
    if df is not None:
        summary.append(f"long_horizon_error: {entry.path}  (source={source})")
    else:
        summary.append("long_horizon_error: NOT FOUND")
    return df


def load_error_vs_uncertainty(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    """Uncertainty vs horizon data.  Source: error_vs_uncertainty/ only."""
    entry = _pick_all_csv(inv, "error_vs_uncertainty")
    df = _read_entry(entry, include, exclude)
    if df is not None:
        summary.append(f"error_vs_uncertainty: {entry.path}")
    else:
        summary.append("error_vs_uncertainty: NOT FOUND")
    return df


def load_correlation_by_horizon(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    entry = _pick_all_csv(inv, "correlation_by_horizon")
    df = _read_entry(entry, include, exclude)
    if df is not None:
        summary.append(f"correlation_by_horizon: {entry.path}")
    else:
        summary.append("correlation_by_horizon: NOT FOUND")
    return df


def load_noise_robustness(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    entry = _pick_all_csv(inv, "noise_robustness")
    df = _read_entry(entry, include, exclude)
    if df is not None:
        summary.append(f"noise_robustness: {entry.path}")
    else:
        summary.append("noise_robustness: NOT FOUND")
    return df


def load_auroc(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    entry = _pick_all_csv(inv, "auroc")
    df = _read_entry(entry, include, exclude)
    if df is not None:
        summary.append(f"auroc: {entry.path}")
    else:
        summary.append("auroc: NOT FOUND")
    return df


def load_risk_coverage(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    entry = _pick_all_csv(inv, "risk_coverage")
    df = _read_entry(entry, include, exclude)
    if df is not None:
        summary.append(f"risk_coverage: {entry.path}")
    else:
        summary.append("risk_coverage: NOT FOUND")
    return df


def load_horizon_auc(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    # horizon_auc often has no _all file; concat all per-condition files
    entries = inv.all_for_family("horizon_auc")
    if not entries:
        summary.append("horizon_auc: NOT FOUND")
        return None
    frames = []
    paths = []
    for e in entries:
        df = _read_entry(e, include, exclude)
        if df is not None:
            frames.append(df)
            paths.append(str(e.path))
    if not frames:
        summary.append("horizon_auc: NOT FOUND (after filtering)")
        return None
    combined = pd.concat(frames, ignore_index=True)
    if "condition" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["condition", "seed", "checkpoint_step"], keep="first"
        )
    summary.append(f"horizon_auc: {len(paths)} files combined")
    return combined


def load_growth_rates(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    # Prefer growth_rates_all.csv (has growth-rate columns)
    entries = inv.all_for_family("growth_rates")
    for e in entries:
        if "growth_rates_all" in e.path.stem:
            df = _read_entry(e, include, exclude)
            if df is not None:
                summary.append(f"growth_rates: {e.path}")
                return df
    # Fallback to any all file
    entry = _pick_all_csv(inv, "growth_rates")
    df = _read_entry(entry, include, exclude)
    if df is not None and entry is not None:
        summary.append(f"growth_rates: {entry.path}")
    else:
        summary.append("growth_rates: NOT FOUND")
    return df


def load_hallucination_stats(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Return (per_episode_df, distribution_df)."""
    entries = inv.all_for_family("hallucination_stats")
    per_ep: Optional[pd.DataFrame] = None
    distrib: Optional[pd.DataFrame] = None

    for e in entries:
        if "per_episode" in e.path.stem:
            per_ep = _read_entry(e, include, exclude)
            if per_ep is not None:
                summary.append(f"hallucination_stats (per_ep): {e.path}")
        elif "distribution" in e.path.stem:
            distrib = _read_entry(e, include, exclude)
            if distrib is not None:
                summary.append(f"hallucination_stats (distrib): {e.path}")

    # Fallback: hallucination_horizon_all in stats folder
    if per_ep is None:
        for e in entries:
            if "hallucination_horizon" in e.path.stem and "distribution" not in e.path.stem:
                per_ep = _read_entry(e, include, exclude)
                if per_ep is not None:
                    summary.append(f"hallucination_stats (per_ep fallback): {e.path}")
                    break

    if per_ep is None and distrib is None:
        summary.append("hallucination_stats: NOT FOUND")
    return per_ep, distrib


def load_tensorboard_aggregated(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[dict[str, pd.DataFrame]]:
    """Return dict mapping condition → DataFrame from aggregated TensorBoard CSVs."""
    from .transforms import _LABEL_MAP

    entries = inv.all_for_family("tensorboard")
    agg_entries = [e for e in entries if e.is_aggregated]
    if not agg_entries:
        summary.append("tensorboard (aggregated): NOT FOUND")
        return None

    # Build reverse mapping: short_label -> list of full condition names
    short_to_full: dict[str, list[str]] = {}
    for full_cond, short_label in _LABEL_MAP.items():
        short_to_full.setdefault(short_label, []).append(full_cond)

    # Build exclude set including full condition names that map to the same short label
    exclude_full = set(exclude) if exclude else set()
    if exclude:
        for excl_pattern in exclude:
            # Check if pattern is a full condition name in _LABEL_MAP
            if excl_pattern in _LABEL_MAP:
                short_label = _LABEL_MAP[excl_pattern]
                exclude_full.update(short_to_full.get(short_label, []))
            # Also check if pattern is itself a short label
            elif excl_pattern in short_to_full:
                exclude_full.update(short_to_full[excl_pattern])

    # Build include set including full condition names that map to the same short label
    include_full = set(include) if include else None
    if include:
        include_full = set(include)
        for incl_pattern in include:
            # Check if pattern is a full condition name in _LABEL_MAP
            if incl_pattern in _LABEL_MAP:
                short_label = _LABEL_MAP[incl_pattern]
                include_full.update(short_to_full.get(short_label, []))
            # Also check if pattern is itself a short label
            elif incl_pattern in short_to_full:
                include_full.update(short_to_full[incl_pattern])

    result: dict[str, pd.DataFrame] = {}
    for e in agg_entries:
        df = _read_entry(e, include=None, exclude=None)
        if df is None:
            continue
        # Condition from filename (e.g. finetune-bs-pen0.25.csv)
        cond = e.path.stem
        # Apply condition filters
        if include_full and cond not in include_full:
            continue
        if cond in exclude_full:
            continue
        result[cond] = df

    if result:
        summary.append(
            f"tensorboard (aggregated): {len(result)} conditions "
            f"[{', '.join(sorted(result.keys()))}]"
        )
    else:
        summary.append("tensorboard (aggregated): NOT FOUND (after filtering)")
    return result or None


def load_faithfulness(
    inv: Inventory,
    summary: list[str],
    include: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
) -> Optional[pd.DataFrame]:
    entry = _pick_all_csv(inv, "faithfulness_gap")
    # Prefer faithfulness_comparison.csv
    entries = inv.all_for_family("faithfulness_gap")
    for e in entries:
        if "comparison" in e.path.stem:
            entry = e
            break
    df = _read_entry(entry, include, exclude)
    if df is not None and entry is not None:
        summary.append(f"faithfulness: {entry.path}")
    else:
        summary.append("faithfulness: NOT FOUND")
    return df
