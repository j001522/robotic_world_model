"""
sanity.py — Data quality checks run before plotting.

Each check function returns a list of warning strings (empty = OK).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Contiguous horizons starting at 1
# ---------------------------------------------------------------------------
def check_contiguous_horizons(
    df: pd.DataFrame,
    horizon_col: str = "horizon_t",
    group_col: str = "condition",
) -> list[str]:
    """Warn if horizons are not contiguous integers starting at 1."""
    warnings: list[str] = []
    if horizon_col not in df.columns:
        return warnings
    for cond, grp in df.groupby(group_col):
        hs = sorted(grp[horizon_col].dropna().unique())
        if not hs:
            continue
        expected = list(range(1, len(hs) + 1))
        int_hs = [int(h) for h in hs]
        if int_hs != expected:
            warnings.append(
                f"[{cond}] horizons not contiguous from 1: "
                f"got {int_hs[:5]}{'...' if len(int_hs) > 5 else ''}"
            )
    return warnings


# ---------------------------------------------------------------------------
# AUROC: warn if n_positive < 5
# ---------------------------------------------------------------------------
def check_auroc_n_positive(df: pd.DataFrame, threshold: int = 5) -> list[str]:
    warnings: list[str] = []
    if "n_positive" not in df.columns:
        return warnings
    low = df[df["n_positive"] < threshold]
    if not low.empty:
        for _, row in low.iterrows():
            cond = row.get("condition", "?")
            seed = row.get("seed", "?")
            n = row["n_positive"]
            warnings.append(
                f"AUROC: n_positive={n} < {threshold} for {cond} seed={seed}"
            )
    return warnings


# ---------------------------------------------------------------------------
# Risk-coverage: coverage_fraction monotonic and within (0, 1]
# ---------------------------------------------------------------------------
def check_risk_coverage(
    df: pd.DataFrame,
    group_col: str = "condition",
) -> list[str]:
    warnings: list[str] = []
    if "coverage_fraction" not in df.columns:
        return warnings
    for cond, grp in df.groupby(group_col):
        cov = grp.sort_values("uncertainty_percentile")["coverage_fraction"].values
        # Check range
        if np.any(cov <= 0) or np.any(cov > 1.0 + 1e-9):
            warnings.append(
                f"[{cond}] coverage_fraction outside (0, 1]: "
                f"min={cov.min():.4f}, max={cov.max():.4f}"
            )
        # Check monotonicity (coverage should increase with percentile)
        diffs = np.diff(cov)
        n_decrease = int(np.sum(diffs < -1e-9))
        if n_decrease > 0:
            warnings.append(
                f"[{cond}] coverage_fraction non-monotonic: "
                f"{n_decrease} decreasing steps"
            )
    return warnings


# ---------------------------------------------------------------------------
# Hallucination: warn if median < 10 or always == max horizon
# ---------------------------------------------------------------------------
def check_hallucination_stats(
    df: pd.DataFrame,
    horizon_col: str = "hallucination_horizon",
    group_col: str = "condition",
) -> list[str]:
    warnings: list[str] = []
    if horizon_col not in df.columns:
        return warnings
    for cond, grp in df.groupby(group_col):
        vals = grp[horizon_col].dropna()
        if vals.empty:
            continue
        med = vals.median()
        if med < 10:
            warnings.append(
                f"[{cond}] hallucination median={med:.1f} < 10 "
                f"(model may be very unstable)"
            )
        max_h = vals.max()
        if (vals == max_h).all():
            warnings.append(
                f"[{cond}] all hallucination horizons == {max_h} "
                f"(threshold may be too lenient)"
            )
    return warnings


# ---------------------------------------------------------------------------
# Correlation: warn about very weak or very few samples
# ---------------------------------------------------------------------------
def check_correlations(df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if "n_samples" in df.columns:
        low = df[df["n_samples"] < 10]
        if not low.empty:
            warnings.append(
                f"correlation: {len(low)} rows with n_samples < 10"
            )
    return warnings


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def run_all_checks(
    datasets: dict[str, pd.DataFrame | None],
) -> list[str]:
    """Run all applicable sanity checks on the loaded datasets.

    Parameters
    ----------
    datasets : dict
        Mapping of dataset name to DataFrame (or None if not loaded).

    Returns
    -------
    List of warning strings.
    """
    all_warnings: list[str] = []

    # Hallucination horizon
    df = datasets.get("hallucination_horizon")
    if df is not None:
        all_warnings.extend(check_contiguous_horizons(df))

    # Error vs uncertainty
    df = datasets.get("error_vs_uncertainty")
    if df is not None:
        all_warnings.extend(
            check_contiguous_horizons(df, horizon_col="prediction_step")
        )

    # AUROC
    df = datasets.get("auroc")
    if df is not None:
        all_warnings.extend(check_auroc_n_positive(df))

    # Risk coverage
    df = datasets.get("risk_coverage")
    if df is not None:
        all_warnings.extend(check_risk_coverage(df))

    # Hallucination stats (per-episode)
    df = datasets.get("hallucination_stats_per_ep")
    if df is not None:
        all_warnings.extend(check_hallucination_stats(df))

    # Correlations
    df = datasets.get("correlation_by_horizon")
    if df is not None:
        all_warnings.extend(check_correlations(df))

    for w in all_warnings:
        log.warning("Sanity: %s", w)

    return all_warnings
