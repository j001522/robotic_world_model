"""
plot_decision.py — Decision utility plots: risk-coverage and AUROC.

Generates:
    decision_utility/decision_utility_auroc_vs_horizon[__suffix].pdf  (default)
    decision_utility/decision_utility_auroc_summary__hgt30[__suffix].pdf  (optional)
    decision_utility/decision_utility_risk_coverage[__suffix].pdf
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, geom_point, geom_col,
    geom_errorbar, geom_hline, geom_vline, geom_text, labs,
    scale_y_continuous, scale_x_continuous, coord_cartesian,
    annotate,
)

from .transforms import (
    add_short_labels, aggregate_seeds, get_smoothed_column,
    compute_auroc_per_horizon,
)
from .styles import paper_theme, paper_theme_no_title, colour_scales, save_plot

log = logging.getLogger(__name__)


def _build_filename_suffix(
    horizon_filter: Optional[int],
    smooth_mode: str,
    smooth_window: Optional[int],
    ewma_alpha: float,
    mode: Optional[str] = None,
) -> str:
    """Build filename suffix based on configuration."""
    parts = []
    if mode and mode != "full":
        parts.append(mode)
    if horizon_filter is not None:
        parts.append(f"hgt{horizon_filter}")
    if smooth_mode == "rolling_mean" and smooth_window:
        parts.append(f"smooth_rollmean_w{smooth_window}")
    elif smooth_mode == "ewma":
        parts.append(f"smooth_ewma_a{ewma_alpha:.2f}")
    return "__" + "__".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Risk-coverage
# ---------------------------------------------------------------------------
def _plot_risk_coverage(
    df: pd.DataFrame,
    output_dir: Path,
    horizon_filter: Optional[int],
    smooth_mode: str,
    smooth_window: Optional[int],
    ewma_alpha: float,
    coverage_target: float = 0.80,
    paper: bool = False,
    paper_font_size: int = 16,
) -> list[Path]:
    """Mean error vs coverage fraction, one line per condition.

    Risk-coverage is a selective-prediction (abstention) curve:
    - x-axis: Coverage = fraction of samples retained (1 = keep all)
    - y-axis: Risk = mean prediction error on retained subset
    - Samples are ranked by epistemic uncertainty; lowest-uncertainty
      samples are kept first.
    """
    df = df.copy()

    # Aggregate across seeds
    agg = aggregate_seeds(
        df,
        group_cols=["condition", "uncertainty_percentile"],
        value_cols=["mean_error_filtered", "coverage_fraction"],
        agg="mean_std",
    )

    # Resolve column names after aggregation
    x_col_base = "coverage_fraction_mean" if "coverage_fraction_mean" in agg.columns else "coverage_fraction"
    y_col_base = "mean_error_filtered_mean" if "mean_error_filtered_mean" in agg.columns else "mean_error_filtered"
    y_std = "mean_error_filtered_std" if "mean_error_filtered_std" in agg.columns else None

    # Apply smoothing only to the y-axis (risk), not to coverage
    group_cols = ["condition"]
    if smooth_mode != "none":
        window = smooth_window or 10
        agg, y_col = get_smoothed_column(agg, y_col_base, group_cols, smooth_mode, window, ewma_alpha)
    else:
        y_col = y_col_base

    # Std ribbon
    if y_std and y_std in agg.columns:
        agg = agg.rename(columns={y_std: "y_std"})
        agg["y_std"] = agg["y_std"].fillna(0)
    else:
        agg["y_std"] = 0.0
    agg["ymin"] = agg[y_col] - agg["y_std"]
    agg["ymax"] = agg[y_col] + agg["y_std"]

    # Sort by coverage within each condition
    agg = agg.sort_values(["condition", x_col_base])

    agg = add_short_labels(agg)
    conds = sorted(set(agg["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    # Get no-filtering risk (at coverage ~ 1.0) for each condition
    no_filter_risk = {}
    for c in conds:
        cond_data = agg[agg["condition_short"] == c]
        if not cond_data.empty:
            max_cov_idx = cond_data[x_col_base].to_numpy().argmax()
            no_filter_risk[c] = cond_data.iloc[max_cov_idx][y_col]

    # Build caption
    caption_parts = [
        "Risk-Coverage Curve (Selective Prediction / Abstention)",
        "",
        "X-axis: Coverage (fraction of samples retained).",
        "  Coverage=1 means keep all samples (no filtering).",
        "  Coverage decreases as more high-uncertainty samples are removed.",
        "",
        "Y-axis: Risk = mean prediction error on retained subset.",
        "  Filtering ranks samples by epistemic uncertainty within each condition.",
        "  Lower risk at a given coverage indicates better uncertainty quality.",
        "",
        f"Vertical dashed line: coverage={coverage_target:.0%} (common operating point).",
    ]
    if horizon_filter is not None:
        caption_parts.append(f"Note: Risk-coverage data is pre-computed; horizon filter not applied here.")
    caption_parts.append("")
    caption_parts.append("No-filtering risk (coverage=1.0) per condition:")
    for c, r in sorted(no_filter_risk.items()):
        caption_parts.append(f"  {c}: {r:.4f}")
    caption = "\n".join(caption_parts)

    # Build filename
    suffix = _build_filename_suffix(horizon_filter, smooth_mode, smooth_window, ewma_alpha)
    filename = f"decision_utility_risk_coverage{suffix}"

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    labs_kwargs = {
        "title": "" if paper else "Risk-Coverage Curve",
        "subtitle": "" if paper else (
            "Risk = mean prediction error on retained subset | "
            "Filtering by lowest epistemic uncertainty (ranking-based)"
        ),
        "x": "Coverage (fraction kept)",
        "y": "Risk (mean prediction error)",
        "colour": "" if paper else "Condition",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(agg, aes(x=x_col_base, y=y_col, colour="condition_short",
                        fill="condition_short"))
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.15, colour="none")
        + geom_line(size=1.0)
        + geom_vline(xintercept=coverage_target, linetype="dashed", colour="grey", size=0.5)
        + col_sc + fill_sc
        + labs(**labs_kwargs)
        + coord_cartesian(xlim=(0, 1.02))
        + theme
    )
    return save_plot(p, output_dir, filename, caption=None if paper else caption)


# ---------------------------------------------------------------------------
# AUROC helpers
# ---------------------------------------------------------------------------
def _check_auroc_sanity(df: pd.DataFrame) -> None:
    """Check AUROC sanity conditions and warn if needed."""
    if "n_positive" in df.columns:
        low_pos = df[df["n_positive"] < 5]
        if not low_pos.empty:
            if "horizon_t" in low_pos.columns:
                horizons_affected = sorted(set(low_pos["horizon_t"].tolist()))
            else:
                horizons_affected = ["N/A"]
            log.warning(
                "AUROC sanity: %d rows with n_positive < 5 (AUROC unstable); "
                "affected horizons: %s",
                len(low_pos),
                horizons_affected[:10],
            )
    else:
        log.warning("AUROC sanity: 'n_positive' column not found; cannot verify label counts")


def _compute_or_load_auroc(
    auroc_df: Optional[pd.DataFrame],
    eu_df: Optional[pd.DataFrame],
    top_q: Optional[float],
) -> Optional[pd.DataFrame]:
    """Return per-horizon AUROC DataFrame.

    Strategy:
    1. If auroc_df has ``horizon_t``, use it directly.
    2. Otherwise, compute per-horizon AUROC from error_vs_uncertainty data.
    3. If neither is available, return None.
    """
    q = top_q if top_q else 10.0

    # Case 1: CSV already has per-horizon AUROC
    if auroc_df is not None and "horizon_t" in auroc_df.columns:
        log.info("Using CSV-based per-horizon AUROC data")
        return auroc_df

    # Case 2: Compute from error_vs_uncertainty
    if eu_df is not None and not eu_df.empty:
        log.info(
            "AUROC CSV lacks horizon_t; computing per-horizon AUROC from "
            "error_vs_uncertainty (top_q=%.1f%%)",
            q,
        )
        computed = compute_auroc_per_horizon(eu_df, top_q=q)
        if not computed.empty:
            return computed
        log.warning("Per-horizon AUROC computation returned empty DataFrame")

    # Case 3: Only aggregated AUROC available
    if auroc_df is not None and not auroc_df.empty:
        log.warning(
            "AUROC data is aggregated only (no horizon_t); "
            "AUROC-vs-horizon plot will be skipped, summary bar only"
        )
        return auroc_df

    return None


# ---------------------------------------------------------------------------
# AUROC vs horizon — line plot (primary)
# ---------------------------------------------------------------------------
def _plot_auroc_vs_horizon(
    df: pd.DataFrame,
    output_dir: Path,
    horizon_filter: Optional[int],
    smooth_mode: str,
    smooth_window: Optional[int],
    ewma_alpha: float,
    top_q: Optional[float],
    paper: bool = False,
    paper_font_size: int = 16,
) -> list[Path]:
    """AUROC vs horizon_t — one line per condition (primary plot)."""
    if "horizon_t" not in df.columns:
        log.warning("AUROC-vs-horizon: no horizon_t column; skipping")
        return []

    _check_auroc_sanity(df)

    # Do NOT filter by horizon for the line plot — show the full curve.
    # The horizon_filter is only relevant for the summary bar plot.

    agg = aggregate_seeds(
        df,
        group_cols=["condition", "horizon_t"],
        value_cols=["auroc"],
        agg="mean_std",
    )

    # Apply smoothing if requested
    group_cols = ["condition"]
    if smooth_mode != "none":
        window = smooth_window or 5
        agg, y_col = get_smoothed_column(agg, "auroc_mean", group_cols, smooth_mode, window, ewma_alpha)
    else:
        agg = agg.rename(columns={"auroc_mean": "y"})
        y_col = "y"

    if "auroc_std" in agg.columns:
        agg["y_std"] = agg["auroc_std"].fillna(0)
    else:
        agg["y_std"] = 0.0
    agg["ymin"] = agg[y_col] - agg["y_std"]
    agg["ymax"] = agg[y_col] + agg["y_std"]

    agg = add_short_labels(agg)
    conds = sorted(set(agg["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    q_str = f"{top_q:.0f}" if top_q else "10"

    # Caption
    caption_parts = [
        "AUROC vs Prediction Horizon",
        "",
        f"Labeling protocol: positives = top {q_str}% errors computed WITHIN each horizon",
        "(per-horizon labeling — horizons are NOT pooled when computing labels).",
        "",
        "Horizontal dashed line: AUROC = 0.5 (random classifier).",
        "Higher AUROC indicates epistemic uncertainty better distinguishes",
        "high-error from low-error predictions at that horizon.",
    ]
    if horizon_filter is not None:
        caption_parts.append(f"\nNote: horizon_filter h > {horizon_filter} is applied to summary bar only.")
    caption = "\n".join(caption_parts)

    suffix = _build_filename_suffix(None, smooth_mode, smooth_window, ewma_alpha)
    filename = f"decision_utility_auroc_vs_horizon{suffix}"

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    subtitle = f"Per-horizon labeling: positives = top {q_str}% errors within each horizon"
    labs_kwargs = {
        "title": "" if paper else "AUROC vs Prediction Horizon",
        "subtitle": "" if paper else subtitle,
        "x": "Prediction horizon (steps)",
        "y": "AUROC",
        "colour": "" if paper else "Condition",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(agg, aes(x="horizon_t", y=y_col, colour="condition_short",
                        fill="condition_short"))
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.15, colour="none")
        + geom_line(size=1.0)
        + geom_hline(yintercept=0.5, linetype="dashed", colour="grey", size=0.5)
        + col_sc + fill_sc
        + labs(**labs_kwargs)
        + scale_y_continuous(limits=(0.0, 1.05))
        + theme
    )
    return save_plot(p, output_dir, filename, caption=None if paper else caption)


# ---------------------------------------------------------------------------
# AUROC summary bar (optional)
# ---------------------------------------------------------------------------
def _plot_auroc_summary(
    df: pd.DataFrame,
    output_dir: Path,
    horizon_filter: Optional[int],
    top_q: Optional[float],
    paper: bool = False,
    paper_font_size: int = 16,
) -> list[Path]:
    """Mean AUROC per condition (summary bar plot), with horizon filtering."""
    # Filter by horizon if per-horizon data is available
    if horizon_filter is not None and "horizon_t" in df.columns:
        df = df[df["horizon_t"] > horizon_filter].copy()
        log.info("AUROC summary: filtered to horizons > %d (%d rows remain)",
                 horizon_filter, len(df))
    elif horizon_filter is not None:
        log.info("AUROC summary: horizon_filter=%d but no horizon_t column; using all data",
                 horizon_filter)

    if df.empty:
        log.warning("AUROC summary: no data after filtering")
        return []

    _check_auroc_sanity(df)

    agg = aggregate_seeds(
        df,
        group_cols=["condition"],
        value_cols=["auroc"],
        agg="mean_std",
    )
    agg = agg.rename(columns={"auroc_mean": "y", "auroc_std": "y_std"})
    agg["y_std"] = agg["y_std"].fillna(0)
    agg["ymin"] = agg["y"] - agg["y_std"]
    agg["ymax"] = agg["y"] + agg["y_std"]

    agg = add_short_labels(agg)
    conds = sorted(set(agg["condition_short"].tolist()))
    _, fill_sc = colour_scales(conds)

    q_str = f"{top_q:.0f}" if top_q else "10"

    # Caption
    caption_parts = [
        "AUROC Summary by Condition",
        "",
    ]
    if horizon_filter is not None and "horizon_t" in df.columns:
        caption_parts.append(f"Mean AUROC computed over horizons h > {horizon_filter}")
    else:
        caption_parts.append("Mean AUROC aggregated across all available data")
    caption_parts.extend([
        f"Labeling: positives = top {q_str}% errors per horizon (per-horizon labeling).",
        "Error bars: standard deviation across seeds / horizons.",
        "Horizontal dashed line: AUROC = 0.5 (random classifier).",
    ])
    caption = "\n".join(caption_parts)

    suffix = f"__hgt{horizon_filter}" if horizon_filter else ""
    filename = f"decision_utility_auroc_summary{suffix}"

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    subtitle = (
        f"AUROC over h > {horizon_filter}" if horizon_filter
        else "AUROC across all horizons"
    ) + f" | top {q_str}% per-horizon labeling"
    labs_kwargs = {
        "title": "" if paper else "AUROC Summary by Condition",
        "subtitle": "" if paper else subtitle,
        "x": "Condition",
        "y": "Mean AUROC",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(agg, aes(x="condition_short", y="y", fill="condition_short"))
        + geom_col(width=0.6, alpha=0.8)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.15, size=0.5)
        + geom_hline(yintercept=0.5, linetype="dashed", colour="grey", size=0.5)
        + fill_sc
        + labs(**labs_kwargs)
        + scale_y_continuous(limits=(0.0, 1.05))
        + theme
    )
    return save_plot(p, output_dir, filename, caption=None if paper else caption)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plot_decision(
    risk_df: Optional[pd.DataFrame],
    auroc_df: Optional[pd.DataFrame],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
    eu_df: Optional[pd.DataFrame] = None,
) -> list[Path]:
    """Generate decision utility plots.

    Parameters
    ----------
    risk_df : DataFrame or None
        Risk-coverage data.
    auroc_df : DataFrame or None
        Pre-computed AUROC data (may or may not have horizon_t).
    output_dir : Path
        Root output directory (subdirectory ``decision_utility/`` is created).
    summary : list[str]
        Run summary lines.
    plot_config : dict
        Plotting configuration from CLI.
    eu_df : DataFrame or None
        error_vs_uncertainty data, used to compute per-horizon AUROC when
        the AUROC CSV only contains aggregated stats.
    """
    out = output_dir / "decision_utility"
    saved: list[Path] = []

    # Extract config
    horizon_filter = plot_config.get("horizon_filter")
    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)
    top_q = plot_config.get("top_q")
    no_auroc_summary = plot_config.get("no_auroc_summary", False)
    coverage_target = plot_config.get("coverage_target", 0.80)
    paper = plot_config.get("paper", False)
    paper_font_size = int(plot_config.get("paper_font_size", 16))
    plot_risk = plot_config.get("plot_risk", True)
    plot_auroc = plot_config.get("plot_auroc", True)

    # ── Risk-coverage ─────────────────────────────────────────────────
    if plot_risk and risk_df is not None and not risk_df.empty:
        saved.extend(_plot_risk_coverage(
            risk_df, out, horizon_filter,
            smooth_mode, smooth_window, ewma_alpha,
            coverage_target=coverage_target,
            paper=paper, paper_font_size=paper_font_size,
        ))
    elif plot_risk:
        log.info("No risk-coverage data")
        summary.append("plot_decision: no risk-coverage data")

    # ── AUROC ─────────────────────────────────────────────────────────
    auroc_data = _compute_or_load_auroc(auroc_df, eu_df, top_q) if plot_auroc else None

    if plot_auroc and auroc_data is not None and not auroc_data.empty:
        if "horizon_t" in auroc_data.columns:
            # Primary plot: AUROC vs horizon
            saved.extend(_plot_auroc_vs_horizon(
                auroc_data, out, horizon_filter,
                smooth_mode, smooth_window, ewma_alpha, top_q,
                paper=paper, paper_font_size=paper_font_size,
            ))
            # Optional summary bar
            if not no_auroc_summary:
                saved.extend(_plot_auroc_summary(
                    auroc_data, out, horizon_filter, top_q,
                    paper=paper, paper_font_size=paper_font_size,
                ))
        else:
            # Only aggregated data — bar plot only
            log.warning(
                "AUROC data is aggregated (no horizon_t); only summary bar "
                "plot is possible"
            )
            if not no_auroc_summary:
                saved.extend(_plot_auroc_summary(
                    auroc_data, out, horizon_filter, top_q,
                    paper=paper, paper_font_size=paper_font_size,
                ))
            summary.append(
                "plot_decision: AUROC aggregated only — "
                "AUROC-vs-horizon skipped (need error_vs_uncertainty for offline computation)"
            )
    elif plot_auroc:
        log.info("No AUROC data available (CSV or computed)")
        summary.append("plot_decision: no AUROC data")

    summary.append(f"plot_decision: {len(saved)} plots saved")
    return saved
