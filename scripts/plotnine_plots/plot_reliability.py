"""
plot_reliability.py — Hallucination horizon distributions.

Generates:
    reliability/hallucination_horizon_distribution.pdf
    reliability/hallucination_horizon_boxplot.pdf
    reliability/hallucination_horizon_strip.pdf

When raw error curves are available, hallucination horizons are
**recomputed on the fly** using a shared threshold derived from a
baseline condition.  This ensures fair cross-condition comparison
(see PLOTTING.md for details).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from plotnine import (
    ggplot, aes, geom_histogram, geom_boxplot, geom_jitter,
    geom_point, geom_text,
    facet_wrap, labs, scale_x_continuous, scale_y_continuous,
    coord_flip, position_jitter,
)

from .transforms import add_short_labels, recompute_hallucination_horizons
from .styles import paper_theme, paper_theme_no_title, colour_scales, save_plot

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------
def _plot_hallucination_histogram(
    df: pd.DataFrame,
    horizon_col: str,
    output_dir: Path,
    threshold_info: str = "",
) -> list[Path]:
    """Per-condition histogram of hallucination horizons."""
    df = add_short_labels(df)
    conds = sorted(df["condition_short"].unique())
    _, fill_sc = colour_scales(conds)

    title = "Hallucination Horizon Distribution"
    if threshold_info:
        title += f"\n({threshold_info})"

    p = (
        ggplot(df, aes(x=horizon_col, fill="condition_short"))
        + geom_histogram(bins=30, alpha=0.7, position="identity")
        + facet_wrap("condition_short", scales="free_y")
        + fill_sc
        + labs(title=title,
               x="Hallucination horizon (steps)",
               y="Count", fill="Condition")
        + paper_theme()
    )
    return save_plot(p, output_dir, "hallucination_horizon_distribution",
                     width=7, height=5)


def _plot_hallucination_boxplot(
    df: pd.DataFrame,
    horizon_col: str,
    output_dir: Path,
    threshold_info: str = "",
    paper: bool = False,
    paper_font_size: int = 16,
) -> list[Path]:
    """Boxplot of hallucination horizons across conditions."""
    df = add_short_labels(df)
    conds = sorted(df["condition_short"].unique())
    _, fill_sc = colour_scales(conds)

    title = "Hallucination Horizon by Condition"
    if threshold_info:
        title += f"\n({threshold_info})"

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    labs_kwargs = {
        "title": "" if paper else title,
        "x": "Condition",
        "y": "Trajectory step",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(df, aes(x="condition_short", y=horizon_col,
                       fill="condition_short"))
        + geom_boxplot(alpha=0.7, outlier_alpha=0.3)
        + fill_sc
        + coord_flip()
        + labs(**labs_kwargs)
        + theme
    )
    return save_plot(p, output_dir, "hallucination_horizon_boxplot")


def _plot_hallucination_strip(
    df: pd.DataFrame,
    horizon_col: str,
    output_dir: Path,
    threshold_info: str = "",
) -> list[Path]:
    """Strip/jitter plot — better than boxplots when n is small."""
    df = add_short_labels(df)
    conds = sorted(df["condition_short"].unique())
    col_sc, fill_sc = colour_scales(conds)

    # Compute per-condition medians for annotation
    medians = (
        df.groupby("condition_short")[horizon_col]
        .median()
        .reset_index()
        .rename(columns={horizon_col: "median_h"})
    )

    title = "Hallucination Horizon (shared threshold)"
    if threshold_info:
        title += f"\n({threshold_info})"

    p = (
        ggplot(df, aes(x="condition_short", y=horizon_col,
                       colour="condition_short"))
        + geom_jitter(width=0.15, height=0, size=3, alpha=0.8)
        + col_sc
        + coord_flip()
        + labs(title=title,
               x="Condition", y="Hallucination horizon (steps)",
               colour="Condition")
        + paper_theme()
    )
    return save_plot(p, output_dir, "hallucination_horizon_strip")


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def plot_reliability(
    per_ep_df: Optional[pd.DataFrame],
    distrib_df: Optional[pd.DataFrame],
    output_dir: Path,
    summary: list[str],
    raw_error_df: Optional[pd.DataFrame] = None,
    plot_config: Optional[dict] = None,
) -> list[Path]:
    """Generate hallucination distribution plots.

    Parameters
    ----------
    per_ep_df   : per-episode hallucination data (legacy, per-condition thresholds)
    distrib_df  : aggregated hallucination distribution stats
    raw_error_df: raw per-step error curves — used to recompute hallucination
                  horizons with a shared baseline threshold (preferred path).
    """
    out = output_dir / "reliability"
    saved: list[Path] = []
    plot_config = plot_config or {}
    paper = plot_config.get("paper", False)
    paper_font_size = int(plot_config.get("paper_font_size", 16))
    plot_box = plot_config.get("plot_box", True)
    plot_hist = plot_config.get("plot_hist", True)
    plot_strip = plot_config.get("plot_strip", True)

    # ── Preferred path: recompute from raw error curves ────────────────
    recomputed = None
    threshold_info = ""
    if raw_error_df is not None and not raw_error_df.empty:
        recomputed = recompute_hallucination_horizons(raw_error_df)
        if not recomputed.empty:
            threshold_val = recomputed["threshold"].iloc[0]
            method = recomputed["threshold_method"].iloc[0]
            threshold_info = f"shared threshold={threshold_val:.4f}, method={method}"
            log.info("Using recomputed shared-threshold hallucination horizons")
        else:
            recomputed = None

    # Decide which data to use
    if recomputed is not None and not recomputed.empty:
        src = recomputed
        horizon_col = "hallucination_horizon"
        summary.append(f"plot_reliability: using SHARED threshold ({threshold_info})")
    elif per_ep_df is not None:
        src = per_ep_df
        horizon_col = "hallucination_horizon"
        threshold_info = "per-condition threshold (diagnostic only)"
        summary.append(
            "plot_reliability: WARNING — using per-condition thresholds "
            "(raw error curves not available for recomputation)"
        )
    elif distrib_df is not None:
        src = distrib_df
        # Try to find the right column
        horizon_col = "hallucination_horizon"
        for c in ("hallucination_horizon", "horizon_t", "median"):
            if c in src.columns:
                horizon_col = c
                break
        threshold_info = "aggregated data"
        summary.append("plot_reliability: using aggregated distribution data")
    else:
        summary.append("plot_reliability: SKIPPED (no hallucination data)")
        return saved

    if horizon_col not in src.columns:
        summary.append(
            f"plot_reliability: SKIPPED (column '{horizon_col}' not found)"
        )
        return saved

    if "condition" not in src.columns:
        summary.append("plot_reliability: SKIPPED (no condition column)")
        return saved

    # ── Generate plots ─────────────────────────────────────────────────
    if plot_hist:
        saved.extend(_plot_hallucination_histogram(src, horizon_col, out, threshold_info))
    if plot_box:
        saved.extend(_plot_hallucination_boxplot(
            src, horizon_col, out, threshold_info,
            paper=paper, paper_font_size=paper_font_size,
        ))
    if plot_strip:
        saved.extend(_plot_hallucination_strip(src, horizon_col, out, threshold_info))

    summary.append(f"plot_reliability: {len(saved)} plots saved")
    return saved
