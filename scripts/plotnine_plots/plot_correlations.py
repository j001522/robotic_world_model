"""
plot_correlations.py — Pearson/Spearman correlation vs prediction horizon.

Generates:
    correlations/pearson_vs_horizon.pdf
    correlations/spearman_vs_horizon.pdf
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, geom_hline, labs,
    scale_y_continuous,
)

from .transforms import add_short_labels, aggregate_seeds, get_smoothed_column
from .styles import paper_theme, colour_scales, save_plot

log = logging.getLogger(__name__)


def _build_filename_suffix(
    smooth_mode: str,
    smooth_window: Optional[int],
    ewma_alpha: float,
    horizon_filter: Optional[int] = None,
    mode: Optional[str] = None,
) -> str:
    """Build filename suffix based on configuration.

    Parameters
    ----------
    smooth_mode : str
        Smoothing mode (none, rolling_mean, ewma).
    smooth_window : int or None
        Window size for rolling mean.
    ewma_alpha : float
        Alpha for EWMA smoothing.
    horizon_filter : int or None
        Horizon filter (e.g., h>30).
    mode : str or None
        Error mode (vel, full, etc.).

    Returns
    -------
    str
        Filename suffix like "__hgt30__vel__smooth_rollmean_w10".
    """
    parts = []
    if horizon_filter is not None:
        parts.append(f"hgt{horizon_filter}")
    if mode:
        parts.append(mode)
    if smooth_mode == "rolling_mean" and smooth_window:
        parts.append(f"smooth_rollmean_w{smooth_window}")
    elif smooth_mode == "ewma":
        parts.append(f"smooth_ewma_a{ewma_alpha:.2f}")
    return "__" + "__".join(parts) if parts else ""


def _plot_corr_vs_horizon(
    df: pd.DataFrame,
    corr_col: str,
    title: str,
    output_dir: Path,
    filename_base: str,
    smooth_mode: str,
    smooth_window: Optional[int],
    ewma_alpha: float,
) -> list[Path]:
    """One correlation metric vs prediction_step."""
    if corr_col not in df.columns:
        log.info("Column %s not found; skipping", corr_col)
        return []

    agg = aggregate_seeds(
        df,
        group_cols=["condition", "prediction_step"],
        value_cols=[corr_col],
        agg="mean_std",
    )
    y_col_base = f"{corr_col}_mean" if f"{corr_col}_mean" in agg.columns else corr_col
    y_std = f"{corr_col}_std" if f"{corr_col}_std" in agg.columns else None

    # Apply smoothing if requested
    group_cols = ["condition"]
    if smooth_mode != "none":
        window = smooth_window or 10
        agg, y_col = get_smoothed_column(agg, y_col_base, group_cols, smooth_mode, window, ewma_alpha)
    else:
        agg = agg.rename(columns={y_col_base: "y"})
        y_col = "y"

    if y_std and y_std in agg.columns:
        agg = agg.rename(columns={y_std: "y_std"})
        agg["y_std"] = agg["y_std"].fillna(0)
    else:
        agg["y_std"] = 0.0
    agg["ymin"] = agg[y_col] - agg["y_std"]
    agg["ymax"] = agg[y_col] + agg["y_std"]

    agg = add_short_labels(agg)
    conds = sorted(agg["condition_short"].unique())
    col_sc, fill_sc = colour_scales(conds)
    # compute limits on y for better visualization
    y_limits = (agg[y_col].min() - 1.5 * agg["y_std"].max(), agg[y_col].max() + 1.5 * agg["y_std"].max())

    # Build filename
    suffix = _build_filename_suffix(smooth_mode, smooth_window, ewma_alpha)
    filename = f"{filename_base}{suffix}"

    p = (
        ggplot(agg, aes(x="prediction_step", y=y_col,
                        colour="condition_short",
                        fill="condition_short"))
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.15, colour="none")
        + geom_line(size=1.0)
        + geom_hline(yintercept=0, linetype="dashed", colour="grey", size=0.4)
        + col_sc + fill_sc
        + labs(title=title, x="Prediction horizon", y=title.split()[0] + " r",
               colour="Condition", fill="Condition")
        + scale_y_continuous(limits=y_limits)
        + paper_theme()
    )
    return save_plot(p, output_dir, filename)


def plot_correlations(
    df: Optional[pd.DataFrame],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Generate correlation plots."""
    out = output_dir / "correlations"
    saved: list[Path] = []

    if df is None:
        summary.append("plot_correlations: SKIPPED (no data)")
        return saved

    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)

    saved.extend(_plot_corr_vs_horizon(
        df, "pearson_r", "Pearson Correlation vs Horizon",
        out, "correlations_pearson_vs_horizon",
        smooth_mode, smooth_window, ewma_alpha,
    ))
    saved.extend(_plot_corr_vs_horizon(
        df, "spearman_r", "Spearman Correlation vs Horizon",
        out, "correlations_spearman_vs_horizon",
        smooth_mode, smooth_window, ewma_alpha,
    ))

    summary.append(f"plot_correlations: {len(saved)} plots saved")
    return saved
