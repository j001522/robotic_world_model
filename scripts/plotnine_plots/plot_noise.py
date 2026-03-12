"""
plot_noise.py — Error vs noise level from noise_robustness data.

Generates:
    noise_robustness/error_vs_noise_level.pdf
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_errorbar, labs,
)

from .transforms import add_short_labels, aggregate_seeds
from .styles import paper_theme, paper_theme_no_title, colour_scales, save_plot
from .transforms import select_error_columns

log = logging.getLogger(__name__)


def plot_noise(
    df: Optional[pd.DataFrame],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Error vs noise level plot."""
    out = output_dir / "noise_robustness"
    saved: list[Path] = []

    if df is None:
        summary.append("plot_noise: SKIPPED (no data)")
        return saved

    # Noise plots typically have few points; don't apply smoothing
    paper = plot_config.get("paper", False)
    paper_font_size = int(plot_config.get("paper_font_size", 16))
    error_mode = plot_config.get("error_mode", "full")

    if "error_mean" not in df.columns:
        try:
            df = select_error_columns(df, error_mode)
        except ValueError as exc:
            summary.append(f"plot_noise: SKIPPED ({exc})")
            return saved
    # Aggregate across seeds
    agg = aggregate_seeds(
        df,
        group_cols=["condition", "noise_level"],
        value_cols=["error_mean"],
        agg="mean_std",
    )
    agg = agg.rename(columns={
        "error_mean_mean": "y",
        "error_mean_std": "y_std",
    })
    agg["y_std"] = agg["y_std"].fillna(0)
    agg["ymin"] = agg["y"] - agg["y_std"]
    agg["ymax"] = agg["y"] + agg["y_std"]

    agg = add_short_labels(agg)
    conds = sorted(set(agg["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    labs_kwargs = {
        "title": "" if paper else "Error vs Noise Level",
        "x": "Noise level",
        "y": "Mean ARE (relative)",
        "colour": "" if paper else "Condition",
    }
    p = (
        ggplot(agg, aes(x="noise_level", y="y", colour="condition_short"))
        + geom_line(size=1.0)
        + geom_point(size=2.5)
        + geom_errorbar(aes(ymin="ymin", ymax="ymax"), width=0.02, size=0.5)
        + col_sc
        + labs(**labs_kwargs)
        + theme
    )
    suffix = "__vel" if error_mode == "vel" else ""
    saved.extend(save_plot(p, out, f"error_vs_noise_level{suffix}"))
    summary.append(f"plot_noise: {len(saved)} plots saved")
    return saved
