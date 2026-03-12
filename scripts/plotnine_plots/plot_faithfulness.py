"""
plot_faithfulness.py — Faithfulness gap and real vs imagined rewards.

Generates:
    faithfulness/faithfulness_gap_vs_step.pdf
    faithfulness/real_vs_imagined_reward.pdf
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, geom_point, geom_hline,
    labs, scale_linetype_manual,
)

from .transforms import add_short_labels, get_smoothed_column
from .styles import paper_theme, colour_scales, save_plot

log = logging.getLogger(__name__)


def _plot_faithfulness_gap(df: pd.DataFrame, output_dir: Path,
                           smooth_mode: str, smooth_window: Optional[int],
                           ewma_alpha: float) -> list[Path]:
    """Faithfulness gap vs training step."""
    if "faithfulness_gap" not in df.columns or "step" not in df.columns:
        log.info("Missing faithfulness_gap or step column")
        return []

    # Apply smoothing if requested
    group_cols = ["condition"]
    if smooth_mode != "none":
        window = smooth_window or 50
        df, y_col = get_smoothed_column(df, "faithfulness_gap", group_cols, smooth_mode, window, ewma_alpha)
    else:
        y_col = "faithfulness_gap"

    df = add_short_labels(df)
    conds = sorted(df["condition_short"].unique())
    col_sc, _ = colour_scales(conds)

    # Build filename suffix
    if smooth_mode == "rolling_mean" and smooth_window:
        smooth_suffix = f"__smooth_rollmean_w{smooth_window}"
    elif smooth_mode == "ewma":
        smooth_suffix = f"__smooth_ewma_a{ewma_alpha:.2f}"
    else:
        smooth_suffix = ""

    p = (
        ggplot(df, aes(x="step", y=y_col,
                       colour="condition_short"))
        + geom_line(size=1.0)
        + geom_point(size=2)
        + geom_hline(yintercept=0, linetype="dashed", colour="grey", size=0.4)
        + col_sc
        + labs(title="Faithfulness Gap vs Training Step",
               x="Training step",
               y="Faithfulness gap (imagined - real)",
               colour="Condition")
        + paper_theme()
    )
    return save_plot(p, output_dir, f"faithfulness_gap_vs_step{smooth_suffix}")


def _plot_real_vs_imagined(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Real and imagined reward curves on the same plot."""
    needed = {"step", "real_reward_mean", "imagination_reward_mean", "condition"}
    if not needed.issubset(df.columns):
        log.info("Missing columns for real_vs_imagined: %s",
                 needed - set(df.columns))
        return []

    df = add_short_labels(df)

    # Reshape to long format for two lines per condition
    real = df[["step", "condition_short", "real_reward_mean"]].copy()
    real = real.rename(columns={"real_reward_mean": "reward"})
    real["source"] = "Real"

    imag = df[["step", "condition_short", "imagination_reward_mean"]].copy()
    imag = imag.rename(columns={"imagination_reward_mean": "reward"})
    imag["source"] = "Imagined"

    # Add std if available
    if "real_reward_std" in df.columns:
        real["std"] = df["real_reward_std"].values
    else:
        real["std"] = 0.0
    imag["std"] = 0.0  # Usually not available for imagined

    long = pd.concat([real, imag], ignore_index=True)
    long["ymin"] = long["reward"] - long["std"]
    long["ymax"] = long["reward"] + long["std"]

    conds = sorted(long["condition_short"].unique())
    col_sc, fill_sc = colour_scales(conds)

    p = (
        ggplot(long, aes(x="step", y="reward",
                         colour="condition_short",
                         linetype="source"))
        + geom_line(size=1.0)
        + col_sc
        + scale_linetype_manual(values={"Real": "solid", "Imagined": "dashed"})
        + labs(title="Real vs Imagined Reward",
               x="Training step", y="Reward",
               colour="Condition", linetype="Source")
        + paper_theme()
    )
    return save_plot(p, output_dir, "real_vs_imagined_reward")


def plot_faithfulness(
    df: Optional[pd.DataFrame],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Generate faithfulness plots."""
    out = output_dir / "faithfulness"
    saved: list[Path] = []

    if df is None:
        summary.append("plot_faithfulness: SKIPPED (no data)")
        return saved

    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)

    saved.extend(_plot_faithfulness_gap(df, out, smooth_mode, smooth_window, ewma_alpha))
    saved.extend(_plot_real_vs_imagined(df, out))

    summary.append(f"plot_faithfulness: {len(saved)} plots saved")
    return saved
