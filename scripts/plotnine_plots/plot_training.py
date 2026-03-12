"""
plot_training.py — Training curves from TensorBoard aggregated data.

Generates:
    training/reward_vs_step.pdf
    training/autoregressive_error_vs_step.pdf
    training/epistemic_uncertainty_vs_step.pdf
    training/epistemic_uncertainty_vs_step__zscore.pdf
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, labs, scale_x_continuous,
    coord_cartesian,
)

from .transforms import extract_tb_metric, add_short_labels, robust_zscore, get_smoothed_column
from .styles import paper_theme, paper_theme_no_title, colour_scales, build_colour_map, save_plot

log = logging.getLogger(__name__)

# TensorBoard metric stems
_REWARD_STEM = "Episode_Reward_track_lin_vel_xy_exp"
_REWARD_TRAIN_MEAN_STEM = "Train_mean_reward"
_AE_STEM = "System_Dynamics_autoregressive_error"
_EU_STEM = "Model_Based_epistemic_uncertainty"


def _plot_tb_curve(
    df: pd.DataFrame,
    y_col: str,
    std_col: str,
    title: str,
    y_label: str,
) -> Optional[ggplot]:
    """Generic training curve with mean ± std ribbon."""
    if df.empty:
        return None
    df = add_short_labels(df)
    conds = sorted(set(df["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    p = (
        ggplot(df, aes(x="step", y=y_col, colour="condition_short",
                       fill="condition_short"))
        + geom_ribbon(
            aes(ymin=f"{y_col} - {std_col}", ymax=f"{y_col} + {std_col}"),
            alpha=0.2, colour="none",
        )
        + geom_line(size=1.0)
        + col_sc + fill_sc
        + labs(title=title, x="Training step", y=y_label,
               colour="Condition", fill="Condition")
        + paper_theme()
    )
    return p


def _plot_tb_curve_safe(
    df: pd.DataFrame,
    title: str,
    y_label: str,
    extra_layers=None,
    y_lims = [None, None],
    line_size=1.0,
    smooth_mode: str = "none",
    smooth_window: Optional[int] = None,
    ewma_alpha: float = 0.1,
    paper: bool = False,
    paper_font_size: int = 16,
) -> Optional[ggplot]:
    """Plot training curve with ribbon for value ± std."""
    if df.empty:
        return None

    # Apply smoothing if requested
    # TensorBoard data may not have a seed column (pre-aggregated)
    group_cols = [c for c in ["condition", "seed"] if c in df.columns]
    if not group_cols:
        group_cols = ["condition"]
    if smooth_mode != "none":
        window = smooth_window or 100
        df, y_col = get_smoothed_column(df, "value", group_cols, smooth_mode, window, ewma_alpha)
    else:
        y_col = "value"

    df = add_short_labels(df)
    conds = sorted(set(df["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    df = df.copy()
    df["ymin"] = df[y_col] - df["std"]
    df["ymax"] = df[y_col] + df["std"]

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    labs_kwargs = {
        "title": "" if paper else title,
        "x": "Training step",
        "y": y_label,
        "colour": "" if paper else "Condition",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(df, aes(x="step", y=y_col, colour="condition_short",
                       fill="condition_short"))
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.2, colour="none")
        + geom_line(size=line_size)
        + col_sc + fill_sc
        + labs(**labs_kwargs)
        + theme
        + coord_cartesian(ylim=y_lims)
    )
    if extra_layers:
        for layer in extra_layers:
            p = p + layer
    return p


def plot_training(
    tb_dict: Optional[dict[str, pd.DataFrame]],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Generate all training curve plots. Returns list of saved paths."""
    out = output_dir / "training"
    saved: list[Path] = []

    if tb_dict is None:
        summary.append("plot_training: SKIPPED (no tensorboard data)")
        return saved

    # Extract config
    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)
    paper = plot_config.get("paper", False)
    paper_font_size = int(plot_config.get("paper_font_size", 16))

    # Build filename suffix for smoothing
    if smooth_mode == "rolling_mean" and smooth_window:
        smooth_suffix = f"__smooth_rollmean_w{smooth_window}"
    elif smooth_mode == "ewma":
        smooth_suffix = f"__smooth_ewma_a{ewma_alpha:.2f}"
    else:
        smooth_suffix = ""

    # 1) Reward — clip to steady-state (skip initial ramp)
    reward_df = extract_tb_metric(tb_dict, _REWARD_STEM)
    if not reward_df.empty:
        # set the y-limits to the min/max reward observed in the steady-state region
        step_min = reward_df["step"].min()
        step_max = reward_df["step"].max()
        y_mean = reward_df["value"].mean()
        y_std = reward_df["value"].std()
        y_min = y_mean - 1.2 * y_std
        y_max = y_mean + 1.2 * y_std

        p = _plot_tb_curve_safe(
            reward_df,
            "Tracking Reward (steady-state)",
            "Reward (lin vel xy)",
            y_lims=[y_min, y_max],
            line_size=0.6,
            smooth_mode=smooth_mode,
            smooth_window=smooth_window,
            ewma_alpha=ewma_alpha,
            paper=paper,
            paper_font_size=paper_font_size,
        )
        if p is not None:
            saved.extend(save_plot(p, out, f"reward_vs_step{smooth_suffix}"))

    # 2) Autoregressive error
    ae_df = extract_tb_metric(tb_dict, _AE_STEM)
    y_min = ae_df["value"].min() - 0.6 * ae_df["std"].max()
    y_max = ae_df["value"].max() + 0.6 * ae_df["std"].max()
    p = _plot_tb_curve_safe(ae_df, "Autoregressive Error", "Error", y_lims=[y_min, y_max],
                           smooth_mode=smooth_mode, smooth_window=smooth_window, ewma_alpha=ewma_alpha,
                           paper=paper, paper_font_size=paper_font_size)
    if p is not None:
        saved.extend(save_plot(p, out, f"autoregressive_error_vs_step{smooth_suffix}"))

    # 3) Epistemic uncertainty (raw)
    eu_df = extract_tb_metric(tb_dict, _EU_STEM)
    p = _plot_tb_curve_safe(eu_df, "Epistemic Uncertainty", "Uncertainty",
                           smooth_mode=smooth_mode, smooth_window=smooth_window, ewma_alpha=ewma_alpha,
                           paper=paper, paper_font_size=paper_font_size)
    if p is not None:
        saved.extend(save_plot(p, out, f"epistemic_uncertainty_vs_step{smooth_suffix}"))

    # 4) Epistemic uncertainty (z-scored per condition)
    if not eu_df.empty:
        eu_z = robust_zscore(eu_df, "value", group_col="condition",
                             out_col="value_z")
        eu_z["std_z"] = 0.0  # z-scoring removes natural std
        eu_z["ymin"] = eu_z["value_z"] - eu_z["std_z"]
        eu_z["ymax"] = eu_z["value_z"] + eu_z["std_z"]
        eu_z = add_short_labels(eu_z)
        conds = sorted(set(eu_z["condition_short"].tolist()))
        col_sc, fill_sc = colour_scales(conds)
        theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
        labs_kwargs = {
            "title": "" if paper else "Epistemic Uncertainty (z-scored)",
            "x": "Training step",
            "y": "Robust z-score",
            "colour": "" if paper else "Condition",
            "fill": "" if paper else "Condition",
        }
        p = (
            ggplot(eu_z, aes(x="step", y="value_z",
                             colour="condition_short",
                             fill="condition_short"))
            + geom_ribbon(aes(ymin="ymin", ymax="ymax"),
                          alpha=0.2, colour="none")
            + geom_line(size=1.0)
            + col_sc + fill_sc
            + labs(**labs_kwargs)
            + theme
        )
        saved.extend(save_plot(p, out, f"epistemic_uncertainty_vs_step__zscore{smooth_suffix}"))

    summary.append(f"plot_training: {len(saved)} plots saved")
    return saved


def _make_plot_ax(smooth_mode, smooth_window, ewma_alpha, base_size):
    """Return a closure that plots a single training curve on a matplotlib Axes."""

    def _plot_ax(ax, df, y_label, y_lims=None, line_size=1.0):
        group_cols = [c for c in ["condition", "seed"] if c in df.columns]
        if not group_cols:
            group_cols = ["condition"]
        if smooth_mode != "none":
            window = smooth_window or 100
            df2, y_col = get_smoothed_column(df, "value", group_cols, smooth_mode, window, ewma_alpha)
        else:
            df2, y_col = df, "value"

        df2 = add_short_labels(df2)
        conds = sorted(set(df2["condition_short"].tolist()))
        cmap = build_colour_map(conds)

        for cond in conds:
            cdf = df2[df2["condition_short"] == cond].sort_index()
            ax.plot(cdf["step"], cdf[y_col], color=cmap[cond], linewidth=line_size, label=cond)
            if "std" in cdf.columns:
                ax.fill_between(
                    cdf["step"],
                    cdf[y_col] - cdf["std"],
                    cdf[y_col] + cdf["std"],
                    color=cmap[cond],
                    alpha=0.2,
                    linewidth=0,
                )

        ax.set_xlabel("Training step", fontsize=base_size + 1)
        ax.set_ylabel(y_label, fontsize=base_size + 1)
        ax.tick_params(axis="both", labelsize=base_size - 1)
        if y_lims is not None:
            ax.set_ylim(y_lims)

    return _plot_ax


def _add_shared_legend(fig, axes, base_size):
    """Add a shared legend below the axes and remove per-axis legends."""
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 6),
            fontsize=base_size - 1,
            frameon=False,
        )
    for ax in axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def plot_training_diptych_paper(
    tb_dict: Optional[dict[str, pd.DataFrame]],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Save a 2-panel figure (reward + epistemic uncertainty) for paper use."""
    saved: list[Path] = []
    if tb_dict is None:
        summary.append("plot_training_diptych_paper: SKIPPED (no tensorboard data)")
        return saved

    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)
    base_size = int(plot_config.get("paper_font_size", 16))

    reward_df = extract_tb_metric(tb_dict, _REWARD_TRAIN_MEAN_STEM)
    eu_df = extract_tb_metric(tb_dict, _EU_STEM)

    if reward_df.empty or eu_df.empty:
        summary.append("plot_training_diptych_paper: SKIPPED (missing metric data)")
        return saved

    reward_mean = reward_df["value"].mean()
    reward_std = reward_df["value"].std()
    reward_lims = [reward_mean - 1.2 * reward_std, reward_mean + 1.2 * reward_std]

    plot_ax = _make_plot_ax(smooth_mode, smooth_window, ewma_alpha, base_size)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    plot_ax(axes[0], reward_df, "Mean Reward", reward_lims, line_size=0.8)
    plot_ax(axes[1], eu_df, "Uncertainty", line_size=1.5)

    _add_shared_legend(fig, axes, base_size)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_reward_uncertainty.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    summary.append("plot_training_diptych_paper: 1 plot saved")
    return saved


def plot_training_ae_paper(
    tb_dict: Optional[dict[str, pd.DataFrame]],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Save a single-panel autoregressive error plot for paper use."""
    saved: list[Path] = []
    if tb_dict is None:
        summary.append("plot_training_ae_paper: SKIPPED (no tensorboard data)")
        return saved

    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)
    base_size = int(plot_config.get("paper_font_size", 16))

    ae_df = extract_tb_metric(tb_dict, _AE_STEM)

    if ae_df.empty:
        summary.append("plot_training_ae_paper: SKIPPED (missing metric data)")
        return saved

    ae_min = 0.5
    ae_max = ae_df["value"].max() + 0.6 * ae_df["std"].max()
    ae_lims = [ae_min, ae_max]

    plot_ax = _make_plot_ax(smooth_mode, smooth_window, ewma_alpha, base_size)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.8))
    plot_ax(ax, ae_df, "Autoregressive Error", ae_lims, line_size=1.2)

    _add_shared_legend(fig, [ax], base_size)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_autoregressive_error.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    summary.append("plot_training_ae_paper: 1 plot saved")
    return saved


def plot_training_triptych_paper(
    tb_dict: Optional[dict[str, pd.DataFrame]],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Save training paper plots: 2-panel (reward + uncertainty) + 1-panel (AE)."""
    saved: list[Path] = []
    saved.extend(plot_training_diptych_paper(tb_dict, output_dir, summary, plot_config))
    saved.extend(plot_training_ae_paper(tb_dict, output_dir, summary, plot_config))
    return saved
