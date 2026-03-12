"""
plot_long_horizon.py — Error and uncertainty vs prediction horizon.

Generates (for each error mode: __full, __vel where available):
    long_horizon/error_vs_horizon__{mode}.pdf
    long_horizon/uncertainty_vs_horizon.pdf
    long_horizon/uncertainty_vs_horizon__zscore.pdf

X-axis shows **trajectory step** (= prediction_step + history_horizon).
The history_horizon is auto-detected from the error_vs_uncertainty data
(which carries a ``history_horizon`` column) or defaults to 32.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, labs, scale_x_continuous,
)

from .transforms import (
    select_error_columns, has_velocity_columns, add_short_labels,
    aggregate_seeds, robust_zscore, get_smoothed_column,
)
from .styles import paper_theme, paper_theme_no_title, colour_scales, save_plot

log = logging.getLogger(__name__)

# Default history horizon (context window size).
_DEFAULT_HISTORY_HORIZON = 32


def _detect_history_horizon(eu_df: Optional[pd.DataFrame]) -> int:
    """Detect history_horizon from the error_vs_uncertainty DataFrame."""
    if eu_df is not None and "history_horizon" in eu_df.columns:
        vals = eu_df["history_horizon"].dropna().unique()
        if len(vals) == 1:
            h = int(vals[0])
            log.info("Detected history_horizon = %d from error_vs_uncertainty data", h)
            return h
        elif len(vals) > 1:
            log.warning(
                "Multiple history_horizon values found: %s; using default %d",
                vals, _DEFAULT_HISTORY_HORIZON,
            )
    return _DEFAULT_HISTORY_HORIZON


def _add_trajectory_step(
    df: pd.DataFrame,
    history_horizon: int,
) -> pd.DataFrame:
    """Ensure the DataFrame has a ``trajectory_step`` column.

    Logic:
    - If ``prediction_step`` column exists, the ``horizon_t`` is already
      in trajectory-step space (as in error_vs_uncertainty data).  Use
      ``horizon_t`` directly.
    - Otherwise ``horizon_t`` is prediction-step-based (as in
      hallucination_horizon data).  Add ``history_horizon`` to convert.
    """
    df = df.copy()
    if "prediction_step" in df.columns:
        # horizon_t is already trajectory steps (33..200)
        df["trajectory_step"] = df["horizon_t"]
    elif "horizon_t" in df.columns:
        # horizon_t is prediction steps (1..168) — convert
        df["trajectory_step"] = df["horizon_t"] + history_horizon
    else:
        raise KeyError("No horizon_t or prediction_step column found")
    return df


def _plot_error_vs_horizon(
    df: pd.DataFrame,
    mode: str,
    output_dir: Path,
    smooth_mode: str,
    smooth_window: Optional[int],
    ewma_alpha: float,
    paper: bool = False,
    paper_font_size: int = 16,
) -> list[Path]:
    """Error (mean +/- std across seeds) vs trajectory step for one error mode."""
    h_col = "trajectory_step"
    try:
        df = select_error_columns(df, mode)
    except ValueError as e:
        log.info("Skipping error_vs_horizon %s: %s", mode, e)
        return []

    # Aggregate across seeds
    agg = aggregate_seeds(
        df,
        group_cols=["condition", h_col],
        value_cols=["error_mean"],
        agg="mean_std",
    )
    # After aggregation columns are error_mean_mean and error_mean_std
    agg = agg.rename(columns={
        "error_mean_mean": "y_raw",
        "error_mean_std": "y_std",
    })
    agg["y_std"] = agg["y_std"].fillna(0)

    # Apply smoothing if requested
    group_cols = ["condition"]
    if smooth_mode != "none":
        window = smooth_window or 10
        agg, y_col = get_smoothed_column(agg, "y_raw", group_cols, smooth_mode, window, ewma_alpha)
    else:
        agg = agg.rename(columns={"y_raw": "y"})
        y_col = "y"

    agg["ymin"] = agg[y_col] - agg["y_std"]
    agg["ymax"] = agg[y_col] + agg["y_std"]

    agg = add_short_labels(agg)
    conds = sorted(set(agg["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    title = "Relative Error vs Horizon"
    if mode == "vel":
        title += " (velocity)"

    # Build filename with smoothing suffix
    suffix_parts = [mode]
    if smooth_mode == "rolling_mean" and smooth_window:
        suffix_parts.append(f"smooth_rollmean_w{smooth_window}")
    elif smooth_mode == "ewma":
        suffix_parts.append(f"smooth_ewma_a{ewma_alpha:.2f}")
    suffix = "__" + "__".join(suffix_parts) if suffix_parts else ""

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    labs_kwargs = {
        "title": "" if paper else title,
        "x": "Trajectory step",
        "y": "Mean ARE (relative)",
        "colour": "" if paper else "Condition",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(agg, aes(x=h_col, y=y_col, colour="condition_short",
                        fill="condition_short"))
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.2, colour="none")
        + geom_line(size=1.0)
        + col_sc + fill_sc
        + labs(**labs_kwargs)
        + theme
    )
    return save_plot(p, output_dir, f"error_vs_horizon{suffix}")


def _plot_uncertainty_vs_horizon(
    df: pd.DataFrame,
    output_dir: Path,
    normalized: bool = False,
    smooth_mode: str = "none",
    smooth_window: Optional[int] = None,
    ewma_alpha: float = 0.1,
    paper: bool = False,
    paper_font_size: int = 16,
) -> list[Path]:
    """Uncertainty vs trajectory step (raw or z-scored)."""
    if "uncertainty_mean" not in df.columns:
        log.info("Skipping uncertainty_vs_horizon: no uncertainty_mean column")
        return []

    h_col = "trajectory_step"

    if normalized:
        df = robust_zscore(df, "uncertainty_mean", group_col="condition",
                           out_col="u_zscore")
        val_col = "u_zscore"
        y_label = "Robust z-score"
        title = "Uncertainty vs Horizon (z-scored)"
        suffix = "__zscore"
    else:
        val_col = "uncertainty_mean"
        y_label = "Uncertainty"
        title = "Uncertainty vs Horizon"
        suffix = ""

    agg = aggregate_seeds(
        df,
        group_cols=["condition", h_col],
        value_cols=[val_col],
        agg="mean_std",
    )
    y_col_base = f"{val_col}_mean" if f"{val_col}_mean" in agg.columns else val_col
    y_std = f"{val_col}_std" if f"{val_col}_std" in agg.columns else None

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
    conds = sorted(set(agg["condition_short"].tolist()))
    col_sc, fill_sc = colour_scales(conds)

    # Build filename with smoothing suffix
    if smooth_mode == "rolling_mean" and smooth_window:
        suffix += f"__smooth_rollmean_w{smooth_window}"
    elif smooth_mode == "ewma":
        suffix += f"__smooth_ewma_a{ewma_alpha:.2f}"

    theme = paper_theme_no_title(base_size=paper_font_size) if paper else paper_theme()
    labs_kwargs = {
        "title": "" if paper else title,
        "x": "Trajectory step",
        "y": y_label,
        "colour": "" if paper else "Condition",
        "fill": "" if paper else "Condition",
    }
    p = (
        ggplot(agg, aes(x=h_col, y=y_col, colour="condition_short",
                        fill="condition_short"))
        + geom_ribbon(aes(ymin="ymin", ymax="ymax"), alpha=0.2, colour="none")
        + geom_line(size=1.0)
        + col_sc + fill_sc
        + labs(**labs_kwargs)
        + theme
    )
    return save_plot(p, output_dir, f"uncertainty_vs_horizon{suffix}")


def plot_long_horizon(
    hall_df: Optional[pd.DataFrame],
    eu_df: Optional[pd.DataFrame],
    output_dir: Path,
    summary: list[str],
    plot_config: dict,
) -> list[Path]:
    """Generate all long-horizon plots.

    Parameters
    ----------
    hall_df : hallucination_horizon data (error curves, horizon_t = prediction steps)
    eu_df   : error_vs_uncertainty data (uncertainty curves, horizon_t = trajectory steps)
    plot_config : dict
        Configuration dict containing smoothing parameters.
    """
    out = output_dir / "long_horizon"
    saved: list[Path] = []

    # Extract config
    smooth_mode = plot_config.get("smooth_mode", "none")
    smooth_window = plot_config.get("smooth_window")
    ewma_alpha = plot_config.get("ewma_alpha", 0.1)
    paper = plot_config.get("paper", False)
    paper_font_size = int(plot_config.get("paper_font_size", 16))
    plot_error = plot_config.get("plot_error", True)
    plot_uncertainty = plot_config.get("plot_uncertainty", True)

    # Detect history horizon from eu_df (it has a history_horizon column)
    history_horizon = _detect_history_horizon(eu_df)

    # --- Error vs horizon ---
    err_src = hall_df if hall_df is not None else eu_df
    if plot_error and err_src is not None:
        err_src = _add_trajectory_step(err_src, history_horizon)
        # Always produce __full
        saved.extend(_plot_error_vs_horizon(
            err_src, "full", out, smooth_mode, smooth_window, ewma_alpha,
            paper=paper, paper_font_size=paper_font_size,
        ))
        # Produce __vel only if velocity columns exist
        if has_velocity_columns(err_src):
            saved.extend(_plot_error_vs_horizon(
                err_src, "vel", out, smooth_mode, smooth_window, ewma_alpha,
                paper=paper, paper_font_size=paper_font_size,
            ))
        else:
            log.info("Velocity columns not found; skipping __vel plots")
    elif plot_error:
        log.info("No error data for long-horizon plots")

    # --- Uncertainty vs horizon ---
    if plot_uncertainty and eu_df is not None:
        eu_with_step = _add_trajectory_step(eu_df, history_horizon)
        saved.extend(_plot_uncertainty_vs_horizon(
            eu_with_step, out, normalized=False,
            smooth_mode=smooth_mode, smooth_window=smooth_window, ewma_alpha=ewma_alpha,
            paper=paper, paper_font_size=paper_font_size,
        ))
        saved.extend(_plot_uncertainty_vs_horizon(
            eu_with_step, out, normalized=True,
            smooth_mode=smooth_mode, smooth_window=smooth_window, ewma_alpha=ewma_alpha,
            paper=paper, paper_font_size=paper_font_size,
        ))
    elif plot_uncertainty:
        log.info("No uncertainty data for long-horizon plots")

    summary.append(f"plot_long_horizon: {len(saved)} plots saved")
    return saved
