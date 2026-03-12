#!/usr/bin/env python3
"""
Figure 2: Long-Horizon Behavior Plots

Creates publication-quality plots showing:
- Error vs horizon (mean ± CI)
- Epistemic uncertainty vs horizon (mean ± CI)
- Both velocity-only and full-state variants

This demonstrates long-horizon behavior and robustness.

Usage:
    python plot_fig2_long_horizon.py \
        --results_dir results/paper \
        --output_dir results/paper/figures/fig2 \
        --conditions bs-nopen,rp-pen025-std
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import shared utilities
from plot_utils import (
    ColorManager, get_label, setup_publication_style,
    condition_matches_any, PAPER_CONDITION_COLORS, PAPER_CONDITION_ORDER
)

# Setup publication style
setup_publication_style()
_color_manager = ColorManager()


def plot_long_horizon_error(
    data: dict[str, pd.DataFrame],
    velocity_only: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot error vs horizon with confidence intervals.

    For Anymal locomotion, velocity-only is recommended as it uses v, w, q_dot
    which are directly control-relevant and scale-consistent.

    Args:
        data: Dictionary of condition -> DataFrame with per-step error data
        velocity_only: If True, plot velocity-only error (recommended for Anymal)
        ax: Optional axis to plot on
        figsize: Figure size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue

        # Filter by horizon (start from 1, not history_horizon)
        # This ensures all steps are shown for proper interpretation
        if 'horizon_t' in df.columns:
            df = df[df['horizon_t'] >= 1].copy()
        elif 'prediction_step' in df.columns:
            df = df[df['prediction_step'] >= 1].copy()

        if len(df) == 0:
            continue

        color = _color_manager.get_color(condition)
        label = get_label(condition)

        # Group by horizon step
        if 'horizon_t' in df.columns:
            horizon_col = 'horizon_t'
        elif 'prediction_step' in df.columns:
            horizon_col = 'prediction_step'
        else:
            continue

        # Select error column based on velocity_only flag
        if velocity_only:
            # Use velocity-only error (v, w, q_dot) for Anymal
            error_mean_col = f'error_vel_rel_mean' if f'error_vel_rel_mean' in df.columns else 'error_rel_mean'
            error_std_col = f'error_vel_rel_std' if f'error_vel_rel_std' in df.columns else 'error_rel_std'
        else:
            # Use full-state error (all components)
            error_mean_col = 'error_rel_mean'
            error_std_col = 'error_rel_std'

        grouped = df.groupby(horizon_col)

        # Plot mean with CI
        if error_mean_col in df.columns:
            means = grouped[error_mean_col].mean()
            stds = grouped[error_mean_col].std()

            x = means.index
            y = means.values

            ax.plot(x, y, color=color, label=label, linewidth=2)
            ax.fill_between(x, y - stds, y + stds, alpha=0.2, color=color)

    error_type = "Velocity-Only" if velocity_only else "Full-State"
    ax.set_xlabel('Horizon Step')
    ax.set_ylabel('Prediction Error (relative)')
    ax.set_title(f'Long-Horizon Error Growth ({error_type})')
    ax.legend(loc='lower right', framealpha=1.0)
    ax.grid(True, alpha=0.3)

    return fig


def plot_long_horizon_uncertainty(
    data: dict[str, pd.DataFrame],
    normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot epistemic uncertainty vs horizon with confidence intervals.

    Args:
        data: Dictionary of condition -> DataFrame with per-step uncertainty data
        normalize: If True, normalize uncertainty per condition using robust z-score
                   (median and IQR) to enable fair comparison across mechanisms
        ax: Optional axis to plot on
        figsize: Figure size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue

        # Filter by horizon (start from 1)
        if 'horizon_t' in df.columns:
            df = df[df['horizon_t'] >= 1].copy()
        elif 'prediction_step' in df.columns:
            df = df[df['prediction_step'] >= 1].copy()

        if len(df) == 0:
            continue

        color = _color_manager.get_color(condition)
        label = get_label(condition)

        # Group by horizon step
        if 'horizon_t' in df.columns:
            horizon_col = 'horizon_t'
        elif 'prediction_step' in df.columns:
            horizon_col = 'prediction_step'
        else:
            continue

        grouped = df.groupby(horizon_col)

        # Plot mean uncertainty with CI
        if 'uncertainty_mean' in df.columns:
            means = grouped['uncertainty_mean'].mean()
            stds = grouped['uncertainty_mean'].std()

            x = means.index
            y = means.values

            # Normalize per condition if requested
            if normalize:
                # Use robust z-score: (x - median) / IQR
                all_uncertainty = df['uncertainty_mean'].values
                median = np.median(all_uncertainty)
                q75 = np.percentile(all_uncertainty, 75)
                q25 = np.percentile(all_uncertainty, 25)
                iqr = q75 - q25
                if iqr > 1e-8:
                    y = (y - median) / iqr
                    stds = stds / iqr

            ax.plot(x, y, color=color, label=label, linewidth=2)
            ax.fill_between(x, y - stds, y + stds, alpha=0.2, color=color)

    ylabel = 'Epistemic Uncertainty (robust z-score, normalized per condition)' if normalize else 'Epistemic Uncertainty'
    ax.set_xlabel('Horizon Step')
    ax.set_ylabel(ylabel)
    ax.set_title('Epistemic Uncertainty Growth (normalized per condition)')
    ax.legend(loc='lower right', framealpha=1.0, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    return fig


def load_growth_rates(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load growth rates data for display."""
    data = {}

    for csv_file in data_dir.glob("growth_rates_*.csv"):
        if 'all' in csv_file.name:
            continue

        # Extract condition name from filename
        condition = csv_file.stem.replace('growth_rates_', '')

        try:
            df = pd.read_csv(csv_file)
            data[condition] = df
        except Exception as e:
            print(f"  WARNING: Could not load {csv_file}: {e}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 2: Long-Horizon Behavior Plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for figures')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated conditions to plot (default: all)')
    parser.add_argument('--velocity_only', action='store_true', default=True,
                        help='Plot velocity-only errors (recommended for Anymal)')
    parser.add_argument('--full_state', action='store_true',
                        help='Plot full-state errors instead of velocity-only')
    parser.add_argument('--show_combined', action='store_true', default=True,
                        help='Create combined 2-panel figure')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format (default: pdf)')
    parser.add_argument('--colors', type=str, default=None,
                        help='Path to YAML file with custom color mappings')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load custom colors if specified
    if args.colors:
        from plot_utils import load_custom_colors
        load_custom_colors(args.colors)

    # Determine whether to use velocity_only
    use_velocity_only = args.velocity_only and not args.full_state

    print(f"Loading data from: {results_dir}")
    print(f"Velocity-only mode: {use_velocity_only}")

    # Load error_vs_uncertainty data for error and uncertainty curves
    error_data = {}
    error_data_dir = results_dir / 'error_vs_uncertainty'
    if error_data_dir.exists():
        for csv_file in sorted(error_data_dir.glob("*.csv")):
            if 'all' in csv_file.name:
                continue

            # Parse condition from filename
            if 'error_vs_uncertainty_' in csv_file.name:
                condition = csv_file.stem.replace('error_vs_uncertainty_', '')

                try:
                    df = pd.read_csv(csv_file)
                    error_data[condition] = df
                except Exception as e:
                    print(f"  WARNING: Could not load {csv_file}: {e}")

    print(f"Loaded {len(error_data)} error datasets")

    # Filter conditions
    if args.conditions:
        include_conditions = [c.strip() for c in args.conditions.split(',')]
        filtered_data = {
            cond: df for cond, df in error_data.items()
            if condition_matches_any(cond, include_conditions)
        }
        error_data = filtered_data
        print(f"Filtered to {len(error_data)} conditions: {list(error_data.keys())}")

    # Load growth rates for display
    growth_data = load_growth_rates(results_dir / 'growth_rates')

    # Plot individual figures
    print("\nCreating individual figures...")

    # (a) Error vs horizon
    print("  Creating error vs horizon plot...")
    fig = plot_long_horizon_error(error_data, velocity_only=use_velocity_only)
    output_path = output_dir / f'fig2a_error_vs_horizon.{args.format}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_path}")

    # (b) Uncertainty vs horizon
    print("  Creating uncertainty vs horizon plot...")
    fig = plot_long_horizon_uncertainty(error_data, normalize=True)
    output_path = output_dir / f'fig2b_uncertainty_vs_horizon.{args.format}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_path}")

    # Combined 2-panel figure
    if args.show_combined:
        print("\nCreating combined figure...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        plot_long_horizon_error(error_data, velocity_only=use_velocity_only, ax=axes[0], figsize=(14, 5))
        plot_long_horizon_uncertainty(error_data, normalize=True, ax=axes[1], figsize=(14, 5))

        # Keep legend on first plot only
        axes[1].get_legend().remove()

        fig.tight_layout()
        output_path = output_dir / f'fig2_long_horizon_combined.{args.format}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_path}")

    # Print summary table in console
    print("\n" + "="*60)
    print("Summary: Growth Rates")
    print("="*60)
    for condition, df in growth_data.items():
        if 'error_growth_rate' in df.columns and 'uncertainty_growth_rate' in df.columns:
            error_rate = df['error_growth_rate'].mean()
            uncertainty_rate = df['uncertainty_growth_rate'].mean()
            print(f"\n{condition}:")
            print(f"  Error growth rate: {error_rate:.6f}")
            print(f"  Uncertainty growth rate: {uncertainty_rate:.6f}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == '__main__':
    main()
