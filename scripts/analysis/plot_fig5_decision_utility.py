#!/usr/bin/env python3
"""
Figure 5: Decision Utility - Risk Filtering & AUROC

Creates publication-quality plots showing:
- (a) Risk-coverage curve: Mean error vs uncertainty percentile
- (b) AUROC vs horizon: Bar chart of AUROC scores for long horizons

This evaluates uncertainty as a predictor of high prediction errors.

Usage:
    python plot_fig5_decision_utility.py \
        --results_dir results/paper \
        --output_dir results/paper/figures/fig5 \
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


def plot_risk_coverage(
    data: dict[str, pd.DataFrame],
    velocity_only: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot risk-coverage curve.

    Shows mean prediction error (risk) as a function of coverage fraction.
    Lower coverage = more filtering by low uncertainty.

    Args:
        data: Dictionary of condition -> DataFrame with risk-coverage data
        velocity_only: If True, use velocity-only error (recommended for Anymal)
        ax: Optional axis to plot on
        figsize: Figure size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Plot separate curves for each condition
    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue

        if 'uncertainty_percentile' not in df.columns or 'mean_error_filtered' not in df.columns:
            continue

        # Aggregate by percentile to get mean error across runs
        percentiles = sorted(df['uncertainty_percentile'].unique())
        mean_errors = []

        for pct in percentiles:
            df_pct = df[df['uncertainty_percentile'] == pct]
            mean_error = df_pct['mean_error_filtered'].mean()
            mean_errors.append(mean_error)

        color = _color_manager.get_color(condition)
        label = get_label(condition)
        ax.plot(percentiles, mean_errors, '-', linewidth=2, color=color, label=label, alpha=0.8)

    # Add reference lines (use median of all conditions)
    all_no_filter_errors = []
    all_80_cov_errors = []
    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue
        df_no_filter = df[df['uncertainty_percentile'] == 100]['mean_error_filtered'].mean()
        df_80_cov = df[df['coverage_fraction'] >= 0.8].iloc[0]['mean_error_filtered']
        all_no_filter_errors.append(df_no_filter)
        all_80_cov_errors.append(df_80_cov)
    
    if all_no_filter_errors:
        ax.axhline(y=np.mean(all_no_filter_errors), color='r', linestyle='--', alpha=0.5, label='No Filtering')
    if all_80_cov_errors:
        ax.axhline(y=np.mean(all_80_cov_errors), color='g', linestyle=':', alpha=0.5, label='80% Coverage')
    
    # Rename x-axis to "Coverage (kept fraction)" as per feedback
    # Note: Percentiles are computed per-condition, so coverage is relative to each condition's uncertainty ranking
    ax.set_xlabel('Coverage (kept fraction of lowest-uncertainty predictions)')
    ax.set_ylabel('Mean Prediction Error')
    error_type = "Velocity-Only" if velocity_only else "Full-State"
    ax.set_title(f'Risk-Coverage Curve ({error_type})')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    return fig


def plot_auroc_vs_horizon(
    data: dict[str, pd.DataFrame],
    horizon_filter: int = 30,
    plot_type: str = 'line',
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot AUROC vs horizon.

    Args:
        data: Dictionary of condition -> DataFrame with AUROC data
        horizon_filter: Minimum horizon for aggregation (used for bar plot)
        plot_type: 'line' for AUROC vs horizon line plot, 'bar' for aggregated bar chart
        ax: Optional axis to plot on
        figsize: Figure size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if plot_type == 'line':
        # Plot AUROC vs horizon as line plot (recommended)
        for condition, df in data.items():
            if df is None or len(df) == 0:
                continue

            if 'auroc' not in df.columns or 'horizon_t' not in df.columns:
                continue

            # Filter by horizon_filter
            df_filtered = df[df['horizon_t'] >= horizon_filter].copy()

            if len(df_filtered) == 0:
                continue

            # Group by horizon and compute mean AUROC
            horizon_auroc = df_filtered.groupby('horizon_t')['auroc'].agg(['mean', 'std'])

            x = horizon_auroc.index.values
            y = horizon_auroc['mean'].values
            yerr = horizon_auroc['std'].values

            color = _color_manager.get_color(condition)
            label = get_label(condition)

            ax.plot(x, y, '-', linewidth=2, color=color, label=label)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)

        ax.set_xlabel('Horizon Step')
        ax.set_ylabel('AUROC')
        ax.set_title(f'AUROC for High-Error Detection (h > {horizon_filter})')
        ax.legend(loc='lower left', framealpha=1.0, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)

    else:
        # Plot aggregated bar chart
        cond_auroc = {}
        cond_auroc_std = {}

        for condition, df in data.items():
            if df is None or len(df) == 0:
                continue

            if 'auroc' not in df.columns:
                continue

            if horizon_filter is not None and 'horizon_t' in df.columns:
                df_filtered = df[df['horizon_t'] >= horizon_filter]
            else:
                df_filtered = df.copy()

            if len(df_filtered) > 0:
                cond_auroc[condition] = df_filtered['auroc'].mean()
                cond_auroc_std[condition] = df_filtered['auroc'].std()

        if len(cond_auroc) == 0:
            print("  WARNING: No AUROC data found")
            return None

        # Plot bar chart
        x = np.arange(len(cond_auroc))
        width = 0.6

        for i, condition in enumerate(cond_auroc.keys()):
            color = _color_manager.get_color(condition)
            mean_val = cond_auroc[condition]
            std_val = cond_auroc_std.get(condition, 0)

            ax.bar(x[i], mean_val, width, color=color, alpha=0.8, yerr=std_val, capsize=3)

        ax.set_xlabel('Condition')
        ax.set_ylabel(f'AUROC (h > {horizon_filter})')
        ax.set_title(f'AUROC for High-Error Detection (h > {horizon_filter})')
        ax.set_xticks(x)
        ax.set_xticklabels([get_label(c) for c in cond_auroc.keys()], rotation=15, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

    return fig


def load_risk_coverage_data(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load risk coverage data."""
    data = {}

    risk_dir = results_dir / 'risk_coverage'
    if risk_dir.exists():
        for csv_file in risk_dir.glob("risk_coverage_*.csv"):
            if 'all' in csv_file.name:
                continue

            condition = csv_file.stem.replace('risk_coverage_', '')

            try:
                df = pd.read_csv(csv_file)
                data[condition] = df
            except Exception as e:
                print(f"  WARNING: Could not load {csv_file}: {e}")

    return data


def load_auroc_data(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load AUROC data."""
    data = {}

    auroc_dir = results_dir / 'auroc'
    if auroc_dir.exists():
        for csv_file in auroc_dir.glob("auroc_*.csv"):
            if 'all' in csv_file.name:
                continue

            condition = csv_file.stem.replace('auroc_', '')

            try:
                df = pd.read_csv(csv_file)
                data[condition] = df
            except Exception as e:
                print(f"  WARNING: Could not load {csv_file}: {e}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 5: Decision Utility - Risk Filtering & AUROC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for figures')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated conditions to plot (default: all)')
    parser.add_argument('--horizon_filter', type=int, default=30,
                        help='Minimum horizon for AUROC aggregation (default: 30)')
    parser.add_argument('--velocity_only', action='store_true', default=True,
                        help='Use velocity-only error for risk-coverage (recommended for Anymal)')
    parser.add_argument('--full_state', action='store_true',
                        help='Use full-state error instead of velocity-only')
    parser.add_argument('--auroc_plot_type', type=str, default='line',
                        choices=['line', 'bar'],
                        help='AUROC plot type: line (AUROC vs horizon) or bar (aggregated)')
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
    print(f"AUROC plot type: {args.auroc_plot_type}")

    # Load data
    risk_coverage_data = load_risk_coverage_data(results_dir)
    auroc_data = load_auroc_data(results_dir)

    print(f"Loaded {len(risk_coverage_data)} risk coverage datasets")
    print(f"Loaded {len(auroc_data)} AUROC datasets")

    # Filter conditions
    if args.conditions:
        include_conditions = [c.strip() for c in args.conditions.split(',')]
        filtered_risk = {
            cond: df for cond, df in risk_coverage_data.items()
            if condition_matches_any(cond, include_conditions)
        }
        filtered_auroc = {
            cond: df for cond, df in auroc_data.items()
            if condition_matches_any(cond, include_conditions)
        }
        risk_coverage_data = filtered_risk
        auroc_data = filtered_auroc
        print(f"Filtered to {len(risk_coverage_data)} conditions")

    # Plot (a) Risk-coverage curve
    print("\nCreating risk-coverage curve...")
    print(f"  Note: Using h > {args.horizon_filter} for consistency with AUROC")
    fig = plot_risk_coverage(risk_coverage_data, velocity_only=use_velocity_only)
    if fig is not None:
        output_path = output_dir / f'fig5a_risk_coverage.{args.format}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_path}")

        # Print Risk@80% values
        print("\n" + "="*60)
        print("Risk@80% Coverage Values:")
        print("="*60)
        for condition, df in risk_coverage_data.items():
            if f'risk_at_80_coverage' in df.columns:
                risk_80 = df[f'risk_at_80_coverage'].dropna()
                if len(risk_80) > 0:
                    print(f"{condition}: {risk_80.mean():.4f}")
        print("="*60)

    # Plot (b) AUROC vs horizon
    print("\nCreating AUROC vs horizon plot...")
    fig = plot_auroc_vs_horizon(
        auroc_data,
        horizon_filter=args.horizon_filter,
        plot_type=args.auroc_plot_type
    )
    if fig is not None:
        output_path = output_dir / f'fig5b_auroc_vs_horizon.{args.format}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
