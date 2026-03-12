#!/usr/bin/env python3
"""
Figure 3: Hallucination Horizon & Growth Rates Plots

Creates publication-quality plots showing:
- (a) Hallucination horizon distribution (box/violin plot)
- (b) Relationship between error growth rate and uncertainty growth rate

This shows model reliability over long horizons.

Usage:
    python plot_fig3_hallucination.py \
        --results_dir results/paper \
        --output_dir results/paper/figures/fig3 \
        --conditions bs-nopen,rp-pen025-std
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Import shared utilities
from plot_utils import (
    ColorManager, get_label, setup_publication_style,
    condition_matches_any, PAPER_CONDITION_COLORS, PAPER_CONDITION_ORDER
)

# Setup publication style
setup_publication_style()
_color_manager = ColorManager()


def plot_hallucination_distribution(
    data: dict[str, pd.DataFrame],
    plot_type: str = 'box',
    use_short_labels: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot hallucination horizon distribution.

    Args:
        data: Dictionary of condition -> DataFrame with hallucination horizon data
        plot_type: 'box' for boxplot, 'violin' for violin plot
        use_short_labels: If True, use short labels (B0/Bp/R0/Rp) instead of full labels
        ax: Optional axis to plot on
        figsize: Figure size
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract hallucination horizon values
    conditions = []
    all_h_values = []

    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue

        if 'hallucination_horizon' not in df.columns:
            continue

        h_values = df['hallucination_horizon'].dropna().values
        conditions.append(condition)
        all_h_values.extend(h_values)

    if len(all_h_values) == 0:
        print("  WARNING: No hallucination horizon data found")
        return None

    # Group by condition
    groups = []
    for condition in conditions:
        h_values = data[condition]['hallucination_horizon'].dropna().values
        groups.append(h_values)

    # Add small jitter to zero-variance groups for violin plots
    if plot_type == 'violin':
        groups_with_jitter = []
        for h_values in groups:
            if np.std(h_values) == 0:
                # Add tiny noise to zero-variance data for visualization
                jitter = np.random.normal(0, 0.5, size=len(h_values))
                groups_with_jitter.append(h_values + jitter)
            else:
                groups_with_jitter.append(h_values)
        groups = groups_with_jitter

    # Generate labels
    if use_short_labels:
        # Use short labels (B0, Bp, R0, Rp)
        labels = []
        for condition in conditions:
            # Extract short label from condition name
            # Expected format: bs-nopen, bs-pen025, rp-nopen, rp-pen025, etc.
            # Convert to: B0, Bp, R0, Rp
            cond_lower = condition.lower()
            if 'bs' in cond_lower:
                if 'pen' in cond_lower:
                    labels.append('Bp')
                else:
                    labels.append('B0')
            elif 'rp' in cond_lower:
                if 'pen' in cond_lower:
                    labels.append('Rp')
                else:
                    labels.append('R0')
            else:
                # Fallback to first 2 characters of label
                labels.append(get_label(condition)[:2])
    else:
        labels = [get_label(c) for c in conditions]

    # Plot
    if plot_type == 'box':
        bp = ax.boxplot(groups, labels=labels, patch_artist=True)

        # Color boxes
        colors = [_color_manager.get_color(c) for c in conditions]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Hallucination Horizon (steps)')
        ax.set_title('Hallucination Horizon Distribution')
        ax.grid(True, axis='y', alpha=0.3)

    elif plot_type == 'violin':
        parts = ax.violinplot(groups, positions=range(len(conditions)),
                               showmeans=True, showmedians=True, bw_method='silverman')

        # Color violins
        colors = [_color_manager.get_color(c) for c in conditions]
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Hallucination Horizon (steps)')
        ax.set_title('Hallucination Horizon Distribution')
        ax.grid(True, axis='y', alpha=0.3)

    # Only add legend for box plots (violin plots have x-axis labels)
    if plot_type == 'box' and not use_short_labels:
        ax.legend(loc='lower right', framealpha=1.0)

    return fig


def plot_growth_rates_correlation(
    data: dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """Plot correlation between error growth rate and uncertainty growth rate."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Extract growth rates
    conditions = []
    error_rates = []
    uncertainty_rates = []

    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue

        if 'error_growth_rate' not in df.columns or 'uncertainty_growth_rate' not in df.columns:
            continue

        valid_mask = df['error_growth_rate'].notna() & df['uncertainty_growth_rate'].notna()

        if valid_mask.sum() > 0:
            conditions.append(condition)
            error_rates.extend(df[valid_mask]['error_growth_rate'].values)
            uncertainty_rates.extend(df[valid_mask]['uncertainty_growth_rate'].values)

    if len(conditions) == 0:
        print("  WARNING: No growth rate data found")
        return None

    # Compute correlation
    x = np.array(uncertainty_rates)
    y = np.array(error_rates)

    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 10:
        print("  WARNING: Insufficient data for correlation plot")
        return None

    x_valid = x[mask]
    y_valid = y[mask]

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(x_valid, y_valid)

    # Plot scatter with colors by condition
    for condition, df in data.items():
        if df is None or len(df) == 0:
            continue

        if 'error_growth_rate' not in df.columns:
            continue

        valid_mask = df['error_growth_rate'].notna() & df['uncertainty_growth_rate'].notna()

        if valid_mask.sum() < 5:
            continue

        color = _color_manager.get_color(condition)
        label = get_label(condition)

        ax.scatter(df[valid_mask]['uncertainty_growth_rate'],
                df[valid_mask]['error_growth_rate'],
                color=color, alpha=0.6, label=label, s=30, edgecolors='k', linewidths=0.5)

    ax.set_xlabel('Uncertainty Growth Rate')
    ax.set_ylabel('Error Growth Rate')
    ax.set_title(f'Error vs Uncertainty Growth Rate (r={pearson_r:.3f}, p={pearson_p:.3f})')
    ax.legend(loc='lower right', framealpha=1.0)
    ax.grid(True, alpha=0.3)

    return fig


def load_growth_rates_data(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load growth rates data."""
    data = {}

    growth_dir = results_dir / 'growth_rates'
    if growth_dir.exists():
        for csv_file in growth_dir.glob("growth_rates_*.csv"):
            if 'all' in csv_file.name:
                continue

            condition = csv_file.stem.replace('growth_rates_', '')

            try:
                df = pd.read_csv(csv_file)
                data[condition] = df
            except Exception as e:
                print(f"  WARNING: Could not load {csv_file}: {e}")

    return data


def load_hallucination_stats(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load per-episode hallucination horizon data for distribution plot."""
    data = {}

    stats_dir = results_dir / 'hallucination_stats'
    if stats_dir.exists():
        for csv_file in stats_dir.glob("hallucination_horizon_*.csv"):
            if csv_file.stem.endswith('_all'):
                continue

            condition = csv_file.stem.replace('hallucination_horizon_', '')

            try:
                df = pd.read_csv(csv_file)
                data[condition] = df
            except Exception as e:
                print(f"  WARNING: Could not load {csv_file}: {e}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 3: Hallucination Horizon & Growth Rates',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for figures')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated conditions to plot (default: all)')
    parser.add_argument('--plot_type', type=str, default='box',
                        choices=['box', 'violin'],
                        help='Distribution plot type (default: box)')
    parser.add_argument('--use_short_labels', action='store_true', default=True,
                        help='Use short labels (B0/Bp/R0/Rp) instead of full labels')
    parser.add_argument('--skip_growth_rates', action='store_true',
                        help='Skip growth rates correlation plot (recommended as it makes little sense)')
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

    print(f"Loading data from: {results_dir}")

    # Load hallucination statistics
    hallucination_data = load_hallucination_stats(results_dir)

    # Load growth rates for correlation plot
    growth_rates_data = load_growth_rates_data(results_dir)

    print(f"Loaded {len(hallucination_data)} hallucination datasets")
    print(f"Loaded {len(growth_rates_data)} growth rate datasets")

    # Filter conditions
    if args.conditions:
        include_conditions = [c.strip() for c in args.conditions.split(',')]
        filtered_hallucination = {
            cond: df for cond, df in hallucination_data.items()
            if condition_matches_any(cond, include_conditions)
        }
        filtered_growth = {
            cond: df for cond, df in growth_rates_data.items()
            if condition_matches_any(cond, include_conditions)
        }
        hallucination_data = filtered_hallucination
        growth_rates_data = filtered_growth
        print(f"Filtered to {len(hallucination_data)} conditions")

    # Plot (a) Hallucination distribution
    print("\nCreating hallucination distribution plot...")
    fig = plot_hallucination_distribution(
        hallucination_data,
        plot_type=args.plot_type,
        use_short_labels=args.use_short_labels
    )
    if fig is not None:
        output_path = output_dir / f'fig3a_hallucination_distribution.{args.format}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {output_path}")

    # Plot (b) Growth rates correlation (optional)
    if not args.skip_growth_rates:
        print("\nCreating growth rates correlation plot...")
        print("  NOTE: This plot is often not meaningful for publication (pooled across conditions)")
        fig = plot_growth_rates_correlation(growth_rates_data)
        if fig is not None:
            output_path = output_dir / f'fig3b_growth_rates_correlation.{args.format}'
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {output_path}")
    else:
        print("\nSkipping growth rates correlation plot (as requested)")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary: Hallucination Horizon")
    print("="*60)
    for condition, df in hallucination_data.items():
        if 'hallucination_horizon_mean' in df.columns:
            print(f"\n{condition}:")
            print(f"  Mean: {df['hallucination_horizon_mean'].iloc[0]:.1f}")
            print(f"  Median: {df['hallucination_horizon_median'].iloc[0]:.1f}")
            if 'fraction_short_hallucination' in df.columns:
                print(f"  Fraction short (<30): {df['fraction_short_hallucination'].iloc[0]:.1%}")

    print("\n" + "="*60)
    print("Done!")


if __name__ == '__main__':
    main()
