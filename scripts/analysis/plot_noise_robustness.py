#!/usr/bin/env python3
"""
Plot C: Noise Robustness (OOD Prediction Error)

Creates publication-quality plots showing prediction error vs input noise level.
This visualizes how world model predictions degrade under out-of-distribution inputs.

Features:
- Line plots and bar plots showing error vs noise level
- Aggregates across seeds with mean/std bands
- Supports filtering conditions via --conditions flag

Usage:
    # Basic plot from noise robustness CSVs
    python plot_noise_robustness.py \
        --data_dir results/paper/noise_robustness \
        --output_dir results/paper/figures
    
    # Filter to specific conditions
    python plot_noise_robustness.py \
        --data_dir results/paper/noise_robustness \
        --output_dir results/paper/figures \
        --conditions bs-nopen,rp-pen025-std
    
    # Specify output format
    python plot_noise_robustness.py \
        --data_dir results/paper/noise_robustness \
        --output_dir results/paper/figures \
        --format pdf
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import shared utilities
from plot_utils import ColorManager, get_label, setup_publication_style, condition_matches_any, load_custom_colors

# Setup publication style
setup_publication_style()

# Create a module-level color manager for consistent colors across all plots
_color_manager = ColorManager()


def load_noise_robustness_data(data_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load noise robustness data from CSV files and aggregate by condition.
    
    Supports two file formats:
    1. Combined file: noise_robustness_all.csv (contains all seeds)
    2. Aggregated file: noise_robustness_aggregated.csv (already aggregated)
    3. Corrected evaluation: noise_robustness_corrected.csv
    4. TensorBoard export: noise_robustness_data.csv
    
    Returns:
        - Combined DataFrame with all data (or None if only aggregated data)
        - Dict mapping condition -> aggregated DataFrame (mean/std across seeds)
    """
    data_path = Path(data_dir)
    
    # Try to load files in order of preference
    possible_files = [
        "noise_robustness_all.csv",          # Preferred: all seeds
        "noise_robustness_corrected.csv",    # Corrected evaluation (all seeds)
        "noise_robustness_aggregated.csv",   # Already aggregated
        "noise_robustness_data.csv",         # TensorBoard export
    ]
    
    all_df = None
    
    for filename in possible_files:
        filepath = data_path / filename
        if filepath.exists():
            all_df = pd.read_csv(filepath)
            print(f"  Loaded: {filepath} ({len(all_df)} rows)")
            break
    
    if all_df is None:
        # Fallback: try to find any CSV files with noise data
        csv_files = list(data_path.glob("noise_robustness*.csv"))
        if csv_files:
            all_df = pd.read_csv(csv_files[0])
            print(f"  Loaded: {csv_files[0]} ({len(all_df)} rows)")
        else:
            print(f"  No noise robustness data found in {data_dir}")
            return None, {}
    
    # Aggregate by condition
    condition_data = {}
    
    if 'condition' not in all_df.columns:
        print("  Warning: 'condition' column not found in data")
        return all_df, {}
    
    for condition in all_df['condition'].unique():
        cond_df = all_df[all_df['condition'] == condition]
        
        # Check if we have per-seed data or already aggregated
        if 'seed' in cond_df.columns and cond_df['seed'].nunique() > 1:
            n_seeds = cond_df['seed'].nunique()
            
            # Group by noise_level and aggregate across seeds
            agg_records = []
            for noise_level, group in cond_df.groupby('noise_level'):
                agg_records.append({
                    'noise_level': noise_level,
                    'error_mean': group['error_mean'].mean(),
                    'error_std': group['error_mean'].std() if len(group) > 1 else group.get('error_std', pd.Series([0])).values[0],
                    'n_seeds': len(group),
                })
            
            agg_df = pd.DataFrame(agg_records)
            condition_data[condition] = agg_df
            print(f"  Aggregated: {condition} ({n_seeds} seeds, {len(agg_df)} noise levels)")
        else:
            # Already aggregated or single seed
            condition_data[condition] = cond_df.copy()
            print(f"  Loaded: {condition} ({len(cond_df)} noise levels)")
    
    return all_df, condition_data


def plot_noise_robustness_line(
    condition_data: dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create line plot showing error vs noise level.
    
    Args:
        condition_data: Dict mapping condition name to DataFrame
        ax: Matplotlib axes to plot on
        figsize: Figure size if creating new figure
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    for condition, df in condition_data.items():
        if df.empty:
            print(f"  Warning: No data for {condition}")
            continue
        
        # Get color and label
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        # Get data
        noise_levels = df['noise_level'].values
        errors = df['error_mean'].values
        stds = df.get('error_std', pd.Series([0] * len(df))).values
        
        # Plot
        ax.plot(noise_levels, errors, 'o-', color=color, label=label, 
                markersize=8, linewidth=2)
        ax.fill_between(noise_levels, errors - stds, errors + stds, 
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Input Noise Level (σ)', fontsize=12)
    ax.set_ylabel('Prediction Error (relative)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Prediction Error vs Input Noise', fontsize=14)
    
    ax.legend(loc='best')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_noise_robustness_bar(
    condition_data: dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create grouped bar plot for noise robustness.
    
    Args:
        condition_data: Dict mapping condition name to DataFrame
        ax: Matplotlib axes to plot on
        figsize: Figure size if creating new figure
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get all noise levels
    all_noise_levels = set()
    for df in condition_data.values():
        all_noise_levels.update(df['noise_level'].values)
    noise_levels = sorted(all_noise_levels)
    
    x = np.arange(len(noise_levels))
    n_conditions = len(condition_data)
    width = 0.8 / max(n_conditions, 1)
    
    for i, (condition, df) in enumerate(condition_data.items()):
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        # Align to noise_levels
        errors = []
        stds = []
        for noise in noise_levels:
            row = df[df['noise_level'] == noise]
            if len(row) > 0:
                errors.append(row['error_mean'].values[0])
                stds.append(row.get('error_std', pd.Series([0])).values[0])
            else:
                errors.append(np.nan)
                stds.append(np.nan)
        
        offset = (i - n_conditions / 2 + 0.5) * width
        ax.bar(x + offset, errors, width, yerr=stds, label=label, 
               color=color, alpha=0.8, capsize=3)
    
    ax.set_xlabel('Input Noise Level (σ)', fontsize=12)
    ax.set_ylabel('Prediction Error (relative)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Prediction Error vs Input Noise', fontsize=14)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in noise_levels])
    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    return fig


def plot_noise_degradation_rate(
    condition_data: dict[str, pd.DataFrame],
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Create bar plot showing error degradation rate (slope of error vs noise).
    
    A lower degradation rate indicates more robust predictions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    conditions = []
    slopes = []
    colors = []
    
    for condition, df in condition_data.items():
        if len(df) < 2:
            continue
        
        # Compute slope (degradation rate)
        noise_levels = df['noise_level'].values
        errors = df['error_mean'].values
        
        # Linear regression
        slope, intercept = np.polyfit(noise_levels, errors, 1)
        
        conditions.append(get_label(condition))
        slopes.append(slope)
        colors.append(_color_manager.get_color(condition))
    
    # Sort by slope
    sorted_idx = np.argsort(slopes)
    conditions = [conditions[i] for i in sorted_idx]
    slopes = [slopes[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    ax.barh(conditions, slopes, color=colors, alpha=0.8)
    ax.set_xlabel('Error Degradation Rate (slope)')
    ax.set_title('Noise Robustness: Error Degradation Rate')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig


def create_paper_figures(
    condition_data: dict[str, pd.DataFrame],
    output_dir: Path,
    format: str = 'pdf',
):
    """
    Create publication-ready noise robustness figures.
    """
    # Main figure: line plot
    print("\nCreating noise robustness line plot...")
    fig = plot_noise_robustness_line(
        condition_data,
        title='Prediction Error vs Input Noise'
    )
    output_path = output_dir / f'noise_robustness_line.{format}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # Bar plot
    print("\nCreating noise robustness bar plot...")
    fig = plot_noise_robustness_bar(
        condition_data,
        title='Prediction Error vs Input Noise'
    )
    output_path = output_dir / f'noise_robustness_bar.{format}'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # Degradation rate plot (if we have enough data)
    if any(len(df) >= 2 for df in condition_data.values()):
        print("\nCreating degradation rate plot...")
        fig = plot_noise_degradation_rate(condition_data)
        output_path = output_dir / f'noise_degradation_rate.{format}'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)
    
    # Save combined summary data as CSV for reference
    combined_records = []
    for condition, df in condition_data.items():
        for _, row in df.iterrows():
            record = row.to_dict()
            record['condition'] = condition
            combined_records.append(record)
    
    if combined_records:
        combined_df = pd.DataFrame(combined_records)
        csv_path = output_dir / 'noise_robustness_summary.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot noise robustness (prediction error vs input noise)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing noise robustness CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as data_dir)')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for figures')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated list of conditions to include (default: all)')
    parser.add_argument('--colors', type=str, default=None,
                        help='Path to YAML file with custom color mappings')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively instead of saving')

    args = parser.parse_args()

    # Load custom colors if specified
    load_custom_colors(args.colors)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.data_dir)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.data_dir}")
    all_df, condition_data = load_noise_robustness_data(args.data_dir)
    
    if not condition_data:
        print("No noise robustness data found!")
        print("Expected files: noise_robustness_*.csv")
        return
    
    print(f"\nLoaded {len(condition_data)} conditions")
    
    # Filter conditions if specified
    if args.conditions:
        include_conditions = [c.strip() for c in args.conditions.split(',')]
        filtered_data = {}
        for cond in condition_data:
            if condition_matches_any(cond, include_conditions):
                filtered_data[cond] = condition_data[cond]
        
        if not filtered_data:
            print(f"WARNING: No conditions matched filter '{args.conditions}'")
            print(f"Available conditions: {list(condition_data.keys())}")
        else:
            print(f"Filtered to {len(filtered_data)} conditions: {list(filtered_data.keys())}")
            condition_data = filtered_data
    
    if args.show:
        # Interactive mode
        fig = plot_noise_robustness_line(condition_data)
        plt.show()
    else:
        # Save figures
        create_paper_figures(condition_data, output_dir, args.format)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
