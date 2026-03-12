#!/usr/bin/env python3
"""
Plot Training Curves from TensorBoard Data

Creates publication-quality plots showing training progress over time:
- Mean episode reward
- Epistemic uncertainty
- Autoregressive prediction error

Features:
- Reads aggregated CSV files from extract_tensorboard_data.py
- Supports condition filtering
- Configurable smoothing windows
- Consistent colors via plot_utils

Usage:
    # Basic plot from TensorBoard CSVs
    python plot_training_curves.py \
        --data_dir results/paper/tensorboard/aggregated \
        --output_dir results/paper/figures/training_curves
    
    # With condition filtering
    python plot_training_curves.py \
        --data_dir results/paper/tensorboard/aggregated \
        --output_dir results/paper/figures/training_curves \
        --conditions bs-nopen,rp-pen025-std
    
    # Custom smoothing
    python plot_training_curves.py \
        --data_dir results/paper/tensorboard/aggregated \
        --output_dir results/paper/figures/training_curves \
        --smooth_window 30 --autoregressive_smooth_window 10
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


def load_aggregated_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load aggregated CSV files from data directory.
    
    Each CSV file is named by condition (e.g., 'bs-nopen.csv') and contains
    columns like 'step', 'Train_mean_reward_mean', 'Train_mean_reward_std', etc.
    
    Returns:
        Dict mapping condition name to DataFrame
    """
    data_path = Path(data_dir)
    data = {}
    for csv_file in data_path.glob('*.csv'):
        if csv_file.stem == 'all_runs':
            continue  # Skip the raw all_runs file
        condition = csv_file.stem
        df = pd.read_csv(csv_file)
        data[condition] = df
    return data


def plot_mean_reward(
    data: dict[str, pd.DataFrame],
    smooth_window: int = 20,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mean episode reward over training.
    
    Args:
        data: Dict mapping condition name to DataFrame
        smooth_window: Rolling window for smoothing
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
    
    for condition, df in data.items():
        if 'Train_mean_reward_mean' not in df.columns:
            continue
        
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        steps = df['step'].values
        mean = df['Train_mean_reward_mean'].values
        std = df.get('Train_mean_reward_std', pd.Series([0] * len(df))).values
        
        # Smooth
        mean_smooth = pd.Series(mean).rolling(smooth_window, min_periods=1).mean().values
        std_smooth = pd.Series(std).rolling(smooth_window, min_periods=1).mean().values
        
        ax.plot(steps, mean_smooth, color=color, label=label)
        ax.fill_between(steps, mean_smooth - std_smooth, mean_smooth + std_smooth, 
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title(title or 'Policy Learning Curves')
    ax.legend(loc='lower right')
    
    return fig


def plot_epistemic_uncertainty(
    data: dict[str, pd.DataFrame],
    smooth_window: int = 20,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot epistemic uncertainty over training.
    
    Args:
        data: Dict mapping condition name to DataFrame
        smooth_window: Rolling window for smoothing
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
    
    for condition, df in data.items():
        col = 'Model_Based_epistemic_uncertainty_mean'
        if col not in df.columns:
            continue
        
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        steps = df['step'].values
        mean = df[col].values
        std = df.get('Model_Based_epistemic_uncertainty_std', pd.Series([0] * len(df))).values
        
        valid = ~np.isnan(mean)
        if valid.sum() < smooth_window:
            continue
        
        mean_smooth = pd.Series(mean).rolling(smooth_window, min_periods=1).mean().values
        std_smooth = pd.Series(std).rolling(smooth_window, min_periods=1).mean().values
        
        ax.plot(steps[valid], mean_smooth[valid], color=color, label=label)
        ax.fill_between(steps[valid], mean_smooth[valid] - std_smooth[valid],
                       mean_smooth[valid] + std_smooth[valid], alpha=0.2, color=color)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Epistemic Uncertainty')
    ax.set_title(title or 'Model Epistemic Uncertainty Over Training')
    ax.legend()
    
    return fig


def plot_autoregressive_error(
    data: dict[str, pd.DataFrame],
    smooth_window: int = 5,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot autoregressive prediction error over training.
    
    Args:
        data: Dict mapping condition name to DataFrame
        smooth_window: Rolling window for smoothing (default smaller for this metric)
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
    
    for condition, df in data.items():
        col = 'System_Dynamics_autoregressive_error_mean'
        if col not in df.columns:
            continue
        
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        steps = df['step'].values
        mean = df[col].values
        std = df.get('System_Dynamics_autoregressive_error_std', pd.Series([0] * len(df))).values
        
        valid = ~np.isnan(mean) & ~np.isinf(mean)
        if valid.sum() < smooth_window:
            continue
        
        if smooth_window > 1:
            mean_smooth = pd.Series(mean).rolling(smooth_window, min_periods=1).mean().values
            std_smooth = pd.Series(std).rolling(smooth_window, min_periods=1).mean().values
        else:
            mean_smooth = mean
            std_smooth = std
        
        ax.plot(steps[valid], mean_smooth[valid], color=color, label=label)
        ax.fill_between(steps[valid], mean_smooth[valid] - std_smooth[valid],
                       mean_smooth[valid] + std_smooth[valid], alpha=0.2, color=color)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Autoregressive Error (relative)')
    ax.set_title(title or 'World Model Prediction Error Over Training')
    ax.legend()
    
    return fig


def create_paper_figures(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    format: str = 'pdf',
    smooth_window: int = 20,
    autoregressive_smooth_window: int = 5,
):
    """
    Create all training curve figures for the paper.
    
    Args:
        data: Dict mapping condition name to DataFrame
        output_dir: Output directory for figures
        format: Output format ('pdf', 'png', 'svg')
        smooth_window: Smoothing window for reward/uncertainty curves
        autoregressive_smooth_window: Smoothing window for autoregressive error
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Mean Reward
    print("\nCreating mean reward plot...")
    fig = plot_mean_reward(data, smooth_window=smooth_window)
    output_path = output_dir / f'mean_reward.{format}'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    
    # 2. Epistemic Uncertainty
    print("\nCreating epistemic uncertainty plot...")
    fig = plot_epistemic_uncertainty(data, smooth_window=smooth_window)
    output_path = output_dir / f'epistemic_uncertainty.{format}'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    
    # 3. Autoregressive Error
    print("\nCreating autoregressive error plot...")
    fig = plot_autoregressive_error(data, smooth_window=autoregressive_smooth_window)
    output_path = output_dir / f'autoregressive_error.{format}'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")
    
    # 4. Combined figure (all three in one)
    print("\nCreating combined training curves figure...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    plot_mean_reward(data, smooth_window=smooth_window, ax=axes[0])
    plot_epistemic_uncertainty(data, smooth_window=smooth_window, ax=axes[1])
    plot_autoregressive_error(data, smooth_window=autoregressive_smooth_window, ax=axes[2])
    
    # Only keep legend on first plot to avoid clutter
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    
    fig.tight_layout()
    output_path = output_dir / f'training_curves_combined.{format}'
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot training curves from aggregated TensorBoard data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing aggregated TensorBoard CSVs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as data_dir)')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for figures')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated list of conditions to include (default: all)')
    parser.add_argument('--smooth_window', type=int, default=20,
                        help='Smoothing window for reward/uncertainty curves (default: 20)')
    parser.add_argument('--autoregressive_smooth_window', type=int, default=5,
                        help='Smoothing window for autoregressive error (default: 5)')
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
    
    print(f"Loading data from: {args.data_dir}")
    data = load_aggregated_data(args.data_dir)
    
    if not data:
        print("No aggregated data found!")
        print(f"Expected CSV files in: {args.data_dir}")
        print("Run extract_tensorboard_data.py first to generate aggregated CSVs")
        return
    
    print(f"Loaded {len(data)} conditions: {list(data.keys())}")
    
    # Filter conditions if specified
    if args.conditions:
        include_conditions = [c.strip() for c in args.conditions.split(',')]
        filtered_data = {}
        for cond in data:
            if condition_matches_any(cond, include_conditions):
                filtered_data[cond] = data[cond]
        
        if not filtered_data:
            print(f"WARNING: No conditions matched filter '{args.conditions}'")
            print(f"Available conditions: {list(data.keys())}")
        else:
            print(f"Filtered to {len(filtered_data)} conditions: {list(filtered_data.keys())}")
            data = filtered_data
    
    if args.show:
        # Interactive mode - show one plot
        fig = plot_mean_reward(data, smooth_window=args.smooth_window)
        plt.show()
    else:
        # Save all figures
        create_paper_figures(
            data, 
            output_dir, 
            args.format,
            smooth_window=args.smooth_window,
            autoregressive_smooth_window=args.autoregressive_smooth_window,
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
