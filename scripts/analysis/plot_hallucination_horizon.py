#!/usr/bin/env python3
"""
Plot A: Hallucination Horizon Visualization

Creates publication-quality plots showing per-step prediction error over rollout horizon.
This visualizes how world model predictions diverge from reality over time.

Features:
- Plots error vs prediction horizon for each condition
- Supports filtering to velocity-related state dimensions (v, w, q_dot)
- Aggregates across seeds with mean/std bands

Usage:
    # Basic plot from hallucination horizon CSVs
    python plot_hallucination_horizon.py \
        --data_dir results/paper/hallucination_horizon \
        --output_dir results/paper/figures
    
    # Specify output format
    python plot_hallucination_horizon.py \
        --data_dir results/paper/hallucination_horizon \
        --output_dir results/paper/figures \
        --format pdf
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import shared utilities
from plot_utils import ColorManager, get_label, setup_publication_style, condition_matches_any

# Setup publication style
setup_publication_style()

# Create a module-level color manager for consistent colors across all plots
_color_manager = ColorManager()


def load_hallucination_data(data_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load hallucination horizon data from CSV files and aggregate by condition.
    
    Returns:
        - Combined DataFrame with all data
        - Dict mapping condition -> aggregated DataFrame (mean/std across seeds)
    """
    data_path = Path(data_dir)
    
    # Try to load the combined file first (contains all seeds)
    all_data_path = data_path / "hallucination_horizon_all.csv"
    if all_data_path.exists():
        all_df = pd.read_csv(all_data_path)
        print(f"  Loaded combined data: {len(all_df)} rows")
        
        # Aggregate by condition - compute mean/std across seeds
        condition_data = {}
        for condition in all_df['condition'].unique():
            cond_df = all_df[all_df['condition'] == condition]
            n_seeds = cond_df['seed'].nunique()
            
            # Group by horizon_t and aggregate across seeds
            agg_records = []
            for horizon_t, group in cond_df.groupby('horizon_t'):
                agg_records.append({
                    'horizon_t': horizon_t,
                    'error_rel_mean': group['error_rel_mean'].mean(),
                    'error_rel_std': group['error_rel_mean'].std() if len(group) > 1 else 0,
                    'error_abs_mean': group['error_abs_mean'].mean(),
                    'error_abs_std': group['error_abs_mean'].std() if len(group) > 1 else 0,
                    'n_seeds': len(group),
                })
            
            agg_df = pd.DataFrame(agg_records)
            condition_data[condition] = agg_df
            print(f"  Aggregated: {condition} ({n_seeds} seeds, {len(agg_df)} horizon steps)")
        
        return all_df, condition_data
    
    # Fallback: Load per-condition aggregated files
    print("  No combined file found, loading per-condition files...")
    all_df = None
    condition_data = {}
    for csv_file in data_path.glob("hallucination_horizon_*.csv"):
        if csv_file.name == "hallucination_horizon_all.csv":
            continue
        
        # Extract condition name from filename
        condition = csv_file.stem.replace("hallucination_horizon_", "")
        df = pd.read_csv(csv_file)
        condition_data[condition] = df
        print(f"  Loaded: {condition} ({len(df)} rows)")
    
    return all_df, condition_data


def plot_hallucination_horizon(
    condition_data: dict[str, pd.DataFrame],
    error_type: str = 'rel',  # 'rel' or 'abs'
    checkpoint_step: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create hallucination horizon plot showing error vs prediction step.
    
    Args:
        condition_data: Dict mapping condition name to DataFrame
        error_type: 'rel' for relative error, 'abs' for absolute error
        checkpoint_step: Specific checkpoint to plot (None = use all/latest)
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
    
    mean_col = f'error_{error_type}_mean'
    std_col = f'error_{error_type}_std'
    
    for condition, df in condition_data.items():
        # Filter by checkpoint step if specified
        if checkpoint_step is not None and 'checkpoint_step' in df.columns:
            df = df[df['checkpoint_step'] == checkpoint_step]
        elif 'checkpoint_step' in df.columns:
            # Use the latest checkpoint
            checkpoint_step = df['checkpoint_step'].max()
            df = df[df['checkpoint_step'] == checkpoint_step]
        
        if df.empty:
            print(f"  Warning: No data for {condition}")
            continue
        
        # Check if columns exist
        if mean_col not in df.columns:
            print(f"  Warning: {mean_col} not found in {condition}")
            continue
        
        # Get data
        horizon = df['horizon_t'].values
        mean_error = df[mean_col].values
        std_error = df.get(std_col, pd.Series([0] * len(df))).values
        
        # Get color and label
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        # Plot
        ax.plot(horizon, mean_error, 'o-', color=color, label=label, markersize=4, linewidth=1.5)
        ax.fill_between(horizon, mean_error - std_error, mean_error + std_error, 
                       alpha=0.2, color=color)
    
    # Labels
    ax.set_xlabel('Prediction Step')
    ylabel = 'Relative Prediction Error' if error_type == 'rel' else 'Absolute Prediction Error (MSE)'
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    else:
        step_str = f" (checkpoint {checkpoint_step})" if checkpoint_step else ""
        ax.set_title(f'Hallucination Horizon{step_str}')
    
    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)
    
    # Set x-axis to start from the minimum horizon_t value (accounting for history_horizon offset)
    if condition_data:
        first_df = next(iter(condition_data.values()))
        if not first_df.empty and 'horizon_t' in first_df.columns:
            min_horizon = first_df['horizon_t'].min()
            ax.set_xlim(left=min_horizon - 1)
    
    return fig


def plot_hallucination_horizon_comparison(
    condition_data: dict[str, pd.DataFrame],
    checkpoint_steps: Optional[list[int]] = None,
    error_type: str = 'rel',
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Create side-by-side comparison of hallucination horizons at different checkpoints.
    """
    if checkpoint_steps is None:
        # Get all available checkpoint steps
        all_steps = set()
        for df in condition_data.values():
            if 'checkpoint_step' in df.columns:
                all_steps.update(df['checkpoint_step'].unique())
        checkpoint_steps = sorted(all_steps)[-3:]  # Last 3 checkpoints
    
    n_plots = len(checkpoint_steps)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, sharey=True)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, step in enumerate(checkpoint_steps):
        plot_hallucination_horizon(
            condition_data, 
            error_type=error_type,
            checkpoint_step=step,
            ax=axes[i],
            title=f'Checkpoint {step}'
        )
        if i > 0:
            axes[i].set_ylabel('')
            axes[i].legend().remove()
    
    fig.tight_layout()
    return fig


def plot_hallucination_by_state_component(
    all_data: pd.DataFrame,
    condition: str,
    components: list[str] = ['v', 'w', 'q_dot'],
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot hallucination horizon for different state components.
    
    Requires that the evaluation was run with --save_components flag
    to produce per-component error columns.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to condition
    df = all_data[all_data['condition'] == condition]
    
    component_colors = {
        'v': '#0077BB',      # Linear velocity
        'w': '#EE7733',      # Angular velocity
        'q_dot': '#009988',  # Joint velocities
        'all': '#CC3311',    # All states
    }
    
    component_labels = {
        'v': 'Linear Velocity (v)',
        'w': 'Angular Velocity (ω)',
        'q_dot': 'Joint Velocities (q̇)',
        'all': 'All States',
    }
    
    for component in components:
        col_name = f'error_rel_{component}_mean'
        if col_name not in df.columns:
            print(f"  Warning: {col_name} not found")
            continue
        
        horizon = df['horizon_t'].values
        error = df[col_name].values
        
        color = component_colors.get(component, None)
        label = component_labels.get(component, component)
        
        ax.plot(horizon, error, 'o-', color=color, label=label, markersize=4)
    
    ax.set_xlabel('Prediction Horizon (steps)')
    ax.set_ylabel('Relative Prediction Error')
    ax.set_title(f'Hallucination Horizon by State Component ({get_label(condition)})')
    ax.legend()
    ax.set_ylim(bottom=0)
    
    return fig


def create_paper_figure(
    condition_data: dict[str, pd.DataFrame],
    output_dir: Path,
    format: str = 'pdf',
):
    """
    Create publication-ready hallucination horizon figure.
    """
    # Main figure: relative error vs horizon
    print("\nCreating main hallucination horizon plot...")
    fig = plot_hallucination_horizon(
        condition_data,
        error_type='rel',
        title='World Model Prediction Error Over Rollout Horizon'
    )
    output_path = output_dir / f'hallucination_horizon.{format}'
    fig.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # Additional: absolute error
    print("\nCreating absolute error plot...")
    fig = plot_hallucination_horizon(
        condition_data,
        error_type='abs',
        title='World Model Absolute Prediction Error Over Rollout Horizon'
    )
    output_path = output_dir / f'hallucination_horizon_abs.{format}'
    fig.savefig(output_path)
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # Save combined data as CSV for reference
    combined_records = []
    for condition, df in condition_data.items():
        for _, row in df.iterrows():
            record = row.to_dict()
            record['condition'] = condition
            combined_records.append(record)
    
    if combined_records:
        combined_df = pd.DataFrame(combined_records)
        csv_path = output_dir / 'hallucination_horizon_summary.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot hallucination horizon (per-step prediction error over rollout)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing hallucination horizon CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as data_dir)')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for figures')
    parser.add_argument('--error_type', type=str, default='rel',
                        choices=['rel', 'abs'],
                        help='Error type to plot (rel=relative, abs=absolute)')
    parser.add_argument('--checkpoint_step', type=int, default=None,
                        help='Specific checkpoint step to plot')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated list of conditions to include (default: all)')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively instead of saving')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.data_dir)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.data_dir}")
    all_df, condition_data = load_hallucination_data(args.data_dir)
    
    if not condition_data:
        print("No hallucination horizon data found!")
        print("Expected files: hallucination_horizon_<condition>.csv")
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
        fig = plot_hallucination_horizon(
            condition_data,
            error_type=args.error_type,
            checkpoint_step=args.checkpoint_step,
        )
        plt.show()
    else:
        # Save figures
        create_paper_figure(condition_data, output_dir, args.format)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
