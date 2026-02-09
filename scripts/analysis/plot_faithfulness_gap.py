#!/usr/bin/env python3
"""
Plot B: Faithfulness Gap Visualization

Creates publication-quality plots showing the gap between imagined rewards
(from world model rollouts during training) and real rewards (from actual 
policy execution in the environment).

Features:
- Plots imagined vs real reward over training steps
- Shows faithfulness gap (imagined - real) as a bar or line plot
- Aggregates across seeds with mean/std bands

Usage:
    # Basic plot from faithfulness gap CSVs
    python plot_faithfulness_gap.py \
        --data_dir results/paper/faithfulness_gap \
        --output_dir results/paper/figures
    
    # With TensorBoard data for imagination rewards
    python plot_faithfulness_gap.py \
        --data_dir results/paper/faithfulness_gap \
        --tensorboard_dir results/paper/tensorboard/aggregated \
        --output_dir results/paper/figures
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Import shared utilities
from plot_utils import ColorManager, get_label, setup_publication_style, COLORBLIND_PALETTE, preassign_colors, get_color, condition_matches_any

# Setup publication style
setup_publication_style()

# Create a module-level color manager for consistent colors across all plots
_color_manager = ColorManager()


def load_faithfulness_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load faithfulness gap data from CSV files.
    
    Returns:
        - Comparison DataFrame (with gap computed)
        - All individual results DataFrame
    """
    data_path = Path(data_dir)
    
    comparison_path = data_path / "faithfulness_comparison.csv"
    all_path = data_path / "real_rewards_all.csv"
    
    comparison_df = None
    all_df = None
    
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        print(f"  Loaded comparison data: {len(comparison_df)} rows")
    
    if all_path.exists():
        all_df = pd.read_csv(all_path)
        print(f"  Loaded all results: {len(all_df)} rows")
    
    return comparison_df, all_df


def plot_faithfulness_gap(
    comparison_df: pd.DataFrame,
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create faithfulness gap plot showing imagined vs real rewards.
    
    Args:
        comparison_df: DataFrame with columns: condition, step, 
                      real_reward_mean, imagination_reward_mean, faithfulness_gap
        figsize: Figure size
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
    
    # Left plot: Rewards over training
    ax1 = axes[0]
    conditions = comparison_df['condition'].unique()
    
    for condition in conditions:
        cond_df = comparison_df[comparison_df['condition'] == condition].sort_values('step')
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        steps = cond_df['step'].values
        real = cond_df['real_reward_mean'].values
        real_std = cond_df.get('real_reward_std', pd.Series([0] * len(cond_df))).values
        
        # Plot real rewards (solid)
        ax1.plot(steps, real, 'o-', color=color, label=f'{label} (real)', 
                markersize=6, linewidth=2)
        ax1.fill_between(steps, real - real_std, real + real_std, 
                        alpha=0.15, color=color)
        
        # Plot imagination rewards (dashed) if available
        if 'imagination_reward_mean' in cond_df.columns:
            imag = cond_df['imagination_reward_mean'].values
            valid = ~np.isnan(imag)
            if valid.sum() > 0:
                ax1.plot(steps[valid], imag[valid], '--', color=color, 
                        label=f'{label} (imagined)', linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Real vs Imagined Rewards')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Faithfulness gap
    ax2 = axes[1]
    
    for condition in conditions:
        cond_df = comparison_df[comparison_df['condition'] == condition].sort_values('step')
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        if 'faithfulness_gap' not in cond_df.columns:
            continue
        
        steps = cond_df['step'].values
        gap = cond_df['faithfulness_gap'].values
        valid = ~np.isnan(gap)
        
        if valid.sum() > 0:
            ax2.plot(steps[valid], gap[valid], 'o-', color=color, label=label,
                    markersize=6, linewidth=2)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Faithfulness Gap (imagined - real)')
    ax2.set_title('World Model Overestimation')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    fig.tight_layout()
    return fig


def plot_faithfulness_gap_bar(
    comparison_df: pd.DataFrame,
    step: Optional[int] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Create bar plot comparing final faithfulness gap across conditions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use last step if not specified
    if step is None:
        step = comparison_df['step'].max()
    
    # Get data at specified step
    step_df = comparison_df[comparison_df['step'] == step]
    
    conditions = []
    gaps = []
    colors_list = []
    
    for condition in step_df['condition'].unique():
        cond_row = step_df[step_df['condition'] == condition].iloc[0]
        if 'faithfulness_gap' in cond_row and pd.notna(cond_row['faithfulness_gap']):
            conditions.append(get_label(condition))
            gaps.append(cond_row['faithfulness_gap'])
            colors_list.append(_color_manager.get_color(condition))
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, gaps, color=colors_list, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.annotate(f'{gap:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -12),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=9)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Faithfulness Gap (imagined - real)')
    ax.set_title(f'World Model Overestimation at Step {step}')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.tight_layout()
    return fig


def plot_real_rewards_only(
    all_df: pd.DataFrame,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    Plot real rewards over training when imagination data is not available.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Aggregate by condition and step
    grouped = all_df.groupby(['condition', 'checkpoint_step'])
    
    agg_data = {}
    for (condition, step), group in grouped:
        if condition not in agg_data:
            agg_data[condition] = {'steps': [], 'means': [], 'stds': []}
        agg_data[condition]['steps'].append(step)
        agg_data[condition]['means'].append(group['mean_reward'].mean())
        agg_data[condition]['stds'].append(group['mean_reward'].std())
    
    for condition, data in agg_data.items():
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        # Sort by step
        order = np.argsort(data['steps'])
        steps = np.array(data['steps'])[order]
        means = np.array(data['means'])[order]
        stds = np.array(data['stds'])[order]
        
        ax.plot(steps, means, 'o-', color=color, label=label,
               markersize=6, linewidth=2)
        ax.fill_between(steps, means - stds, means + stds,
                       alpha=0.2, color=color)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Episode Reward (Real)')
    ax.set_title('Real Policy Evaluation Rewards')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def create_paper_figures(
    comparison_df: Optional[pd.DataFrame],
    all_df: Optional[pd.DataFrame],
    output_dir: Path,
    format: str = 'pdf',
):
    """
    Create all faithfulness gap figures for the paper.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if comparison_df is not None and len(comparison_df) > 0:
        # Full comparison plot
        print("\nCreating faithfulness gap comparison plot...")
        fig = plot_faithfulness_gap(comparison_df)
        output_path = output_dir / f'faithfulness_gap.{format}'
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")
        plt.close(fig)
        
        # Bar plot at final step
        print("\nCreating faithfulness gap bar plot...")
        fig = plot_faithfulness_gap_bar(comparison_df)
        output_path = output_dir / f'faithfulness_gap_bar.{format}'
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")
        plt.close(fig)
    
    if all_df is not None and len(all_df) > 0:
        # Real rewards only plot
        print("\nCreating real rewards plot...")
        fig = plot_real_rewards_only(all_df)
        output_path = output_dir / f'real_rewards.{format}'
        fig.savefig(output_path)
        print(f"  Saved: {output_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot faithfulness gap (imagined vs real rewards)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing faithfulness gap CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as data_dir)')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for figures')
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
    comparison_df, all_df = load_faithfulness_data(args.data_dir)
    
    if comparison_df is None and all_df is None:
        print("No faithfulness data found!")
        print("Expected files: faithfulness_comparison.csv or real_rewards_all.csv")
        return
    
    # Filter conditions if specified
    if args.conditions:
        include_conditions = [c.strip() for c in args.conditions.split(',')]
        
        if comparison_df is not None:
            mask = comparison_df['condition'].apply(
                lambda c: condition_matches_any(c, include_conditions)
            )
            filtered_comparison = comparison_df[mask]
            if len(filtered_comparison) == 0:
                print(f"WARNING: No conditions matched filter '{args.conditions}'")
                print(f"Available conditions: {comparison_df['condition'].unique().tolist()}")
            else:
                print(f"Filtered comparison_df to {len(filtered_comparison)} rows")
                comparison_df = filtered_comparison
        
        if all_df is not None:
            mask = all_df['condition'].apply(
                lambda c: condition_matches_any(c, include_conditions)
            )
            filtered_all = all_df[mask]
            if len(filtered_all) == 0:
                print(f"WARNING: No conditions matched filter in all_df")
            else:
                print(f"Filtered all_df to {len(filtered_all)} rows")
                all_df = filtered_all
    
    if args.show:
        # Interactive mode
        if comparison_df is not None:
            fig = plot_faithfulness_gap(comparison_df)
        elif all_df is not None:
            fig = plot_real_rewards_only(all_df)
        plt.show()
    else:
        # Save figures
        create_paper_figures(comparison_df, all_df, output_dir, args.format)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
