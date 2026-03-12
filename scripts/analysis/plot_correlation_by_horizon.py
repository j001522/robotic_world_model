#!/usr/bin/env python3
"""
Plot Correlation Between Prediction Error and Epistemic Uncertainty by Prediction Step

Creates publication-quality plots showing how the correlation between prediction error
and epistemic uncertainty changes across the prediction horizon. This helps analyze
whether the model's uncertainty becomes a better indicator of prediction accuracy as
we roll out further into the future.

Usage:
    python plot_correlation_by_horizon.py \
        --data_dir results/paper/correlation_by_horizon \
        --output_dir results/paper/figures
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import ColorManager, get_label, setup_publication_style, condition_matches_any, load_custom_colors

setup_publication_style()
_color_manager = ColorManager()


def load_correlation_data(data_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    data_path = Path(data_dir)
    
    all_data_path = data_path / "correlation_by_horizon_all.csv"
    if all_data_path.exists():
        all_df = pd.read_csv(all_data_path)
        print(f"  Loaded combined data: {len(all_df)} rows")
        
        condition_data = {}
        for condition in all_df['condition'].unique():
            cond_df = all_df[all_df['condition'] == condition]
            n_seeds = cond_df['seed'].nunique()
            
            agg_records = []
            for pred_step, group in cond_df.groupby('prediction_step'):
                agg_records.append({
                    'prediction_step': pred_step,
                    'pearson_r_mean': group['pearson_r'].mean(),
                    'pearson_r_std': group['pearson_r'].std() if len(group) > 1 else 0,
                    'spearman_r_mean': group['spearman_r'].mean(),
                    'spearman_r_std': group['spearman_r'].std() if len(group) > 1 else 0,
                    'n_seeds': len(group),
                })
            
            agg_df = pd.DataFrame(agg_records)
            condition_data[condition] = agg_df
            print(f"  Aggregated: {condition} ({n_seeds} seeds, {len(agg_df)} prediction steps)")
        
        return all_df, condition_data
    
    print("  No combined file found, loading per-condition files...")
    all_df = None
    condition_data = {}
    for csv_file in data_path.glob("correlation_by_horizon_*.csv"):
        if csv_file.name == "correlation_by_horizon_all.csv":
            continue
        
        condition = csv_file.stem.replace("correlation_by_horizon_", "")
        df = pd.read_csv(csv_file)
        condition_data[condition] = df
        print(f"  Loaded: {condition} ({len(df)} rows)")
    
    return all_df, condition_data


def plot_correlation_over_horizon(
    condition_data: dict[str, pd.DataFrame],
    correlation_type: str = 'pearson',
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
    show_ci: bool = True,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    correlation_col = f'{correlation_type}_r_mean'
    std_col = f'{correlation_type}_r_std'
    
    for condition, df in condition_data.items():
        if df.empty:
            continue
        
        if correlation_col not in df.columns:
            continue
        
        pred_steps = df['prediction_step'].values
        correlation = df[correlation_col].values
        std = df.get(std_col, pd.Series([0] * len(df))).values
        
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        ax.plot(pred_steps, correlation, '-', color=color, label=label,
                linewidth=2, alpha=0.8)
        
        if show_ci and len(df) > 1:
            ax.fill_between(pred_steps, correlation - std, correlation + std,
                          alpha=0.2, color=color)
    
    ax.set_xlabel('Prediction Step')
    correlation_label = 'Pearson' if correlation_type == 'pearson' else 'Spearman'
    ax.set_ylabel(f'{correlation_label} Correlation (r)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{correlation_label} Correlation: Error vs Uncertainty Over Prediction Horizon')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=-1, color='red', linestyle=':', linewidth=1, alpha=0.3)
    
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='best', fontsize=8)
    
    return fig


def plot_correlation_comparison(
    condition_data: dict[str, pd.DataFrame],
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    for condition, df in condition_data.items():
        if df.empty:
            continue
        
        if 'pearson_r_mean' not in df.columns or 'spearman_r_mean' not in df.columns:
            continue
        
        pred_steps = df['prediction_step'].values
        pearson_r = df['pearson_r_mean'].values
        spearman_r = df['spearman_r_mean'].values
        
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        ax.plot(pred_steps, pearson_r, '-', color=color, linewidth=2,
                alpha=0.8, label=f'{label} (Pearson)')
        ax.plot(pred_steps, spearman_r, '--', color=color, linewidth=2,
                alpha=0.8, label=f'{label} (Spearman)')
    
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Correlation (r)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Pearson vs Spearman Correlation Over Prediction Horizon')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.3)
    ax.axhline(y=-1, color='red', linestyle=':', linewidth=1, alpha=0.3)
    
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='best', fontsize=7)
    
    return fig


def plot_correlation_heatmap(
    all_data: pd.DataFrame,
    correlation_type: str = 'pearson',
    figsize: tuple = (10, 6),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    correlation_col = f'{correlation_type}_r'
    
    pivot_df = all_data.pivot_table(
        index=['condition', 'seed'],
        columns='prediction_step',
        values=correlation_col
    )
    
    heatmap_data = pivot_df.groupby('condition').mean()
    
    im = ax.imshow(heatmap_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([get_label(c) for c in heatmap_data.index])
    
    correlation_label = 'Pearson' if correlation_type == 'pearson' else 'Spearman'
    ax.set_title(f'{correlation_label} Correlation Heatmap: Error vs Uncertainty')
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Condition')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{correlation_label} Correlation (r)')
    
    return fig


def create_paper_figures(
    all_data: pd.DataFrame,
    condition_data: dict[str, pd.DataFrame],
    output_dir: Path,
    format: str = 'pdf',
):
    uncertainty_metric = 'std'
    if 'uncertainty_metric' in all_data.columns:
        uncertainty_metric = all_data['uncertainty_metric'].iloc[0]
    print(f"\nUsing uncertainty metric: {uncertainty_metric}")
    
    print("\nCreating Pearson correlation over horizon plot...")
    fig = plot_correlation_over_horizon(
        condition_data,
        correlation_type='pearson',
        title=f'Pearson Correlation: Error vs Uncertainty ({uncertainty_metric})'
    )
    output_path = output_dir / f'correlation_by_horizon_pearson.{format}'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    print("\nCreating Spearman correlation over horizon plot...")
    fig = plot_correlation_over_horizon(
        condition_data,
        correlation_type='spearman',
        title=f'Spearman Correlation: Error vs Uncertainty ({uncertainty_metric})'
    )
    output_path = output_dir / f'correlation_by_horizon_spearman.{format}'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    print("\nCreating correlation comparison plot...")
    fig = plot_correlation_comparison(
        condition_data,
        title=f'Pearson vs Spearman Correlation ({uncertainty_metric})'
    )
    output_path = output_dir / f'correlation_by_horizon_comparison.{format}'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    print("\nCreating correlation heatmap...")
    fig = plot_correlation_heatmap(all_data, correlation_type='pearson')
    output_path = output_dir / f'correlation_by_horizon_heatmap.{format}'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    combined_records = []
    for condition, df in condition_data.items():
        for _, row in df.iterrows():
            record = row.to_dict()
            record['condition'] = condition
            combined_records.append(record)
    
    if combined_records:
        combined_df = pd.DataFrame(combined_records)
        csv_path = output_dir / 'correlation_by_horizon_summary.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot correlation between prediction error and epistemic uncertainty by prediction step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing correlation by horizon CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as data_dir)')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for figures')
    parser.add_argument('--correlation_type', type=str, default='pearson',
                        choices=['pearson', 'spearman'],
                        help='Correlation type to plot (default: pearson)')
    parser.add_argument('--conditions', type=str, default=None,
                        help='Comma-separated list of conditions to include (default: all)')
    parser.add_argument('--exclude', type=str, default=None,
                        help='Comma-separated list of conditions to exclude')
    parser.add_argument('--colors', type=str, default=None,
                        help='Path to YAML file with custom color mappings')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively instead of saving')

    args = parser.parse_args()

    # Load custom colors if specified
    load_custom_colors(args.colors)
    
    if args.output_dir is None:
        output_dir = Path(args.data_dir)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {args.data_dir}")
    all_df, condition_data = load_correlation_data(args.data_dir)
    
    if not condition_data:
        print("No correlation data found!")
        print("Expected files: correlation_by_horizon_<condition>.csv")
        return
    
    print(f"\nLoaded {len(condition_data)} conditions")
    
    # Apply filtering
    if args.conditions or args.exclude:
        # Get available conditions
        available_conditions = list(condition_data.keys())
        
        # Start with all conditions
        filtered_conditions = available_conditions
        
        # Apply exclude filter
        if args.exclude:
            exclude_list = [c.strip() for c in args.exclude.split(',')]
            filtered_conditions = [c for c in filtered_conditions 
                              if not condition_matches_any(c, exclude_list)]
            print(f"Excluded {len(available_conditions) - len(filtered_conditions)} conditions")
        
        # Apply include filter
        if args.conditions:
            include_list = [c.strip() for c in args.conditions.split(',')]
            filtered_conditions = [c for c in filtered_conditions 
                                if condition_matches_any(c, include_list)]
        
        # Build filtered data
        filtered_data = {}
        for cond in filtered_conditions:
            filtered_data[cond] = condition_data[cond]
        
        if not filtered_data:
            print(f"WARNING: No conditions matched filters")
            print(f"Available conditions: {available_conditions}")
        else:
            print(f"Filtered to {len(filtered_data)} conditions: {list(filtered_data.keys())}")
            condition_data = filtered_data
            all_df = all_df[all_df['condition'].isin(filtered_data.keys())]
    
    if args.show:
        fig = plot_correlation_over_horizon(
            condition_data,
            correlation_type=args.correlation_type,
        )
        plt.show()
    else:
        create_paper_figures(all_df, condition_data, output_dir, args.format)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
