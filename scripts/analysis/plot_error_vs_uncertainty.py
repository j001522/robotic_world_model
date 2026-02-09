#!/usr/bin/env python3
"""
Plot Error vs Epistemic Uncertainty during Imagination Rollouts

Creates publication-quality plots showing the relationship between prediction error
and epistemic uncertainty over the rollout horizon. This helps analyze how model
uncertainty correlates with prediction accuracy.

Usage:
    python plot_error_vs_uncertainty.py \
        --data_dir results/paper/error_vs_uncertainty \
        --output_dir results/paper/figures
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# Import shared utilities
from plot_utils import ColorManager, get_label, setup_publication_style, condition_matches_any

# Setup publication style
setup_publication_style()

# Create a module-level color manager for consistent colors across all plots
_color_manager = ColorManager()


def load_error_uncertainty_data(data_dir: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Load error vs uncertainty data from CSV files and aggregate by condition.
    
    Returns:
        - Combined DataFrame with all data
        - Dict mapping condition -> aggregated DataFrame (mean/std across seeds)
    """
    data_path = Path(data_dir)
    
    # Try to load the combined file first (contains all seeds)
    all_data_path = data_path / "error_vs_uncertainty_all.csv"
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
                    'prediction_step': group['prediction_step'].iloc[0] if 'prediction_step' in group.columns else horizon_t,
                    'error_rel_mean': group['error_rel_mean'].mean(),
                    'error_rel_std': group['error_rel_mean'].std() if len(group) > 1 else 0,
                    'error_abs_mean': group['error_abs_mean'].mean(),
                    'error_abs_std': group['error_abs_mean'].std() if len(group) > 1 else 0,
                    'uncertainty_mean': group['uncertainty_mean'].mean(),
                    'uncertainty_std': group['uncertainty_mean'].std() if len(group) > 1 else 0,
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
    for csv_file in data_path.glob("error_vs_uncertainty_*.csv"):
        if csv_file.name == "error_vs_uncertainty_all.csv":
            continue
        
        # Extract condition name from filename
        condition = csv_file.stem.replace("error_vs_uncertainty_", "")
        df = pd.read_csv(csv_file)
        condition_data[condition] = df
        print(f"  Loaded: {condition} ({len(df)} rows)")
    
    return all_df, condition_data


def plot_error_vs_uncertainty_curves(
    condition_data: dict[str, pd.DataFrame],
    error_type: str = 'rel',
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create a dual-axis plot showing both prediction error and uncertainty over horizon.
    
    Args:
        condition_data: Dict mapping condition name to DataFrame
        error_type: 'rel' for relative error, 'abs' for absolute error
        ax: Matplotlib axes to plot on (if None, creates new figure)
        figsize: Figure size if creating new figure
        title: Plot title
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Create twin axis for uncertainty
    ax_uncertainty = ax.twinx()
    
    error_mean_col = f'error_{error_type}_mean'
    error_std_col = f'error_{error_type}_std'
    
    for condition, df in condition_data.items():
        if df.empty:
            print(f"  Warning: No data for {condition}")
            continue
        
        # Check if columns exist
        if error_mean_col not in df.columns or 'uncertainty_mean' not in df.columns:
            print(f"  Warning: Required columns not found in {condition}")
            continue
        
        # Get data
        horizon = df['horizon_t'].values
        mean_error = df[error_mean_col].values
        std_error = df.get(error_std_col, pd.Series([0] * len(df))).values
        mean_uncertainty = df['uncertainty_mean'].values
        std_uncertainty = df.get('uncertainty_std', pd.Series([0] * len(df))).values
        
        # Get color and label
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        # Plot error on left axis
        ax.plot(horizon, mean_error, '-', color=color, label=f"{label} (Error)", 
                linewidth=2, alpha=0.8)
        ax.fill_between(horizon, mean_error - std_error, mean_error + std_error, 
                       alpha=0.15, color=color)
        
        # Plot uncertainty on right axis (dashed line)
        ax_uncertainty.plot(horizon, mean_uncertainty, '--', color=color, 
                           label=f"{label} (Uncertainty)", linewidth=2, alpha=0.8)
        ax_uncertainty.fill_between(horizon, mean_uncertainty - std_uncertainty, 
                                   mean_uncertainty + std_uncertainty, 
                                   alpha=0.1, color=color)
    
    # Labels
    ax.set_xlabel('Prediction Step')
    ylabel = 'Relative Prediction Error' if error_type == 'rel' else 'Absolute Prediction Error (MSE)'
    ax.set_ylabel(ylabel, color='black')
    ax_uncertainty.set_ylabel('Epistemic Uncertainty', color='gray')
    
    # Color the y-axis labels
    ax.tick_params(axis='y', labelcolor='black')
    ax_uncertainty.tick_params(axis='y', labelcolor='gray')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Prediction Error vs Epistemic Uncertainty Over Rollout')
    
    # Set x-axis to start from the minimum horizon_t value
    if condition_data:
        first_df = next(iter(condition_data.values()))
        if not first_df.empty and 'horizon_t' in first_df.columns:
            min_horizon = first_df['horizon_t'].min()
            ax.set_xlim(left=min_horizon - 1)
    
    ax.set_ylim(bottom=0)
    ax_uncertainty.set_ylim(bottom=0)
    
    # Combine legends from both axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_uncertainty.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    
    return fig


def plot_error_vs_uncertainty_scatter(
    all_data: pd.DataFrame,
    condition: Optional[str] = None,
    error_type: str = 'rel',
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Create a scatter plot showing correlation between error and uncertainty.
    
    Args:
        all_data: DataFrame with all data (not aggregated)
        condition: Specific condition to plot (if None, plots all)
        error_type: 'rel' for relative error, 'abs' for absolute error
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    error_col = f'error_{error_type}_mean'
    
    if condition:
        df = all_data[all_data['condition'] == condition]
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        ax.scatter(df[error_col], df['uncertainty_mean'], alpha=0.5, 
                  color=color, s=20, label=label)
        
        # Compute correlation
        corr, p_value = stats.pearsonr(df[error_col], df['uncertainty_mean'])
        
        # Add trend line
        z = np.polyfit(df[error_col], df['uncertainty_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[error_col].min(), df[error_col].max(), 100)
        ax.plot(x_line, p(x_line), '--', color=color, alpha=0.8, 
               label=f'Trend (r={corr:.3f}, p={p_value:.3e})')
        
        ax.set_title(f'Error vs Uncertainty Correlation - {label}')
    else:
        # Plot all conditions
        for cond in all_data['condition'].unique():
            df = all_data[all_data['condition'] == cond]
            color = _color_manager.get_color(cond)
            label = get_label(cond)
            
            ax.scatter(df[error_col], df['uncertainty_mean'], alpha=0.4, 
                      color=color, s=15, label=label)
        
        ax.set_title('Error vs Uncertainty Correlation - All Conditions')
    
    ylabel = 'Relative Prediction Error' if error_type == 'rel' else 'Absolute Prediction Error (MSE)'
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Epistemic Uncertainty')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    
    return fig


def plot_uncertainty_calibration(
    all_data: pd.DataFrame,
    error_type: str = 'rel',
    n_bins: int = 10,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Create a calibration plot showing how well uncertainty predicts error.
    
    Bins data by uncertainty level and shows average error in each bin.
    
    Args:
        all_data: DataFrame with all data
        error_type: 'rel' for relative error, 'abs' for absolute error
        n_bins: Number of uncertainty bins
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    error_col = f'error_{error_type}_mean'
    
    for condition in all_data['condition'].unique():
        df = all_data[all_data['condition'] == condition]
        color = _color_manager.get_color(condition)
        label = get_label(condition)
        
        # Bin by uncertainty
        df['uncertainty_bin'] = pd.qcut(df['uncertainty_mean'], q=n_bins, duplicates='drop')
        
        # Compute mean error per bin
        binned = df.groupby('uncertainty_bin').agg({
            error_col: ['mean', 'std'],
            'uncertainty_mean': 'mean'
        }).reset_index()
        
        # Flatten column names
        binned.columns = ['uncertainty_bin', 'error_mean', 'error_std', 'uncertainty_mid']
        
        # Plot
        ax.plot(binned['uncertainty_mid'], binned['error_mean'], 'o-', 
               color=color, label=label, markersize=6, linewidth=1.5)
        ax.fill_between(binned['uncertainty_mid'], 
                       binned['error_mean'] - binned['error_std'],
                       binned['error_mean'] + binned['error_std'],
                       alpha=0.2, color=color)
    
    ylabel = 'Relative Prediction Error' if error_type == 'rel' else 'Absolute Prediction Error (MSE)'
    ax.set_xlabel('Epistemic Uncertainty (binned)')
    ax.set_ylabel(ylabel)
    ax.set_title('Uncertainty Calibration: Error vs Uncertainty Level')
    ax.legend(loc='best', fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    
    return fig


def create_paper_figures(
    all_data: pd.DataFrame,
    condition_data: dict[str, pd.DataFrame],
    output_dir: Path,
    format: str = 'pdf',
):
    """
    Create all publication-ready figures.
    """
    # Extract uncertainty metric from data if available
    uncertainty_metric = 'std'  # default
    if 'uncertainty_metric' in all_data.columns:
        uncertainty_metric = all_data['uncertainty_metric'].iloc[0]
    print(f"\nUsing uncertainty metric: {uncertainty_metric}")
    
    # Main figure: Error and Uncertainty over horizon (dual-axis)
    print("\nCreating error vs uncertainty curves plot...")
    fig = plot_error_vs_uncertainty_curves(
        condition_data,
        error_type='rel',
        title=f'Prediction Error and Epistemic Uncertainty ({uncertainty_metric}) Over Rollout Horizon'
    )
    output_path = output_dir / f'error_vs_uncertainty_curves.{format}'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # Scatter plot for each condition
    print("\nCreating correlation scatter plots...")
    for condition in condition_data.keys():
        fig = plot_error_vs_uncertainty_scatter(
            all_data, 
            condition=condition,
            error_type='rel'
        )
        output_path = output_dir / f'error_vs_uncertainty_scatter_{condition}.{format}'
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close(fig)
    
    # Combined scatter plot
    fig = plot_error_vs_uncertainty_scatter(all_data, error_type='rel')
    output_path = output_dir / f'error_vs_uncertainty_scatter_all.{format}'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)
    
    # Uncertainty calibration plot
    print("\nCreating uncertainty calibration plot...")
    fig = plot_uncertainty_calibration(all_data, error_type='rel')
    output_path = output_dir / f'uncertainty_calibration.{format}'
    fig.savefig(output_path, bbox_inches='tight')
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
        csv_path = output_dir / 'error_vs_uncertainty_summary.csv'
        combined_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot error vs epistemic uncertainty during imagination rollouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing error vs uncertainty CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for figures (default: same as data_dir)')
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'svg'],
                        help='Output format for figures')
    parser.add_argument('--error_type', type=str, default='rel',
                        choices=['rel', 'abs'],
                        help='Error type to plot (rel=relative, abs=absolute)')
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
    all_df, condition_data = load_error_uncertainty_data(args.data_dir)
    
    if not condition_data:
        print("No error vs uncertainty data found!")
        print("Expected files: error_vs_uncertainty_<condition>.csv")
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
            all_df = all_df[all_df['condition'].isin(filtered_data.keys())]
    
    if args.show:
        # Interactive mode
        fig = plot_error_vs_uncertainty_curves(
            condition_data,
            error_type=args.error_type,
        )
        plt.show()
    else:
        # Save figures
        create_paper_figures(all_df, condition_data, output_dir, args.format)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
