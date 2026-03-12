#!/usr/bin/env python3
"""
Aggregate Hallucination Horizon Statistics

This script aggregates hallucination horizon metrics across episodes,
computing distribution statistics (mean, median, std) and the fraction
of rollouts with short hallucination horizons.

Input:
    hallucination_horizon_<condition>.csv
    Columns: horizon_t, error_rel_mean, error_abs_mean, etc.
    (Per-step error data)

Output:
    hallucination_distribution_<condition>.csv
    New columns:
        - hallucination_horizon_mean: Mean steps to divergence
        - hallucination_horizon_median: Median steps to divergence
        - hallucination_horizon_std: Std dev of steps to divergence
        - fraction_short: % rollouts with H_halluc < threshold

Usage:
    python aggregate_hallucination.py \
        --input_dir results/paper/hallucination_horizon \
        --output_dir results/paper/hallucination_stats \
        --error_threshold 1.0 \
        --short_horizon_threshold 30
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def compute_hallucination_horizon(
    df: pd.DataFrame,
    error_threshold: Optional[float] = None,
    threshold_percentile: float = 95.0,
    baseline_horizon: int = 5,
    history_horizon: int = 1,
    velocity_only: bool = True,
    baseline_condition: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute hallucination horizon per episode (rollout).
    
    Hallucination horizon: First prediction step where error exceeds threshold.
    
    Args:
        df: DataFrame with per-step error data
        error_threshold: Fixed threshold for "high error" (deprecated, use threshold_percentile)
        threshold_percentile: Percentile of baseline error to use as threshold (default: 95)
        baseline_horizon: Number of initial steps to use for baseline distribution (default: 5)
        history_horizon: Number of initial steps (prediction starts at history_horizon+1)
        velocity_only: If True, use velocity-only error columns
        baseline_condition: Condition to use for computing shared threshold (e.g., "bs-nopen")
                       If None, compute threshold per condition (not comparable)
    
    Returns:
        DataFrame with hallucination horizon per episode
    """
    # Compute shared threshold from baseline condition if specified
    shared_threshold = None
    shared_threshold_method = None
    if baseline_condition is not None and error_threshold is None:
        baseline_df = df[df['condition'] == baseline_condition]
        if len(baseline_df) > 0:
            if velocity_only:
                error_col = f'error_vel_rel_mean' if f'error_vel_rel_mean' in baseline_df.columns else 'error_rel_mean'
            else:
                error_col = 'error_rel_mean'
            
            if error_col in baseline_df.columns:
                baseline_errors = np.array(baseline_df[error_col].dropna().tolist())
                baseline_steps_col = 'horizon_t' if 'horizon_t' in baseline_df.columns else 'prediction_step'
                baseline_steps = np.array(baseline_df[baseline_steps_col].dropna().tolist())
                
                baseline_mask = baseline_steps <= baseline_horizon
                if baseline_mask.sum() > 0:
                    shared_threshold = np.percentile(baseline_errors[baseline_mask], threshold_percentile)
                    shared_threshold_method = f'shared_percentile_{threshold_percentile}'
                    print(f"  Shared threshold from {baseline_condition}: {shared_threshold:.6f}")
                else:
                    print(f"  WARNING: Insufficient baseline data for {baseline_condition}")
            else:
                print(f"  WARNING: Error column {error_col} not found")
        else:
            print(f"  WARNING: No data found for baseline condition {baseline_condition}")
    
    results = []
    
    # Group by unique runs (episodes)
    group_cols = ['condition', 'seed', 'checkpoint_step']

    for _, group_df in df.groupby(group_cols, dropna=False):
        # Sort by horizon_t (or prediction_step)
        if 'horizon_t' in group_df.columns:
            sort_col = 'horizon_t'
        elif 'prediction_step' in group_df.columns:
            sort_col = 'prediction_step'
        else:
            print(f"  WARNING: No horizon column found")
            results.append({col: group_df[col].iloc[0] if len(group_df) > 0 else np.nan
                           for col in group_cols})
            continue

        group_df = group_df.sort_values(sort_col).reset_index(drop=True)

        # Select error column based on velocity_only flag
        if velocity_only:
            error_col = f'error_vel_rel_mean' if f'error_vel_rel_mean' in group_df.columns else 'error_rel_mean'
        else:
            error_col = 'error_rel_mean'

        if error_col not in group_df.columns:
            continue

        errors = group_df[error_col].values

        # Remove NaN
        valid_mask = ~np.isnan(errors)
        if valid_mask.sum() < 2:
            results.append({col: group_df[col].iloc[0] for col in group_cols})
            continue

        errors = errors[valid_mask]
        steps = group_df[sort_col].values[valid_mask]

        # Determine threshold
        if shared_threshold is not None:
            # Use shared threshold from baseline condition
            threshold = shared_threshold
            threshold_method = shared_threshold_method
        elif error_threshold is not None:
            # Use fixed threshold (legacy)
            threshold = error_threshold
            threshold_method = 'fixed'
        else:
            # Use percentile-based threshold from short-horizon baseline (per condition)
            # Get errors from initial baseline_horizon steps
            baseline_mask = steps <= baseline_horizon
            if baseline_mask.sum() > 0:
                baseline_errors = errors[baseline_mask]
                threshold = np.percentile(baseline_errors, threshold_percentile)
                threshold_method = f'percentile_{threshold_percentile}'
            else:
                # Fallback to overall percentile
                threshold = np.percentile(errors, threshold_percentile)
                threshold_method = f'percentile_{threshold_percentile}_overall'

        # Find first step where error exceeds threshold
        error_exceeds = errors > threshold

        if error_exceeds.sum() == 0:
            # Never exceeds threshold - use last step
            h_halluc = len(errors)
        else:
            # First exceedance
            h_halluc = steps[error_exceeds][0]

        results.append({
            **{col: group_df[col].iloc[0] for col in group_cols},
            'hallucination_horizon': h_halluc,
            'threshold': threshold,
            'threshold_method': threshold_method,
            'max_error': errors.max(),
            'error_at_hallucination': errors[int(h_halluc - history_horizon)] if h_halluc <= len(errors) else np.nan,
            'n_steps': len(errors),
        })

    return pd.DataFrame(results)


def aggregate_hallucination_distribution(
    df: pd.DataFrame,
    short_threshold: float = 30.0,
) -> pd.DataFrame:
    """
    Aggregate hallucination horizon distribution statistics.

    Args:
        df: DataFrame with hallucination horizon per episode
        short_threshold: Threshold for "short hallucination"

    Returns:
        DataFrame with aggregated statistics per condition
    """
    aggregated = []

    if 'condition' not in df.columns:
        return pd.DataFrame()

    for condition in df['condition'].unique():
        cond_df = df[df['condition'] == condition]

        # Filter valid hallucination horizons
        valid_mask = cond_df['hallucination_horizon'].notna()

        if valid_mask.sum() == 0:
            continue

        cond_df_valid = cond_df[valid_mask]
        h_values = cond_df_valid['hallucination_horizon'].values

        # Distribution statistics
        h_mean = h_values.mean()
        h_median = np.median(h_values)
        h_std = h_values.std()
        h_min = h_values.min()
        h_max = h_values.max()

        # Fraction with short hallucination
        fraction_short = (h_values < short_threshold).sum() / len(h_values)

        aggregated.append({
            'condition': condition,
            'hallucination_horizon_mean': h_mean,
            'hallucination_horizon_median': h_median,
            'hallucination_horizon_std': h_std,
            'hallucination_horizon_min': h_min,
            'hallucination_horizon_max': h_max,
            'fraction_short_hallucination': fraction_short,
            'n_episodes': len(h_values),
        })

    return pd.DataFrame(aggregated)


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate hallucination horizon statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing hallucination_horizon CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for aggregated statistics')
    parser.add_argument('--error_threshold', type=float, default=None,
                        help='Fixed error threshold for hallucination detection (deprecated)')
    parser.add_argument('--threshold_percentile', type=float, default=95.0,
                        help='Percentile of baseline error to use as threshold (default: 95)')
    parser.add_argument('--baseline_horizon', type=int, default=5,
                        help='Number of initial steps for baseline distribution (default: 5)')
    parser.add_argument('--short_horizon_threshold', type=float, default=30.0,
                        help='Threshold for "short hallucination" (default: 30.0)')
    parser.add_argument('--history_horizon', type=int, default=1,
                        help='History horizon (prediction starts at h+1) (default: 1)')
    parser.add_argument('--velocity_only', action='store_true', default=True,
                        help='Use velocity-only error (recommended for Anymal)')
    parser.add_argument('--full_state', action='store_true',
                        help='Use full-state error instead of velocity-only')
    parser.add_argument('--baseline_condition', type=str, default=None,
                        help='Condition to use for computing shared threshold (e.g., "bs-nopen"). '
                             'If None, compute threshold per condition (not comparable)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine whether to use velocity_only
    use_velocity_only = args.velocity_only and not args.full_state
    
    # Determine threshold method
    if args.baseline_condition is not None:
        print(f"Aggregating hallucination from: {input_dir}")
        print(f"Using SHARED threshold from baseline condition: {args.baseline_condition}")
        print(f"  Threshold: {args.threshold_percentile}% of {args.baseline_horizon}-step baseline")
    elif args.error_threshold is not None:
        print(f"Aggregating hallucination from: {input_dir}")
        print(f"Using fixed error threshold: {args.error_threshold} (DEPRECATED)")
    else:
        print(f"Aggregating hallucination from: {input_dir}")
        print(f"Using PER-CONDITION threshold (not comparable across conditions)")
        print(f"  Threshold: {args.threshold_percentile}% of {args.baseline_horizon}-step baseline per condition")
    
    print(f"Short horizon threshold: {args.short_horizon_threshold}")
    print(f"Velocity-only mode: {use_velocity_only}")

    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    all_per_episode = []
    all_aggregated = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            # Compute hallucination horizon per episode
            per_episode_df = compute_hallucination_horizon(
                df,
                error_threshold=args.error_threshold,
                threshold_percentile=args.threshold_percentile,
                baseline_horizon=args.baseline_horizon,
                history_horizon=args.history_horizon,
                velocity_only=use_velocity_only,
                baseline_condition=args.baseline_condition,
            )
            per_episode_df.to_csv(output_dir / csv_file.name, index=False)
            all_per_episode.append(per_episode_df)

            # Aggregate distribution
            aggregated_df = aggregate_hallucination_distribution(per_episode_df, args.short_horizon_threshold)

            # Print summary
            if len(aggregated_df) > 0:
                row = aggregated_df.iloc[0]
                print(f"  Mean hallucination horizon: {row['hallucination_horizon_mean']:.1f} ± {row['hallucination_horizon_std']:.1f}")
                print(f"  Median: {row['hallucination_horizon_median']:.1f}")
                print(f"  Fraction short (<{args.short_horizon_threshold}): {row['fraction_short_hallucination']:.1%}")

            all_aggregated.append(aggregated_df)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save combined per-episode data
    if all_per_episode:
        combined_per_episode = pd.concat(all_per_episode, ignore_index=True)
        combined_path = output_dir / "hallucination_per_episode_all.csv"
        combined_per_episode.to_csv(combined_path, index=False)
        print(f"\nSaved per-episode data: {combined_path}")

    # Save combined aggregated data
    if all_aggregated:
        combined_aggregated = pd.concat(all_aggregated, ignore_index=True)
        combined_path = output_dir / "hallucination_distribution_all.csv"
        combined_aggregated.to_csv(combined_path, index=False)
        print(f"Saved aggregated statistics: {combined_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
