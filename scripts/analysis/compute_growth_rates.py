#!/usr/bin/env python3
"""
Compute Growth Rates for Error and Uncertainty

This script computes linear growth rates from error_vs_uncertainty data.
These growth rates quantify how quickly errors and uncertainties accumulate
during open-loop prediction rollouts.

Input:
    error_vs_uncertainty_<condition>.csv
    Columns: horizon_t, prediction_step, error_rel_mean, error_rel_std,
             error_abs_mean, error_abs_std, uncertainty_mean, uncertainty_std

Output:
    growth_rates_<condition>.csv
    New columns:
        - error_growth_rate: Linear slope of error vs prediction_step
        - uncertainty_growth_rate: Linear slope of uncertainty vs prediction_step
        - error_uncertainty_correlation: Correlation between growth rates

Usage:
    python compute_growth_rates.py \
        --input_dir results/paper/error_vs_uncertainty \
        --output_dir results/paper/growth_rates
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats


def compute_growth_rates_single(
    df: pd.DataFrame,
    horizon_range: Optional[tuple] = None,
    min_samples: int = 5,
) -> pd.DataFrame:
    """
    Compute growth rates for a single condition.

    Args:
        df: DataFrame with error and uncertainty per horizon step
        horizon_range: Tuple of (min_step, max_step) to fit regression
        min_samples: Minimum samples required for regression

    Returns:
        DataFrame with added growth rate columns
    """
    results = []

    # Group by unique runs (condition, seed, checkpoint_step)
    group_cols = ['condition', 'seed', 'checkpoint_step']

    for _, group_df in df.groupby(group_cols, dropna=False):
        # Sort by prediction_step
        group_df = group_df.sort_values('prediction_step').reset_index(drop=True)

        # Filter by horizon range if specified
        if horizon_range is not None:
            h_min, h_max = horizon_range
            mask = (group_df['prediction_step'] >= h_min) & (group_df['prediction_step'] <= h_max)
            group_df = group_df[mask].reset_index(drop=True)

        # Need minimum samples for regression
        if len(group_df) < min_samples:
            results.append({
                **{col: group_df[col].iloc[0] if len(group_df) > 0 else np.nan
                   for col in group_cols},
                'error_growth_rate': np.nan,
                'uncertainty_growth_rate': np.nan,
                'error_uncertainty_growth_correlation': np.nan,
                'n_samples': len(group_df),
            })
            continue

        # Extract data for regression
        x = group_df['prediction_step'].values
        y_error = group_df['error_rel_mean'].values
        y_uncertainty = group_df['uncertainty_mean'].values

        # Linear regression for error growth rate
        slope_error, intercept_error, r_value_error, p_value_error, std_err_error = stats.linregress(x, y_error)
        error_growth_rate = slope_error

        # Linear regression for uncertainty growth rate
        slope_unc, intercept_unc, r_value_unc, p_value_unc, std_err_unc = stats.linregress(x, y_uncertainty)
        uncertainty_growth_rate = slope_unc

        # Correlation between growth rates (if multiple runs available)
        error_uncertainty_growth_correlation = np.nan

        results.append({
            **{col: group_df[col].iloc[0] if len(group_df) > 0 else np.nan
               for col in group_cols},
            'error_growth_rate': error_growth_rate,
            'error_growth_rate_stderr': std_err_error,
            'error_growth_rate_p_value': p_value_error,
            'error_growth_rate_r_squared': r_value_error**2,
            'uncertainty_growth_rate': uncertainty_growth_rate,
            'uncertainty_growth_rate_stderr': std_err_unc,
            'uncertainty_growth_rate_p_value': p_value_unc,
            'uncertainty_growth_rate_r_squared': r_value_unc**2,
            'error_uncertainty_growth_correlation': error_uncertainty_growth_correlation,
            'n_samples': len(group_df),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compute growth rates from error vs uncertainty data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing error_vs_uncertainty CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for growth rates CSV files')
    parser.add_argument('--horizon_min', type=int, default=None,
                        help='Minimum horizon step for regression (inclusive)')
    parser.add_argument('--horizon_max', type=int, default=None,
                        help='Maximum horizon step for regression (inclusive)')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Minimum samples required for regression (default: 5)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build horizon range tuple
    horizon_range = None
    if args.horizon_min is not None and args.horizon_max is not None:
        horizon_range = (args.horizon_min, args.horizon_max)

    print(f"Computing growth rates from: {input_dir}")
    print(f"Output directory: {output_dir}")
    if horizon_range is not None:
        print(f"Horizon range: {horizon_range[0]} to {horizon_range[1]}")

    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    all_results = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            # Check required columns
            required_cols = ['prediction_step', 'error_rel_mean', 'uncertainty_mean']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  WARNING: Missing columns: {missing_cols}")
                continue

            # Compute growth rates
            results_df = compute_growth_rates_single(df, horizon_range, args.min_samples)
            results_df.to_csv(output_dir / csv_file.name, index=False)
            all_results.append(results_df)

            # Print summary for this file
            valid_growth = results_df['error_growth_rate'].notna()
            if valid_growth.sum() > 0:
                print(f"  Error growth rate: {results_df['error_growth_rate'].mean():.6f} ± {results_df['error_growth_rate'].std():.6f}")
                print(f"  Uncertainty growth rate: {results_df['uncertainty_growth_rate'].mean():.6f} ± {results_df['uncertainty_growth_rate'].std():.6f}")
            else:
                print(f"  WARNING: No valid growth rates computed")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save combined file
        combined_path = output_dir / "growth_rates_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined results: {combined_path}")

        # Aggregate by condition
        if 'condition' in combined_df.columns:
            print("\nAggregating by condition...")
            for condition in combined_df['condition'].unique():
                cond_df = combined_df[combined_df['condition'] == condition]

                valid_growth = cond_df['error_growth_rate'].notna()
                if valid_growth.sum() > 0:
                    print(f"\n{condition}:")
                    print(f"  Mean error growth rate: {cond_df['error_growth_rate'].mean():.6f}")
                    print(f"  Mean uncertainty growth rate: {cond_df['uncertainty_growth_rate'].mean():.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
