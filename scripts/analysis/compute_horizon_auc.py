#!/usr/bin/env python3
"""
Compute Horizon-AUC (Area Under Error vs Horizon Curve)

This script computes the Area Under the Curve (AUC) for error vs horizon,
summarizing prediction accuracy across the entire prediction horizon.

Higher Horizon-AUC indicates worse long-horizon performance (errors accumulate).

Input:
    hallucination_horizon_<condition>.csv
    Columns: horizon_t, error_rel_mean, error_abs_mean, etc.

Output:
    horizon_auc_<condition>.csv
    New columns:
        - horizon_auc_rel: AUC of relative error curve
        - horizon_auc_abs: AUC of absolute error curve
        - auc_normalized: AUC normalized by max possible

Usage:
    python compute_horizon_auc.py \
        --input_dir results/paper/hallucination_horizon \
        --output_dir results/paper/horizon_auc
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def compute_horizon_auc_single(
    df: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Compute Horizon-AUC for a single condition.

    Args:
        df: DataFrame with error data per horizon step
        normalize: Whether to normalize AUC by max possible value

    Returns:
        DataFrame with added AUC columns
    """
    results = []

    # Group by unique runs
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

        # Extract error curves
        if 'error_rel_mean' in group_df.columns:
            error_rel = group_df['error_rel_mean'].values
        else:
            print(f"  WARNING: No error_rel_mean column")
            error_rel = None

        if 'error_abs_mean' in group_df.columns:
            error_abs = group_df['error_abs_mean'].values
        else:
            error_abs = None

        # Compute AUC using trapezoidal rule
        def compute_auc(error_series):
            if error_series is None or len(error_series) < 2:
                return np.nan

            # Remove NaN values
            valid_mask = ~np.isnan(error_series)
            if valid_mask.sum() < 2:
                return np.nan

            x = np.arange(len(error_series))[valid_mask]
            y = error_series[valid_mask]

            # Trapezoidal integration
            auc = np.trapz(y, x)

            # Normalize by max possible (max_y * max_x)
            if normalize:
                max_auc = y.max() * x.max()
                if max_auc > 0:
                    auc = auc / max_auc

            return auc

        # Compute AUCs
        horizon_auc_rel = compute_auc(error_rel) if error_rel is not None else np.nan
        horizon_auc_abs = compute_auc(error_abs) if error_abs is not None else np.nan

        results.append({
            **{col: group_df[col].iloc[0] if len(group_df) > 0 else np.nan
                   for col in group_cols},
            'horizon_auc_rel': horizon_auc_rel,
            'horizon_auc_abs': horizon_auc_abs,
            'n_samples': len(group_df),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compute Horizon-AUC from error vs horizon data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing hallucination_horizon CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for Horizon-AUC CSV files')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize AUC by max possible value (default: True)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing Horizon-AUC from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Normalize AUC: {args.normalize}")

    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    all_results = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            # Compute Horizon-AUC
            results_df = compute_horizon_auc_single(df, args.normalize)
            results_df.to_csv(output_dir / csv_file.name, index=False)
            all_results.append(results_df)

            # Print summary for this file
            valid_auc = results_df['horizon_auc_rel'].notna()
            if valid_auc.sum() > 0:
                print(f"  Mean Horizon-AUC (rel): {results_df['horizon_auc_rel'].mean():.4f} ± {results_df['horizon_auc_rel'].std():.4f}")
            else:
                print(f"  WARNING: No valid AUCs computed")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save combined file
        combined_path = output_dir / "horizon_auc_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined results: {combined_path}")

        # Aggregate by condition
        if 'condition' in combined_df.columns:
            print("\nAggregating by condition...")
            for condition in combined_df['condition'].unique():
                cond_df = combined_df[combined_df['condition'] == condition]

                valid_auc = cond_df['horizon_auc_rel'].notna()
                if valid_auc.sum() > 0:
                    print(f"\n{condition}:")
                    print(f"  Mean Horizon-AUC: {cond_df['horizon_auc_rel'].mean():.4f}")
                    print(f"  Std Horizon-AUC: {cond_df['horizon_auc_rel'].std():.4f}")
                    print(f"  N runs: {valid_auc.sum()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
