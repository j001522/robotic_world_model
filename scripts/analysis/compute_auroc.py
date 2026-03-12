#!/usr/bin/env python3
"""
Compute AUROC (Area Under ROC Curve) for High-Error Detection

This script computes the Area Under the ROC Curve (AUROC) to evaluate
whether epistemic uncertainty can predict high prediction errors.

Higher AUROC indicates better uncertainty calibration for failure prediction.

Label definition:
- Positive class: Error in top 10% (high error event)
- Score: Epistemic uncertainty

Input:
    error_vs_uncertainty_<condition>.csv
    Columns: horizon_t, prediction_step, error_rel_mean, uncertainty_mean, etc.

Output:
    auroc_<condition>.csv
    New columns:
        - auroc: Area under ROC curve
        - optimal_threshold: Best threshold for F1 score
        - f1_max: Maximum F1 score
        - tpr_at_0.1_fpr: True positive rate at 10% false positive rate

Usage:
    python compute_auroc.py \
        --input_dir results/paper/error_vs_uncertainty \
        --output_dir results/paper/auroc \
        --top_percentile 10
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_auroc_single(
    df: pd.DataFrame,
    top_percentile: float = 10.0,
    horizon_min: Optional[int] = None,
    per_horizon: bool = True,
) -> pd.DataFrame:
    """
    Compute AUROC for high-error detection.

    Args:
        df: DataFrame with error and uncertainty per prediction step
        top_percentile: Top percentile to label as "high error" (default: 10%)
        horizon_min: Minimum horizon step to include (default: all)
        per_horizon: If True, compute AUROC per horizon (recommended for line plots)
                    If False, pool all horizons together (deprecated)

    Returns:
        DataFrame with AUROC metrics per run (and per horizon if per_horizon=True)
    """
    results = []

    # Determine grouping columns based on per_horizon flag
    if per_horizon:
        # Group by unique runs AND horizon step
        group_cols = ['condition', 'seed', 'checkpoint_step', 'horizon_t']
    else:
        # Group only by unique runs (pooled across horizons)
        group_cols = ['condition', 'seed', 'checkpoint_step']

    for group_keys, group_df in df.groupby(group_cols, dropna=False):
        if len(group_df) < 10:
            # Too few samples for ROC
            record = {col: group_keys[i] if i < len(group_keys) else np.nan
                      for i, col in enumerate(group_cols)}
            record['auroc'] = np.nan
            results.append(record)
            continue

        # Check required columns
        if 'error_rel_mean' not in group_df.columns:
            continue
        if 'uncertainty_mean' not in group_df.columns:
            continue

        # Extract data
        errors = group_df['error_rel_mean'].values
        uncertainties = group_df['uncertainty_mean'].values

        # Remove NaN
        valid_mask = ~np.isnan(errors) & ~np.isnan(uncertainties)
        if valid_mask.sum() < 10:
            record = {col: group_keys[i] if i < len(group_keys) else np.nan
                      for i, col in enumerate(group_cols)}
            record['auroc'] = np.nan
            results.append(record)
            continue

        errors = errors[valid_mask]
        uncertainties = uncertainties[valid_mask]

        # Define labels: top X% = positive (high error)
        # IMPORTANT: Compute threshold within this group (per-horizon or per-run)
        error_threshold = np.percentile(errors, 100.0 - top_percentile)
        labels = (errors >= error_threshold).astype(int)

        # Compute ROC curve
        # Higher uncertainty = more likely to be positive (high error)
        fpr, tpr, thresholds = roc_curve(labels, uncertainties, pos_label=1)
        auroc_value = auc(fpr, tpr)

        # Find optimal threshold (maximizes TPR - FPR or closest to (0,1))
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)

        # Compute TPR at 10% FPR (low false positive rate = low filtering)
        fpr_10_idx = np.argmin(np.abs(fpr - 0.1))
        tpr_at_fpr_10 = tpr[fpr_10_idx]

        record = {
            col: group_keys[i] if i < len(group_keys) else np.nan
            for i, col in enumerate(group_cols)
        }
        record['auroc'] = auroc_value
        record['optimal_threshold'] = thresholds[optimal_idx]
        record['optimal_f1'] = 2 * tpr[optimal_idx] * fpr[optimal_idx] / (tpr[optimal_idx] + fpr[optimal_idx] + 1e-8)
        record['tpr_at_0.1_fpr'] = tpr_at_fpr_10
        record['error_threshold'] = error_threshold
        record['n_positive'] = labels.sum()
        record['n_samples'] = len(labels)
        record['n_filtered'] = fpr_10_idx if fpr_10_idx < len(fpr) else len(fpr)
        results.append(record)

    # Filter by horizon_min if specified (only for non-per_horizon mode)
    if not per_horizon and horizon_min is not None and 'horizon_t' in df.columns:
        results_df = pd.DataFrame(results)
        results_df = results_df[results_df['horizon_t'] >= horizon_min]
        return results_df

    return pd.DataFrame(results)


def compute_auroc_aggregated(df, horizon_filter=None):
    """Compute mean AUROC per condition, with optional horizon filter."""
    if horizon_filter is not None:
        df_filtered = df[df['prediction_step'] >= horizon_filter].copy()
    else:
        df_filtered = df.copy()

    aggregated = []

    for condition in df_filtered['condition'].unique():
        cond_df = df_filtered[df_filtered['condition'] == condition]

        valid_auroc = cond_df['auroc'].notna()

        if valid_auroc.sum() > 0:
            auroc_mean = cond_df['auroc'].mean()
            auroc_std = cond_df['auroc'].std()

            aggregated.append({
                'condition': condition,
                'auroc_mean': auroc_mean,
                'auroc_std': auroc_std,
                'auroc_min': cond_df['auroc'].min(),
                'auroc_max': cond_df['auroc'].max(),
                'n_runs': valid_auroc.sum(),
            })
        else:
            aggregated.append({
                'condition': condition,
                'auroc_mean': np.nan,
                'auroc_std': np.nan,
                'auroc_min': np.nan,
                'auroc_max': np.nan,
                'n_runs': 0,
            })

    return pd.DataFrame(aggregated)


def main():
    parser = argparse.ArgumentParser(
        description='Compute AUROC for high-error detection from error vs uncertainty data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing error_vs_uncertainty CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for AUROC CSV files')
    parser.add_argument('--top_percentile', type=float, default=10.0,
                        help='Top percentile to label as "high error" (default: 10.0)')
    parser.add_argument('--horizon_min', type=int, default=None,
                        help='Minimum horizon step for AUROC computation (default: all)')
    parser.add_argument('--horizon_filter', type=int, default=30,
                        help='Horizon filter for aggregated mean AUROC (default: 30)')
    parser.add_argument('--per_horizon', action='store_true', default=True,
                        help='Compute AUROC per horizon (recommended for line plots)')
    parser.add_argument('--pooled', action='store_true',
                        help='Pool all horizons together (deprecated)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether to use per-horizon or pooled AUROC
    use_per_horizon = args.per_horizon and not args.pooled

    print(f"Computing AUROC from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Top percentile for high error: {args.top_percentile}%")
    print(f"Per-horizon mode: {use_per_horizon}")

    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    all_results = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            # Compute AUROC
            results_df = compute_auroc_single(
                df,
                args.top_percentile,
                args.horizon_min,
                per_horizon=use_per_horizon
            )
            results_df.to_csv(output_dir / csv_file.name, index=False)
            all_results.append(results_df)

            # Print summary for this file
            valid_auroc = results_df['auroc'].notna()
            if valid_auroc.sum() > 0:
                print(f"  Mean AUROC: {results_df['auroc'].mean():.4f} ± {results_df['auroc'].std():.4f}")
            else:
                print(f"  WARNING: No valid AUROCs computed")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save combined file
        combined_path = output_dir / "auroc_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined results: {combined_path}")

        # Aggregate by condition
        if 'condition' in combined_df.columns:
            print("\nAggregating by condition...")

            # Overall AUROC (all horizons)
            overall_agg = compute_auroc_aggregated(combined_df, horizon_filter=None)
            print("\nOverall AUROC (all horizons):")
            for _, row in overall_agg.iterrows():
                print(f"  {row['condition']}: {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f} (n={row['n_runs']})")

            # AUROC for h > horizon_filter
            if args.horizon_filter is not None and 'prediction_step' in combined_df.columns:
                long_horizon_agg = compute_auroc_aggregated(combined_df, horizon_filter=args.horizon_filter)
                print(f"\nAUROC (h > {args.horizon_filter}):")
                for _, row in long_horizon_agg.iterrows():
                    print(f"  {row['condition']}: {row['auroc_mean']:.4f} ± {row['auroc_std']:.4f} (n={row['n_runs']})")

    print("\nDone!")


if __name__ == "__main__":
    main()
