#!/usr/bin/env python3
"""
Compute Risk-Coverage Curve

This script computes risk-coverage curves, showing mean error when filtering
predictions by low epistemic uncertainty. This evaluates whether uncertainty
is a useful signal for filtering unreliable predictions.

Risk-coverage curve: For each percentile of uncertainty, compute mean error of
filtered predictions. Low coverage = high filtering, low risk.

Lower curve indicates better uncertainty as a signal of reliability.

Input:
    error_vs_uncertainty_<condition>.csv
    Columns: horizon_t, prediction_step, error_rel_mean, uncertainty_mean, etc.

Output:
    risk_coverage_<condition>.csv
    New columns:
        - uncertainty_percentile: Percentile of epistemic uncertainty
        - mean_error_filtered: Mean error of predictions below this percentile
        - coverage_fraction: Fraction of predictions kept (1 - percentile)
        - risk_at_X: Mean error at specific coverage levels

Usage:
    python compute_risk_coverage.py \
        --input_dir results/paper/error_vs_uncertainty \
        --output_dir results/paper/risk_coverage
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


def compute_risk_coverage_single(
    df: pd.DataFrame,
    coverage_levels: Optional[List[float]] = None,
    n_percentiles: int = 100,
    horizon_filter: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute risk-coverage curve for a single condition.

    Args:
        df: DataFrame with error and uncertainty per prediction step
        coverage_levels: List of coverage levels to extract (e.g., [0.5, 0.8])
        n_percentiles: Number of percentile points for curve
        horizon_filter: Minimum horizon step to include (e.g., 30 for h > 30)

    Returns:
        DataFrame with risk-coverage data points
    """
    if coverage_levels is None:
        coverage_levels = [0.5, 0.6, 0.7, 0.8, 0.9]

    results = []

    # Group by unique runs and horizon steps
    group_cols = ['condition', 'seed', 'checkpoint_step', 'horizon_t', 'prediction_step']

    for _, group_df in df.groupby(group_cols, dropna=False):
        if len(group_df) < 5:
            # Too few samples for meaningful curve
            continue

        # Extract error and uncertainty
        if 'error_rel_mean' not in group_df.columns:
            continue
        if 'uncertainty_mean' not in group_df.columns:
            continue

        errors = group_df['error_rel_mean'].values
        uncertainties = group_df['uncertainty_mean'].values

        # Remove NaN
        valid_mask = ~np.isnan(errors) & ~np.isnan(uncertainties)
        if valid_mask.sum() < 5:
            continue

        errors = errors[valid_mask]
        uncertainties = uncertainties[valid_mask]

        # Compute percentile thresholds
        percentiles = np.linspace(0, 100, n_percentiles)

        curve_data = []

        for pct in percentiles:
            # Get threshold at this percentile
            threshold = np.percentile(uncertainties, pct)

            # Filter predictions with uncertainty <= threshold
            filtered_mask = uncertainties <= threshold

            if filtered_mask.sum() > 0:
                mean_error_filtered = errors[filtered_mask].mean()
            else:
                mean_error_filtered = np.nan

            curve_data.append({
                **{col: group_df[col].iloc[0] for col in group_cols},
                'uncertainty_percentile': pct,
                'uncertainty_threshold': threshold,
                'mean_error_filtered': mean_error_filtered,
                'n_filtered': filtered_mask.sum(),
                'coverage_fraction': filtered_mask.sum() / len(uncertainties),
            })

        results.extend(curve_data)

    if not results:
        # Return empty DataFrame with same structure
        return pd.DataFrame(columns=group_cols + [
            'uncertainty_percentile', 'uncertainty_threshold',
            'mean_error_filtered', 'n_filtered', 'coverage_fraction'
        ])

    results_df = pd.DataFrame(results)

    # Aggregate by condition to get Risk@X metrics
    aggregated = []

    # Group by condition and horizon
    if 'condition' in results_df.columns and 'horizon_t' in results_df.columns:
        for (condition, horizon), group in results_df.groupby(['condition', 'horizon_t'], dropna=False):
            # Compute Risk@X for each coverage level
            risk_metrics = {col: group[col].iloc[0] for col in group_cols}

            for cov_level in coverage_levels:
                # Find closest coverage fraction
                closest_idx = (group['coverage_fraction'] - cov_level).abs().idxmin()
                risk_metrics[f'risk_at_{int(cov_level*100)}_coverage'] = group.iloc[closest_idx]['mean_error_filtered']

            aggregated.append(risk_metrics)

        aggregated_df = pd.DataFrame(aggregated)

        # Merge back to results
        if 'horizon_t' in group_cols:
            group_cols_no_horizon = [col for col in group_cols if col != 'horizon_t']
            results_df = results_df.merge(
                aggregated_df,
                on=['condition'] + group_cols_no_horizon,
                how='left',
                suffixes=('', '_agg')
            )

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Compute risk-coverage curves from error vs uncertainty data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing error_vs_uncertainty CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for risk-coverage CSV files')
    parser.add_argument('--coverage_levels', type=float, nargs='+',
                        default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help='Coverage levels to extract risk metrics (default: 0.5 0.6 0.7 0.8 0.9)')
    parser.add_argument('--n_percentiles', type=int, default=100,
                        help='Number of percentile points for curve (default: 100)')
    parser.add_argument('--horizon_filter', type=int, default=30,
                        help='Minimum horizon step to include (default: 30 for consistency with AUROC)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing risk-coverage from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Coverage levels: {args.coverage_levels}")
    print(f"Horizon filter: h > {args.horizon_filter} if specified")

    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    all_results = []

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")

        try:
            df = pd.read_csv(csv_file)

            # Compute risk-coverage
            results_df = compute_risk_coverage_single(
                df,
                args.coverage_levels,
                args.n_percentiles,
                horizon_filter=args.horizon_filter
            )
            results_df.to_csv(output_dir / csv_file.name, index=False)
            all_results.append(results_df)

            # Print summary for this file
            if f'risk_at_{int(args.coverage_levels[-1]*100)}_coverage' in results_df.columns:
                risk_80 = results_df[f'risk_at_{int(args.coverage_levels[-1]*100)}_coverage'].dropna()
                if len(risk_80) > 0:
                    print(f"  Risk@{int(args.coverage_levels[-1]*100)}% coverage: {risk_80.mean():.4f} ± {risk_80.std():.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save combined file
        combined_path = output_dir / "risk_coverage_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved combined results: {combined_path}")

        # Aggregate by condition
        if 'condition' in combined_df.columns:
            print("\nAggregating by condition...")
            for condition in combined_df['condition'].unique():
                cond_df = combined_df[combined_df['condition'] == condition]

                for cov_level in args.coverage_levels:
                    col = f'risk_at_{int(cov_level*100)}_coverage'
                    if col in cond_df.columns:
                        risk = cond_df[col].dropna()
                        if len(risk) > 0:
                            print(f"  {condition} Risk@{int(cov_level*100)}: {risk.mean():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
