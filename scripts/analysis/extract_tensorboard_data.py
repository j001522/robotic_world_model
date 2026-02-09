#!/usr/bin/env python3
"""
Extract TensorBoard data from training runs and export to CSV for paper figures.

This script:
1. Scans log directories for TensorBoard event files
2. Parses run names to extract experimental metadata (method, seed, penalty, etc.)
3. Groups runs by experimental condition (same config, different seeds)
4. Exports data to CSV files with proper structure for plotting with mean/std

Usage:
    python extract_tensorboard_data.py --log_dir logs/rsl_rl/anymal_d_flat --output_dir results/
    python extract_tensorboard_data.py --log_dir logs/rsl_rl/anymal_d_flat --output_dir results/ --metrics "Train/mean_reward" "Loss/value_function"
    python extract_tensorboard_data.py --log_dir logs/rsl_rl/anymal_d_flat --output_dir results/ --filter "finetune"

Output structure:
    results/
    ├── all_runs.csv                    # All runs with metadata columns
    ├── aggregated/                     # Aggregated by condition (mean, std, min, max)
    │   ├── finetune-rp-pen025-std.csv
    │   ├── finetune-bs-pen025-var.csv
    │   └── ...
    └── individual/                     # Individual run CSVs
        ├── 2026-02-02_23-56-23_finetune-bs-pen025-seed42.csv
        └── ...
"""

import argparse
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import warnings

import pandas as pd
import numpy as np

# TensorBoard imports
try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard not installed. Install with: pip install tensorboard")
    sys.exit(1)


@dataclass
class RunMetadata:
    """Parsed metadata from a run directory name."""
    full_name: str
    timestamp: str
    phase: str  # 'pretrain' or 'finetune'
    method: str  # 'bs' (bootstrap), 'rp' (randomized priors), 'ensemble'
    seed: int
    penalty: Optional[float] = None  # e.g., 0.25 from 'pen025'
    uncertainty_metric: Optional[str] = None  # 'std' or 'var'
    prior_scale: Optional[float] = None  # e.g., 1.0 from 'prior1.0'
    bootstrap: Optional[bool] = None  # from 'noboot' or 'boot'
    
    # Condition key groups runs with same config but different seeds
    condition_key: str = field(init=False)
    
    def __post_init__(self):
        # Build condition key (everything except seed and timestamp)
        parts = [self.phase, self.method]
        if self.penalty is not None:
            parts.append(f"pen{self.penalty}")
        else:
            parts.append("nopen")  # No penalty
        if self.uncertainty_metric:
            parts.append(self.uncertainty_metric)
        if self.prior_scale is not None:
            parts.append(f"prior{self.prior_scale}")
        if self.bootstrap is not None:
            parts.append("boot" if self.bootstrap else "noboot")
        self.condition_key = "-".join(parts)


def parse_run_name(run_dir: str) -> Optional[RunMetadata]:
    """
    Parse a run directory name to extract metadata.
    
    Expected formats:
    - 2026-02-02_19-35-08_pretrain-ensemble-prior1.0-noboot-seed42
    - 2026-02-02_23-56-23_finetune-bs-pen025-seed42
    - 2026-02-03_15-16-13_finetune-rp-pen025-std-seed42
    """
    dir_name = os.path.basename(run_dir)
    
    # Extract timestamp (YYYY-MM-DD_HH-MM-SS)
    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)', dir_name)
    if not timestamp_match:
        return None
    
    timestamp = timestamp_match.group(1)
    run_name = timestamp_match.group(2)
    
    # Extract seed
    seed_match = re.search(r'seed(\d+)', run_name)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))
    
    # Determine phase
    if 'pretrain' in run_name:
        phase = 'pretrain'
    elif 'finetune' in run_name:
        phase = 'finetune'
    else:
        phase = 'unknown'
    
    # Determine method
    if '-rp-' in run_name or 'priors' in run_name or 'prior' in run_name:
        method = 'rp'  # randomized priors
    elif '-bs-' in run_name or 'bootstrap' in run_name:
        method = 'bs'  # bootstrap
    elif 'ensemble' in run_name:
        method = 'ensemble'
    else:
        method = 'unknown'
    
    # Extract penalty (e.g., pen025 -> 0.25)
    penalty = None
    penalty_match = re.search(r'pen(\d+)', run_name)
    if penalty_match:
        penalty = int(penalty_match.group(1)) / 100.0
    
    # Extract uncertainty metric
    uncertainty_metric = None
    if '-std' in run_name or '-std-' in run_name:
        uncertainty_metric = 'std'
    elif '-var' in run_name or '-var-' in run_name:
        uncertainty_metric = 'var'
    
    # Extract prior scale (e.g., prior1.0 -> 1.0)
    prior_scale = None
    prior_match = re.search(r'prior([\d.]+)', run_name)
    if prior_match:
        prior_scale = float(prior_match.group(1))
    
    # Extract bootstrap flag
    bootstrap = None
    if 'noboot' in run_name:
        bootstrap = False
    elif 'boot' in run_name and 'noboot' not in run_name:
        bootstrap = True
    
    return RunMetadata(
        full_name=dir_name,
        timestamp=timestamp,
        phase=phase,
        method=method,
        seed=seed,
        penalty=penalty,
        uncertainty_metric=uncertainty_metric,
        prior_scale=prior_scale,
        bootstrap=bootstrap,
    )


def find_event_files(run_dir: str) -> list[str]:
    """Find all TensorBoard event files in a run directory."""
    event_files = []
    for f in os.listdir(run_dir):
        if f.startswith('events.out.tfevents'):
            event_files.append(os.path.join(run_dir, f))
    return sorted(event_files)


def extract_scalars_from_run(run_dir: str, metrics: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Extract scalar metrics from TensorBoard event files.
    
    Args:
        run_dir: Path to the run directory containing event files
        metrics: Optional list of specific metrics to extract. If None, extracts all.
    
    Returns:
        DataFrame with columns: step, wall_time, metric_name, value
    """
    event_files = find_event_files(run_dir)
    if not event_files:
        warnings.warn(f"No event files found in {run_dir}")
        return pd.DataFrame()
    
    # Use the most recent event file (in case there are multiple)
    event_file = event_files[-1]
    
    # Load events
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()
    
    # Get available scalar tags
    available_tags = ea.Tags().get('scalars', [])
    
    if metrics is not None:
        # Filter to requested metrics
        tags_to_extract = [t for t in available_tags if t in metrics]
    else:
        tags_to_extract = available_tags
    
    # Extract data
    records = []
    for tag in tags_to_extract:
        try:
            events = ea.Scalars(tag)
            for event in events:
                records.append({
                    'step': event.step,
                    'wall_time': event.wall_time,
                    'metric': tag,
                    'value': event.value,
                })
        except Exception as e:
            warnings.warn(f"Error extracting {tag} from {run_dir}: {e}")
    
    return pd.DataFrame(records)


def extract_all_runs(
    log_dir: str,
    metrics: Optional[list[str]] = None,
    filter_pattern: Optional[str] = None,
) -> tuple[pd.DataFrame, dict[str, list[RunMetadata]]]:
    """
    Extract data from all runs in a log directory.
    
    Returns:
        - DataFrame with all data and metadata columns
        - Dict mapping condition_key to list of RunMetadata
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise ValueError(f"Log directory not found: {log_dir}")
    
    all_data = []
    conditions = defaultdict(list)
    
    # Find all run directories
    run_dirs = sorted([d for d in log_path.iterdir() if d.is_dir()])
    
    print(f"Found {len(run_dirs)} run directories")
    
    for run_dir in run_dirs:
        dir_name = run_dir.name
        
        # Apply filter if specified
        if filter_pattern and filter_pattern not in dir_name:
            continue
        
        # Parse metadata
        metadata = parse_run_name(str(run_dir))
        if metadata is None:
            print(f"  Skipping (could not parse): {dir_name}")
            continue
        
        print(f"  Processing: {dir_name}")
        print(f"    Phase: {metadata.phase}, Method: {metadata.method}, Seed: {metadata.seed}")
        print(f"    Condition: {metadata.condition_key}")
        
        # Extract scalars
        df = extract_scalars_from_run(str(run_dir), metrics)
        
        if df.empty:
            print(f"    WARNING: No data extracted")
            continue
        
        # Add metadata columns
        df['run_name'] = metadata.full_name
        df['condition'] = metadata.condition_key
        df['phase'] = metadata.phase
        df['method'] = metadata.method
        df['seed'] = metadata.seed
        df['penalty'] = metadata.penalty
        df['uncertainty_metric'] = metadata.uncertainty_metric
        df['prior_scale'] = metadata.prior_scale
        df['bootstrap'] = metadata.bootstrap
        
        all_data.append(df)
        conditions[metadata.condition_key].append(metadata)
    
    if not all_data:
        return pd.DataFrame(), {}
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df, dict(conditions)


def pivot_and_aggregate(
    df: pd.DataFrame,
    condition: str,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Pivot data for a condition and compute aggregations across seeds.
    
    Returns DataFrame with columns:
        step, <metric>_mean, <metric>_std, <metric>_min, <metric>_max, <metric>_count
    for each metric.
    """
    # Filter to this condition
    condition_df = df[df['condition'] == condition].copy()
    
    if condition_df.empty:
        return pd.DataFrame()
    
    # Get unique metrics
    if metrics is not None:
        available_metrics = [m for m in metrics if m in condition_df['metric'].unique()]
    else:
        available_metrics = condition_df['metric'].unique()
    
    # Pivot each metric and aggregate
    aggregated_data = {}
    
    for metric in available_metrics:
        metric_df = condition_df[condition_df['metric'] == metric]
        
        # Pivot: rows=step, columns=seed, values=value
        pivoted = metric_df.pivot_table(
            index='step',
            columns='seed',
            values='value',
            aggfunc='first'  # In case of duplicates
        )
        
        # Compute aggregations
        safe_name = metric.replace('/', '_').replace(' ', '_')
        aggregated_data[f'{safe_name}_mean'] = pivoted.mean(axis=1)
        aggregated_data[f'{safe_name}_std'] = pivoted.std(axis=1)
        aggregated_data[f'{safe_name}_min'] = pivoted.min(axis=1)
        aggregated_data[f'{safe_name}_max'] = pivoted.max(axis=1)
        aggregated_data[f'{safe_name}_count'] = pivoted.count(axis=1)
        
        # Also include individual seed columns
        for seed in pivoted.columns:
            aggregated_data[f'{safe_name}_seed{seed}'] = pivoted[seed]
    
    result_df = pd.DataFrame(aggregated_data)
    result_df.index.name = 'step'
    result_df = result_df.reset_index()
    
    return result_df


def save_individual_runs(df: pd.DataFrame, output_dir: Path):
    """Save individual run data to separate CSV files."""
    individual_dir = output_dir / 'individual'
    individual_dir.mkdir(parents=True, exist_ok=True)
    
    for run_name in df['run_name'].unique():
        run_df = df[df['run_name'] == run_name]
        
        # Pivot to wide format: step as index, metrics as columns
        pivoted = run_df.pivot_table(
            index='step',
            columns='metric',
            values='value',
            aggfunc='first'
        )
        pivoted = pivoted.reset_index()
        
        # Clean column names for CSV
        pivoted.columns = [c.replace('/', '_').replace(' ', '_') for c in pivoted.columns]
        
        output_path = individual_dir / f'{run_name}.csv'
        pivoted.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")


def save_aggregated_runs(
    df: pd.DataFrame,
    conditions: dict[str, list[RunMetadata]],
    output_dir: Path,
    metrics: Optional[list[str]] = None,
):
    """Save aggregated data per condition."""
    aggregated_dir = output_dir / 'aggregated'
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    
    for condition_key, runs in conditions.items():
        print(f"\nAggregating condition: {condition_key}")
        print(f"  Seeds: {[r.seed for r in runs]}")
        
        agg_df = pivot_and_aggregate(df, condition_key, metrics)
        
        if agg_df.empty:
            print(f"  WARNING: No data for condition {condition_key}")
            continue
        
        output_path = aggregated_dir / f'{condition_key}.csv'
        agg_df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")


def list_available_metrics(log_dir: str, sample_run: Optional[str] = None) -> list[str]:
    """List all available metrics from a sample run."""
    log_path = Path(log_dir)
    
    if sample_run:
        run_dir = log_path / sample_run
    else:
        # Use first available run
        run_dirs = sorted([d for d in log_path.iterdir() if d.is_dir()])
        if not run_dirs:
            return []
        run_dir = run_dirs[0]
    
    event_files = find_event_files(str(run_dir))
    if not event_files:
        return []
    
    ea = event_accumulator.EventAccumulator(
        event_files[-1],
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    
    return sorted(ea.Tags().get('scalars', []))


def main():
    parser = argparse.ArgumentParser(
        description='Extract TensorBoard data from training runs for paper figures.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/rsl_rl/anymal_d_flat',
        help='Directory containing run subdirectories (default: logs/rsl_rl/anymal_d_flat)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/tensorboard_export',
        help='Output directory for CSV files (default: results/tensorboard_export)'
    )
    
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        help='Specific metrics to extract (default: all). Example: --metrics "Train/mean_reward" "Loss/value_function"'
    )
    
    parser.add_argument(
        '--filter',
        type=str,
        default='finetune',
        help='Filter runs by pattern (default: "finetune"). Use --filter "" for no filtering.'
    )
    
    parser.add_argument(
        '--runs',
        type=str,
        nargs='+',
        default=None,
        help='Specific run directories to extract (fallback when name parsing fails). '
             'Can be full paths or just directory names within log_dir. '
             'Example: --runs "2026-01-25_18-04-51_finetune-paper-baseline" --condition_name "baseline"'
    )
    
    parser.add_argument(
        '--condition_name',
        type=str,
        default=None,
        help='Condition name to use for --runs (required when using --runs with unparseable names)'
    )
    
    parser.add_argument(
        '--seed_for_runs',
        type=int,
        default=0,
        help='Seed value to assign to runs specified via --runs (default: 0 for single runs)'
    )
    
    parser.add_argument(
        '--list_metrics',
        action='store_true',
        help='List available metrics and exit'
    )
    
    parser.add_argument(
        '--no_individual',
        action='store_true',
        help='Skip saving individual run CSVs (only save aggregated)'
    )
    
    args = parser.parse_args()
    
    # List metrics mode
    if args.list_metrics:
        print(f"Available metrics in {args.log_dir}:\n")
        metrics = list_available_metrics(args.log_dir)
        for m in metrics:
            print(f"  {m}")
        print(f"\nTotal: {len(metrics)} metrics")
        return
    
    # Direct run extraction mode (fallback for unparseable names)
    if args.runs:
        if not args.condition_name:
            print("ERROR: --condition_name is required when using --runs")
            return
        
        print(f"Direct run extraction mode")
        print(f"Condition name: {args.condition_name}")
        print(f"Runs: {args.runs}")
        
        log_path = Path(args.log_dir)
        all_data = []
        runs_metadata = []
        
        for i, run_spec in enumerate(args.runs):
            # Handle both full paths and directory names
            if os.path.isabs(run_spec):
                run_dir = Path(run_spec)
            else:
                run_dir = log_path / run_spec
            
            if not run_dir.exists():
                print(f"  WARNING: Run directory not found: {run_dir}")
                continue
            
            print(f"  Processing: {run_dir.name}")
            
            df = extract_scalars_from_run(str(run_dir), args.metrics)
            if df.empty:
                print(f"    WARNING: No data extracted")
                continue
            
            # Assign seed: use seed_for_runs if single run, otherwise increment
            seed = args.seed_for_runs + i if len(args.runs) > 1 else args.seed_for_runs
            
            # Create metadata
            metadata = RunMetadata(
                full_name=run_dir.name,
                timestamp="",
                phase="finetune" if "finetune" in run_dir.name.lower() else "pretrain",
                method="unknown",
                seed=seed,
            )
            # Override condition key
            metadata.condition_key = args.condition_name
            
            df['run_name'] = run_dir.name
            df['condition'] = args.condition_name
            df['phase'] = metadata.phase
            df['method'] = metadata.method
            df['seed'] = seed
            df['penalty'] = None
            df['uncertainty_metric'] = None
            df['prior_scale'] = None
            df['bootstrap'] = None
            
            all_data.append(df)
            runs_metadata.append(metadata)
        
        if not all_data:
            print("No data extracted!")
            return
        
        df = pd.concat(all_data, ignore_index=True)
        conditions = {args.condition_name: runs_metadata}
        
    else:
        # Standard extraction mode
        print(f"Extracting TensorBoard data from: {args.log_dir}")
        print(f"Output directory: {args.output_dir}")
        if args.metrics:
            print(f"Metrics: {args.metrics}")
        if args.filter:
            print(f"Filter: {args.filter}")
        print()
        
        # Extract all data
        df, conditions = extract_all_runs(
            args.log_dir,
            metrics=args.metrics,
            filter_pattern=args.filter,
        )
    
    if df.empty:
        print("No data extracted!")
        return
    
    print(f"\nExtracted {len(df)} data points from {df['run_name'].nunique()} runs")
    print(f"Conditions found: {list(conditions.keys())}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save master CSV with all data
    master_path = output_dir / 'all_runs.csv'
    df.to_csv(master_path, index=False)
    print(f"\nSaved master file: {master_path}")
    
    # Save individual runs
    if not args.no_individual:
        print("\nSaving individual run CSVs...")
        save_individual_runs(df, output_dir)
    
    # Save aggregated data
    print("\nSaving aggregated CSVs...")
    save_aggregated_runs(df, conditions, output_dir, args.metrics)
    
    # Print summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Master CSV: {master_path}")
    print(f"Individual CSVs: {output_dir / 'individual'}/")
    print(f"Aggregated CSVs: {output_dir / 'aggregated'}/")
    print("\nConditions and seeds:")
    for condition, runs in conditions.items():
        seeds = sorted([r.seed for r in runs])
        print(f"  {condition}: seeds {seeds}")


if __name__ == '__main__':
    main()
