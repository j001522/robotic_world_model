#!/usr/bin/env python3
"""
Batch evaluate all trained models using pre-recorded trajectories.

This script discovers all finetune runs and their corresponding trajectory files,
then runs all evaluations (hallucination horizon, noise robustness, error vs uncertainty).

Usage:
    # Run all evaluations
    python batch_evaluate.py \
        --log_dir ../../logs/rsl_rl/anymal_d_flat \
        --trajectory_dir ../../results/paper/trajectories \
        --output_dir ../../results/paper

    # Run specific evaluations only
    python batch_evaluate.py \
        --log_dir ../../logs/rsl_rl/anymal_d_flat \
        --trajectory_dir ../../results/paper/trajectories \
        --output_dir ../../results/paper \
        --evaluations hallucination_horizon noise_robustness

    # Dry run
    python batch_evaluate.py \
        --log_dir ../../logs/rsl_rl/anymal_d_flat \
        --trajectory_dir ../../results/paper/trajectories \
        --output_dir ../../results/paper \
        --dry_run

Expected trajectory structure:
    trajectories/
    ├── finetune-bs-nopen/
    │   ├── seed42_step5000.pt
    │   └── ...
    └── finetune-rp-pen025-std/
        └── ...

Output structure:
    results/paper/
    ├── hallucination_horizon/
    │   └── *.csv
    ├── noise_robustness/
    │   └── *.csv
    └── error_vs_uncertainty/
        └── *.csv
"""

import argparse
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
import pandas as pd

# Import shared utilities
from utils import (
    parse_run_name,
    find_checkpoints,
    load_checkpoint_and_config,
    infer_dimensions_from_checkpoint,
    ensure_rsl_rl_rwm_in_path,
    STATE_COMPONENTS,
    VELOCITY_INDICES,
)


def discover_trajectories(trajectory_dir: Path) -> dict[str, dict[int, dict[int, Path]]]:
    """
    Discover all trajectory files organized by condition/seed/step.
    
    Returns:
        Dict: condition -> seed -> step -> trajectory_path
    """
    trajectories = defaultdict(lambda: defaultdict(dict))
    
    for condition_dir in trajectory_dir.iterdir():
        if not condition_dir.is_dir():
            continue
        
        condition = condition_dir.name
        
        for traj_file in condition_dir.glob("*.pt"):
            # Parse filename: seed42_step5000.pt
            match = re.match(r'seed(\d+)_step(\d+)\.pt', traj_file.name)
            if match:
                seed = int(match.group(1))
                step = int(match.group(2))
                trajectories[condition][seed][step] = traj_file
    
    return dict(trajectories)


def discover_checkpoints(log_dir: Path, filter_pattern: str = "finetune") -> dict[str, dict[int, list[tuple[int, Path]]]]:
    """
    Discover all checkpoints organized by condition/seed.
    
    Returns:
        Dict: condition -> seed -> list of (step, checkpoint_path)
    """
    checkpoints = defaultdict(lambda: defaultdict(list))
    
    for run_dir in log_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if filter_pattern and filter_pattern not in run_dir.name:
            continue
        
        metadata = parse_run_name(str(run_dir))
        if metadata is None:
            continue
        
        ckpts = find_checkpoints(str(run_dir))
        if ckpts:
            checkpoints[metadata.condition_key][metadata.seed] = ckpts
    
    return dict(checkpoints)


def match_trajectories_to_checkpoints(
    trajectories: dict,
    checkpoints: dict,
) -> list[tuple[str, int, int, Path, Path]]:
    """
    Match trajectory files to their corresponding checkpoints.
    
    Returns:
        List of (condition, seed, step, trajectory_path, checkpoint_path)
    """
    matches = []
    
    for condition, seeds in trajectories.items():
        if condition not in checkpoints:
            print(f"  WARNING: No checkpoints found for condition {condition}")
            continue
        
        for seed, steps in seeds.items():
            if seed not in checkpoints[condition]:
                print(f"  WARNING: No checkpoints found for {condition} seed {seed}")
                continue
            
            ckpt_steps = {step: path for step, path in checkpoints[condition][seed]}
            
            for step, traj_path in steps.items():
                if step in ckpt_steps:
                    matches.append((condition, seed, step, traj_path, ckpt_steps[step]))
                else:
                    print(f"  WARNING: No checkpoint for {condition} seed {seed} step {step}")
    
    return matches


def evaluate_hallucination_horizon_single(
    checkpoint_path: Path,
    trajectory_path: Path,
    horizon: int = 200,
    device: str = "cuda",
) -> Optional[pd.DataFrame]:
    """Evaluate hallucination horizon for a single checkpoint/trajectory pair."""
    from evaluate_hallucination_horizon import evaluate_hallucination_horizon_offline
    
    try:
        df = evaluate_hallucination_horizon_offline(
            str(checkpoint_path),
            str(trajectory_path),
            horizon=horizon,
            device=device,
        )
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def evaluate_noise_robustness_single(
    checkpoint_path: Path,
    trajectory_path: Path,
    noise_levels: list = [0.0, 0.1, 0.2, 0.4, 0.5, 0.8],
    device: str = "cuda",
) -> Optional[pd.DataFrame]:
    """Evaluate noise robustness for a single checkpoint/trajectory pair."""
    from evaluate_noise_robustness import evaluate_noise_robustness_offline
    
    try:
        df = evaluate_noise_robustness_offline(
            str(checkpoint_path),
            str(trajectory_path),
            noise_levels=noise_levels,
            device=device,
        )
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def evaluate_error_vs_uncertainty_single(
    checkpoint_path: Path,
    trajectory_path: Path,
    horizon: int = 200,
    device: str = "cuda",
) -> Optional[pd.DataFrame]:
    """Evaluate error vs uncertainty for a single checkpoint/trajectory pair."""
    from evaluate_error_vs_uncertainty import evaluate_error_vs_uncertainty
    
    try:
        df = evaluate_error_vs_uncertainty(
            str(checkpoint_path),
            str(trajectory_path),
            horizon=horizon,
            device=device,
        )
        return df
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def run_evaluations(
    matches: list[tuple[str, int, int, Path, Path]],
    output_dir: Path,
    evaluations: list[str],
    horizon: int = 200,
    noise_levels: list = [0.0, 0.1, 0.2, 0.4, 0.5, 0.8],
    device: str = "cuda",
    dry_run: bool = False,
):
    """Run all requested evaluations on matched trajectory/checkpoint pairs."""
    
    # Storage for results
    results = {
        'hallucination_horizon': [],
        'noise_robustness': [],
        'error_vs_uncertainty': [],
    }
    
    for i, (condition, seed, step, traj_path, ckpt_path) in enumerate(matches):
        print(f"\n[{i+1}/{len(matches)}] {condition} seed{seed} step{step}")
        print(f"  Trajectory: {traj_path}")
        print(f"  Checkpoint: {ckpt_path}")
        
        if dry_run:
            continue
        
        # Run requested evaluations
        if 'hallucination_horizon' in evaluations:
            print("  Evaluating hallucination horizon...")
            df = evaluate_hallucination_horizon_single(ckpt_path, traj_path, horizon, device)
            if df is not None:
                df['condition'] = condition
                df['seed'] = seed
                df['checkpoint_step'] = step
                results['hallucination_horizon'].append(df)
        
        if 'noise_robustness' in evaluations:
            print("  Evaluating noise robustness...")
            df = evaluate_noise_robustness_single(ckpt_path, traj_path, noise_levels, device)
            if df is not None:
                df['condition'] = condition
                df['seed'] = seed
                df['checkpoint_step'] = step
                results['noise_robustness'].append(df)
        
        if 'error_vs_uncertainty' in evaluations:
            print("  Evaluating error vs uncertainty...")
            df = evaluate_error_vs_uncertainty_single(ckpt_path, traj_path, horizon, device)
            if df is not None:
                df['condition'] = condition
                df['seed'] = seed
                df['checkpoint_step'] = step
                results['error_vs_uncertainty'].append(df)
    
    if dry_run:
        return
    
    # Save results
    for eval_name, dfs in results.items():
        if not dfs:
            continue
        
        eval_output_dir = output_dir / eval_name
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Combine all results
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Save combined file
        combined_path = eval_output_dir / f"{eval_name}_all.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"\nSaved: {combined_path}")
        
        # Save per-condition aggregated files
        for condition in combined_df['condition'].unique():
            cond_df = combined_df[combined_df['condition'] == condition]
            cond_path = eval_output_dir / f"{eval_name}_{condition}.csv"
            cond_df.to_csv(cond_path, index=False)
            print(f"Saved: {cond_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch evaluate all trained models using pre-recorded trajectories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing run subdirectories')
    parser.add_argument('--trajectory_dir', type=str, required=True,
                        help='Directory containing trajectory files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--filter', type=str, default='finetune',
                        help='Filter runs by pattern (default: "finetune")')
    parser.add_argument('--evaluations', type=str, nargs='+',
                        default=['hallucination_horizon', 'noise_robustness', 'error_vs_uncertainty'],
                        choices=['hallucination_horizon', 'noise_robustness', 'error_vs_uncertainty'],
                        help='Which evaluations to run (default: all)')
    parser.add_argument('--horizon', type=int, default=200,
                        help='Prediction horizon for evaluations (default: 200)')
    parser.add_argument('--noise_levels', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.4, 0.5, 0.8],
                        help='Noise levels for noise robustness evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without executing')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)
    
    # Discover trajectories and checkpoints
    print("Discovering trajectories...")
    trajectories = discover_trajectories(trajectory_dir)
    print(f"  Found {sum(len(seeds) for seeds in trajectories.values())} trajectory sets in {len(trajectories)} conditions")
    
    print("\nDiscovering checkpoints...")
    checkpoints = discover_checkpoints(log_dir, args.filter)
    print(f"  Found {sum(len(seeds) for seeds in checkpoints.values())} runs in {len(checkpoints)} conditions")
    
    # Match trajectories to checkpoints
    print("\nMatching trajectories to checkpoints...")
    matches = match_trajectories_to_checkpoints(trajectories, checkpoints)
    print(f"  Found {len(matches)} matched pairs")
    
    if not matches:
        print("\nNo matched trajectory/checkpoint pairs found!")
        print("Make sure trajectory files are organized as: trajectory_dir/<condition>/seed<N>_step<M>.pt")
        return 1
    
    # Show what will be evaluated
    print(f"\nEvaluations to run: {args.evaluations}")
    print(f"Prediction horizon: {args.horizon}")
    if 'noise_robustness' in args.evaluations:
        print(f"Noise levels: {args.noise_levels}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE]")
    
    # Run evaluations
    run_evaluations(
        matches,
        output_dir,
        args.evaluations,
        horizon=args.horizon,
        noise_levels=args.noise_levels,
        device=args.device,
        dry_run=args.dry_run,
    )
    
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN: Would evaluate {len(matches)} checkpoint/trajectory pairs")
    else:
        print("Evaluation complete!")
        print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
