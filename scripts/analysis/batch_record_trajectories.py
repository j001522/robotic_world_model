#!/usr/bin/env python3
"""
Batch record trajectories from all trained policies.

This script discovers all finetune runs and records trajectories for each,
storing them in a structured directory that evaluation scripts can auto-discover.

Usage:
    # Record trajectories for all finetune runs
    python batch_record_trajectories.py \
        --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
        --log_dir ../../logs/rsl_rl/anymal_d_flat \
        --output_dir ../../results/paper/trajectories

    # Record only specific checkpoint steps
    python batch_record_trajectories.py \
        --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
        --log_dir ../../logs/rsl_rl/anymal_d_flat \
        --output_dir ../../results/paper/trajectories \
        --checkpoint_steps 5000

    # Dry run to see what would be processed
    python batch_record_trajectories.py \
        --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
        --log_dir ../../logs/rsl_rl/anymal_d_flat \
        --output_dir ../../results/paper/trajectories \
        --dry_run

Output structure:
    results/paper/trajectories/
    ├── finetune-bs-nopen/
    │   ├── seed42_step5000.pt
    │   ├── seed43_step5000.pt
    │   └── seed44_step5000.pt
    ├── finetune-rp-pen025-std/
    │   ├── seed42_step5000.pt
    │   └── ...
    └── ...
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Import shared utilities (no Isaac Sim dependency)
from utils import parse_run_name, find_checkpoints


def discover_runs(log_dir: Path, filter_pattern: str = "finetune"):
    """Discover all runs matching the filter pattern."""
    runs = []
    for run_dir in sorted(log_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if filter_pattern and filter_pattern not in run_dir.name:
            continue
        
        metadata = parse_run_name(str(run_dir))
        if metadata is None:
            print(f"  Skipping (could not parse): {run_dir.name}")
            continue
        
        runs.append((run_dir, metadata))
    
    return runs


def main():
    parser = argparse.ArgumentParser(
        description='Batch record trajectories from all trained policies.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--task', type=str, required=True,
                        help='Name of the task (e.g., Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0)')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory containing run subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for trajectory files')
    parser.add_argument('--filter', type=str, default='finetune',
                        help='Filter runs by pattern (default: "finetune")')
    parser.add_argument('--checkpoint_steps', type=int, nargs='+', default=None,
                        help='Specific checkpoint steps to record (default: last checkpoint)')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of trajectories per checkpoint (default: 100)')
    parser.add_argument('--trajectory_length', type=int, default=200,
                        help='Length of each trajectory (default: 200)')
    parser.add_argument('--num_envs', type=int, default=64,
                        help='Number of parallel environments (default: 64)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without executing')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip runs that already have trajectory files')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    
    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        return 1
    
    # Discover runs
    print(f"Discovering runs in: {log_dir}")
    print(f"Filter: {args.filter}")
    runs = discover_runs(log_dir, args.filter)
    
    if not runs:
        print("No runs found!")
        return 1
    
    print(f"\nFound {len(runs)} runs:")
    for run_dir, metadata in runs:
        print(f"  {metadata.condition_key} (seed {metadata.seed})")
    
    # Group by condition
    conditions = {}
    for run_dir, metadata in runs:
        if metadata.condition_key not in conditions:
            conditions[metadata.condition_key] = []
        conditions[metadata.condition_key].append((run_dir, metadata))
    
    print(f"\nConditions: {list(conditions.keys())}")
    
    # Process each run
    total_jobs = 0
    failed_jobs = []
    
    for condition, condition_runs in conditions.items():
        condition_output_dir = output_dir / condition
        
        if not args.dry_run:
            condition_output_dir.mkdir(parents=True, exist_ok=True)
        
        for run_dir, metadata in condition_runs:
            # Find checkpoints
            checkpoints = find_checkpoints(str(run_dir), args.checkpoint_steps)
            
            if not checkpoints:
                print(f"\n  WARNING: No checkpoints found in {run_dir.name}")
                continue
            
            # Use last checkpoint if no specific steps requested
            if args.checkpoint_steps is None:
                checkpoints = [checkpoints[-1]]
            
            for step, ckpt_path in checkpoints:
                output_file = condition_output_dir / f"seed{metadata.seed}_step{step}.pt"
                
                # Skip if exists
                if args.skip_existing and output_file.exists():
                    print(f"\n  Skipping (exists): {output_file}")
                    continue
                
                print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Recording: {condition} seed{metadata.seed} step{step}")
                print(f"  Checkpoint: {ckpt_path}")
                print(f"  Output: {output_file}")
                
                if args.dry_run:
                    total_jobs += 1
                    continue
                
                # Build command - use absolute paths since subprocess runs from scripts/analysis/
                cmd = [
                    sys.executable, 'record_trajectories.py',
                    '--task', args.task,
                    '--checkpoint', str(Path(ckpt_path).resolve()),
                    '--output', str(output_file.resolve()),
                    '--num_trajectories', str(args.num_trajectories),
                    '--trajectory_length', str(args.trajectory_length),
                    '--num_envs', str(args.num_envs),
                    '--seed', str(metadata.seed),
                ]
                
                # Run the recording script
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, cwd=Path(__file__).parent)
                
                if result.returncode != 0:
                    print(f"  ERROR: Recording failed!")
                    failed_jobs.append((condition, metadata.seed, step))
                else:
                    print(f"  SUCCESS: Saved to {output_file}")
                
                total_jobs += 1
    
    # Summary
    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"DRY RUN: Would process {total_jobs} checkpoints")
    else:
        print(f"Processed {total_jobs} checkpoints")
        if failed_jobs:
            print(f"Failed: {len(failed_jobs)}")
            for condition, seed, step in failed_jobs:
                print(f"  - {condition} seed{seed} step{step}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    return 0 if not failed_jobs else 1


if __name__ == '__main__':
    sys.exit(main())
