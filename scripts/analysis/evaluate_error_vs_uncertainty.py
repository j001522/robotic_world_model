#!/usr/bin/env python3
"""
Evaluate Error vs Epistemic Uncertainty during Imagination Rollouts

This script evaluates the relationship between prediction error and epistemic uncertainty
during open-loop imagination rollouts. It collects both metrics at each prediction step
to analyze how uncertainty correlates with prediction accuracy.

The epistemic uncertainty is computed from the ensemble variance (std across ensemble members).

Outputs:
- CSV with columns: condition, seed, checkpoint_step, horizon_t, error_rel, error_abs, epistemic_uncertainty

Usage:
    python evaluate_error_vs_uncertainty.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/2026-02-03_run/model_5000.pt \
        --trajectory_data trajectories.pt \
        --horizon 200 \
        --output_dir results/error_vs_uncertainty/
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
    RunMetadata,
    parse_run_name,
    find_checkpoints,
    load_checkpoint_and_config,
    infer_dimensions_from_checkpoint,
    ensure_rsl_rl_rwm_in_path,
    STATE_COMPONENTS,
    VELOCITY_INDICES,
)


def evaluate_error_vs_uncertainty(
    checkpoint_path: str,
    trajectory_data_path: str,
    horizon: int = 200,
    device: str = "cuda",
    velocity_only: bool = False,
) -> pd.DataFrame:
    """
    Evaluate prediction error and epistemic uncertainty during imagination rollouts.
    
    Args:
        checkpoint_path: Path to model checkpoint
        trajectory_data_path: Path to .pt file with recorded trajectories
        horizon: Maximum prediction horizon
        device: Device to run evaluation on
        velocity_only: If True, compute errors only on velocity components (v, w, q_dot)
    
    Returns:
        DataFrame with columns: horizon_t, error_rel_mean, error_rel_std, error_abs_mean, 
                               error_abs_std, uncertainty_mean, uncertainty_std
    """
    # Convert device string to torch.device, fallback to CPU if CUDA unavailable
    device_str: str = str(device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device_obj = torch.device(device_str)
    
    # Load trajectory data
    traj_data = torch.load(trajectory_data_path, map_location=device_obj)
    states = traj_data['states']  # [N, T, state_dim]
    actions = traj_data['actions']  # [N, T, action_dim]
    
    N, T, state_dim = states.shape
    _, _, action_dim = actions.shape
    
    print(f"  Loaded {N} trajectories of length {T}")
    
    # Load checkpoint
    checkpoint, agent_cfg, run_dir = load_checkpoint_and_config(checkpoint_path, device_str)
    sd_cfg = agent_cfg.get('system_dynamics', {})
    imagination_cfg = agent_cfg.get('imagination', {})
    
    # Reconstruct system dynamics using shared utility
    ensure_rsl_rl_rwm_in_path()
    from rsl_rl.modules import SystemDynamicsEnsemble, EmpiricalNormalization
    
    # Infer contact_dim and termination_dim using shared utility
    contact_dim, termination_dim = infer_dimensions_from_checkpoint(checkpoint)
    
    # Create system dynamics
    system_dynamics = SystemDynamicsEnsemble(
        state_dim=state_dim,
        action_dim=action_dim,
        extension_dim=0,
        contact_dim=contact_dim,
        termination_dim=termination_dim,
        device=device_obj,
        **sd_cfg
    )
    system_dynamics.load_state_dict(checkpoint['system_dynamics_state_dict'])
    system_dynamics.eval()
    
    # Create normalizers
    state_norm_cfg = imagination_cfg.get('state_normalizer', {})
    action_norm_cfg = imagination_cfg.get('action_normalizer', {})
    
    state_normalizer = EmpiricalNormalization(shape=[state_dim], until=1e8).to(device_obj).eval()
    action_normalizer = EmpiricalNormalization(shape=[action_dim], until=1e8).to(device_obj).eval()
    
    if 'mean' in state_norm_cfg and 'std' in state_norm_cfg:
        state_normalizer.load_state_dict({
            '_mean': torch.tensor(state_norm_cfg['mean'], device=device_obj).unsqueeze(0),
            '_std': torch.tensor(state_norm_cfg['std'], device=device_obj).unsqueeze(0),
            '_var': torch.square(torch.tensor(state_norm_cfg['std'], device=device_obj).unsqueeze(0)),
            'count': torch.tensor(0, dtype=torch.long, device=device_obj),
        })
    if 'mean' in action_norm_cfg and 'std' in action_norm_cfg:
        action_normalizer.load_state_dict({
            '_mean': torch.tensor(action_norm_cfg['mean'], device=device_obj).unsqueeze(0),
            '_std': torch.tensor(action_norm_cfg['std'], device=device_obj).unsqueeze(0),
            '_var': torch.square(torch.tensor(action_norm_cfg['std'], device=device_obj).unsqueeze(0)),
            'count': torch.tensor(0, dtype=torch.long, device=device_obj),
        })
    
    # Normalize data
    states_norm = state_normalizer(states.reshape(-1, state_dim)).reshape(N, T, state_dim)
    actions_norm = action_normalizer(actions.reshape(-1, action_dim)).reshape(N, T, action_dim)
    
    history_horizon = sd_cfg.get('history_horizon', 1)
    eval_horizon = min(horizon, T - history_horizon)
    
    # Determine which state indices to use for error computation
    if velocity_only:
        if state_dim >= 33:
            eval_indices = VELOCITY_INDICES
            print(f"  Using velocity-only indices: v(0-2), w(3-5), q_dot(21-32) = {len(eval_indices)} dims")
        else:
            print(f"  WARNING: state_dim={state_dim} < 33, using all indices")
            eval_indices = list(range(state_dim))
    else:
        eval_indices = list(range(state_dim))
    
    eval_indices_tensor = torch.tensor(eval_indices, device=device_obj)
    
    # Run open-loop prediction and collect error + uncertainty
    per_step_errors = []
    per_step_errors_abs = []
    per_step_uncertainties = []
    
    # Get the uncertainty metric from config (std or variance)
    uncertainty_metric = sd_cfg.get('uncertainty_metric', 'std')
    print(f"  Using uncertainty metric: {uncertainty_metric} (from config)")
    
    with torch.inference_mode():
        system_dynamics.reset()
        
        # Initialize with history
        state_pred = states_norm[:, :history_horizon].clone()
        
        for t in range(history_horizon, history_horizon + eval_horizon):
            # Get input window
            if sd_cfg.get('architecture_config', {}).get('type') in ['rnn', 'rssm'] and t > history_horizon:
                state_input = state_pred[:, -1:]
                action_input = actions_norm[:, t-1:t]
            else:
                state_input = state_pred[:, -history_horizon:]
                action_input = actions_norm[:, t-history_horizon:t]
            
            # SystemDynamicsEnsemble.forward returns: 
            # next_state, aleatoric_uncertainty, epistemic_uncertainty, extensions, contacts, terminations
            # The epistemic_uncertainty is already computed using the correct metric (std or variance)
            # based on the model's configuration
            next_state_mean, aleatoric_uncertainty, epistemic_uncertainty, _, _, _ = system_dynamics.forward(state_input, action_input)
            
            # Compute error vs ground truth
            gt_state = states_norm[:, t]
            
            # Extract relevant indices for error computation
            pred_selected = next_state_mean[:, eval_indices_tensor]
            gt_selected = gt_state[:, eval_indices_tensor]
            
            # Relative error (same as training metric)
            rel_error = (pred_selected - gt_selected).abs().sum(dim=-1) / (gt_selected.abs().sum(dim=-1) + 1e-8)
            
            # Absolute error (MSE)
            abs_error = (pred_selected - gt_selected).pow(2).sum(dim=-1)
            
            per_step_errors.append(rel_error.cpu().numpy())
            per_step_errors_abs.append(abs_error.cpu().numpy())
            per_step_uncertainties.append(epistemic_uncertainty.cpu().numpy())
            
            # Append prediction for next step
            state_pred = torch.cat([state_pred, next_state_mean.unsqueeze(1)], dim=1)
    
    # Convert to DataFrame
    per_step_errors = np.array(per_step_errors)  # [horizon, N]
    per_step_errors_abs = np.array(per_step_errors_abs)
    per_step_uncertainties = np.array(per_step_uncertainties)
    
    results = []
    for t in range(eval_horizon):
        results.append({
            'horizon_t': t + history_horizon + 1,  # Actual timestep in trajectory
            'prediction_step': t + 1,  # Step number in the open-loop rollout
            'error_rel_mean': per_step_errors[t].mean(),
            'error_rel_std': per_step_errors[t].std(),
            'error_abs_mean': per_step_errors_abs[t].mean(),
            'error_abs_std': per_step_errors_abs[t].std(),
            'uncertainty_mean': per_step_uncertainties[t].mean(),
            'uncertainty_std': per_step_uncertainties[t].std(),
            'history_horizon': history_horizon,
            'uncertainty_metric': uncertainty_metric,  # Track which metric was used (std or variance)
        })
    
    return pd.DataFrame(results)


def aggregate_results(
    all_results: list[tuple[str, int, int, pd.DataFrame]],
    output_dir: Path,
):
    """
    Aggregate results across seeds for each condition.
    
    Args:
        all_results: List of (condition, seed, checkpoint_step, DataFrame)
        output_dir: Where to save aggregated results
    """
    # Group by condition and checkpoint step
    grouped = defaultdict(lambda: defaultdict(list))
    for condition, seed, step, df in all_results:
        grouped[condition][step].append((seed, df))
    
    # Aggregate each condition
    for condition, step_data in grouped.items():
        agg_records = []
        
        for step, seed_dfs in step_data.items():
            # Combine all seeds for this step
            combined = {}
            for seed, df in seed_dfs:
                for _, row in df.iterrows():
                    t = row['horizon_t']
                    if t not in combined:
                        combined[t] = {
                            'error_rel': [], 'error_abs': [], 'uncertainty': [],
                            'history_horizon': row['history_horizon']
                        }
                    combined[t]['error_rel'].append(row['error_rel_mean'])
                    combined[t]['error_abs'].append(row['error_abs_mean'])
                    combined[t]['uncertainty'].append(row['uncertainty_mean'])
            
            # Get uncertainty_metric from first seed's dataframe
            uncertainty_metric = seed_dfs[0][1]['uncertainty_metric'].iloc[0] if 'uncertainty_metric' in seed_dfs[0][1].columns else 'std'
            
            # Compute mean/std across seeds
            for t, values in combined.items():
                rel_errors = np.array(values['error_rel'])
                abs_errors = np.array(values['error_abs'])
                uncertainties = np.array(values['uncertainty'])
                agg_records.append({
                    'condition': condition,
                    'checkpoint_step': step,
                    'horizon_t': t,
                    'prediction_step': t - values['history_horizon'],
                    'error_rel_mean': rel_errors.mean(),
                    'error_rel_std': rel_errors.std(),
                    'error_abs_mean': abs_errors.mean(),
                    'error_abs_std': abs_errors.std(),
                    'uncertainty_mean': uncertainties.mean(),
                    'uncertainty_std': uncertainties.std(),
                    'n_seeds': len(rel_errors),
                    'history_horizon': values['history_horizon'],
                    'uncertainty_metric': uncertainty_metric,
                })
        
        # Save condition-specific file
        agg_df = pd.DataFrame(agg_records)
        output_path = output_dir / f"error_vs_uncertainty_{condition}.csv"
        agg_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    # Also save combined file (append if exists)
    all_records = []
    for condition, seed, step, df in all_results:
        for _, row in df.iterrows():
            record = row.to_dict()
            record['condition'] = condition
            record['seed'] = seed
            record['checkpoint_step'] = step
            all_records.append(record)
    
    combined_df = pd.DataFrame(all_records)
    output_path = output_dir / "error_vs_uncertainty_all.csv"
    
    # Append to existing file if it exists
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        # Remove duplicates based on condition, seed, checkpoint_step, horizon_t
        combined_df = combined_df.drop_duplicates(
            subset=['condition', 'seed', 'checkpoint_step', 'horizon_t'],
            keep='last'
        )
    
    combined_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate prediction error vs epistemic uncertainty during imagination rollouts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--checkpoint', type=str, help='Single checkpoint to evaluate')
    input_group.add_argument('--log_dir', type=str, help='Directory containing multiple runs')
    
    parser.add_argument('--filter', type=str, default='finetune',
                        help='Filter runs by pattern (default: "finetune"). Use --filter "" for no filtering.')
    parser.add_argument('--checkpoint_steps', type=int, nargs='+', default=None,
                        help='Specific checkpoint steps to evaluate (default: last)')
    
    # Trajectory options
    parser.add_argument('--trajectory_data', type=str, default=None,
                        help='Path to pre-recorded trajectory data (.pt file)')
    parser.add_argument('--horizon', type=int, default=200,
                        help='Maximum prediction horizon (default: 200)')
    
    # Velocity filtering options
    parser.add_argument('--velocity_only', action='store_true',
                        help='Compute errors only on velocity components (v, w, q_dot)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='results/paper/error_vs_uncertainty',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.trajectory_data is None:
        print("ERROR: --trajectory_data is required for offline evaluation.")
        print("To record trajectory data, run:")
        print("  python scripts/analysis/record_trajectories.py --checkpoint <path> --output trajectories.pt")
        return
    
    if args.velocity_only:
        print("Velocity-only mode: computing errors on v (linear vel), w (angular vel), q_dot (joint vel)")
    
    all_results = []
    
    if args.checkpoint:
        # Single checkpoint evaluation
        print(f"Evaluating single checkpoint: {args.checkpoint}")
        
        run_dir = Path(args.checkpoint).parent
        metadata = parse_run_name(str(run_dir))
        
        if metadata is None:
            condition = "unknown"
            seed = 0
        else:
            condition = metadata.condition_key
            seed = metadata.seed
        
        step_match = re.search(r'model_(\d+)\.pt', args.checkpoint)
        step = int(step_match.group(1)) if step_match else 0
        
        df = evaluate_error_vs_uncertainty(
            args.checkpoint,
            args.trajectory_data,
            horizon=args.horizon,
            device=args.device,
            velocity_only=args.velocity_only,
        )
        
        if df is not None:
            all_results.append((condition, seed, step, df))
    
    else:
        # Multiple runs evaluation
        log_path = Path(args.log_dir)
        run_dirs = sorted([d for d in log_path.iterdir() if d.is_dir()])
        
        print(f"Found {len(run_dirs)} run directories")
        
        for run_dir in run_dirs:
            if args.filter and args.filter not in run_dir.name:
                continue
            
            metadata = parse_run_name(str(run_dir))
            if metadata is None:
                print(f"  Skipping (could not parse): {run_dir.name}")
                continue
            
            print(f"\nProcessing: {run_dir.name}")
            print(f"  Condition: {metadata.condition_key}, Seed: {metadata.seed}")
            
            checkpoints = find_checkpoints(str(run_dir), args.checkpoint_steps)
            
            if not checkpoints:
                print(f"  No checkpoints found")
                continue
            
            # Use last checkpoint if no specific steps requested
            if args.checkpoint_steps is None:
                checkpoints = [checkpoints[-1]]
            
            for step, ckpt_path in checkpoints:
                print(f"  Evaluating checkpoint step {step}")
                
                df = evaluate_error_vs_uncertainty(
                    ckpt_path,
                    args.trajectory_data,
                    horizon=args.horizon,
                    device=args.device,
                    velocity_only=args.velocity_only,
                )
                
                if df is not None:
                    all_results.append((metadata.condition_key, metadata.seed, step, df))
    
    if all_results:
        print("\nAggregating results...")
        aggregate_results(all_results, output_dir)
        print("\nDone!")
    else:
        print("\nNo results to aggregate.")


if __name__ == '__main__':
    main()
