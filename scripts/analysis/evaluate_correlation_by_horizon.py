#!/usr/bin/env python3
"""
Evaluate Correlation Between Prediction Error and Epistemic Uncertainty by Prediction Step

This script computes the correlation between prediction error and epistemic uncertainty
at each prediction step during open-loop imagination rollouts. This helps analyze how
well the model's uncertainty predicts prediction accuracy as we roll out further into
the future.

Outputs:
- CSV with columns: condition, seed, checkpoint_step, prediction_step, 
                    pearson_r, pearson_p, spearman_r, spearman_p, n_samples

Usage:
    python evaluate_correlation_by_horizon.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/2026-02-03_run/model_5000.pt \
        --trajectory_data trajectories.pt \
        --horizon 200 \
        --output_dir results/correlation_by_horizon/
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
from scipy import stats

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


def evaluate_correlation_by_horizon(
    checkpoint_path: str,
    trajectory_data_path: str,
    horizon: int = 200,
    device: str = "cuda",
    velocity_only: bool = False,
) -> pd.DataFrame:
    """
    Evaluate correlation between prediction error and epistemic uncertainty at each prediction step.
    
    Args:
        checkpoint_path: Path to model checkpoint
        trajectory_data_path: Path to .pt file with recorded trajectories
        horizon: Maximum prediction horizon
        device: Device to run evaluation on
        velocity_only: If True, compute errors only on velocity components
    
    Returns:
        DataFrame with columns: prediction_step, pearson_r, pearson_p, spearman_r, spearman_p, n_samples
    """
    device_str: str = str(device)
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device_obj = torch.device(device_str)
    
    traj_data = torch.load(trajectory_data_path, map_location=device_obj)
    states = traj_data['states']
    actions = traj_data['actions']
    
    N, T, state_dim = states.shape
    _, _, action_dim = actions.shape
    
    print(f"  Loaded {N} trajectories of length {T}")
    
    checkpoint, agent_cfg, run_dir = load_checkpoint_and_config(checkpoint_path, device_str)
    sd_cfg = agent_cfg.get('system_dynamics', {})
    imagination_cfg = agent_cfg.get('imagination', {})
    
    ensure_rsl_rl_rwm_in_path()
    from rsl_rl.modules import SystemDynamicsEnsemble, EmpiricalNormalization
    
    contact_dim, termination_dim = infer_dimensions_from_checkpoint(checkpoint)
    
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
    
    states_norm = state_normalizer(states.reshape(-1, state_dim)).reshape(N, T, state_dim)
    actions_norm = action_normalizer(actions.reshape(-1, action_dim)).reshape(N, T, action_dim)
    
    history_horizon = sd_cfg.get('history_horizon', 1)
    eval_horizon = min(horizon, T - history_horizon)
    
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
    
    uncertainty_metric = sd_cfg.get('uncertainty_metric', 'std')
    print(f"  Using uncertainty metric: {uncertainty_metric} (from config)")
    
    with torch.inference_mode():
        system_dynamics.reset()
        state_pred = states_norm[:, :history_horizon].clone()
        
        per_step_errors = []
        per_step_uncertainties = []
        
        for t in range(history_horizon, history_horizon + eval_horizon):
            if sd_cfg.get('architecture_config', {}).get('type') in ['rnn', 'rssm'] and t > history_horizon:
                state_input = state_pred[:, -1:]
                action_input = actions_norm[:, t-1:t]
            else:
                state_input = state_pred[:, -history_horizon:]
                action_input = actions_norm[:, t-history_horizon:t]
            
            next_state_mean, aleatoric_uncertainty, epistemic_uncertainty, _, _, _ = system_dynamics.forward(state_input, action_input)
            
            gt_state = states_norm[:, t]
            
            pred_selected = next_state_mean[:, eval_indices_tensor]
            gt_selected = gt_state[:, eval_indices_tensor]
            
            rel_error = (pred_selected - gt_selected).abs().sum(dim=-1) / (gt_selected.abs().sum(dim=-1) + 1e-8)
            
            per_step_errors.append(rel_error.cpu().numpy())
            per_step_uncertainties.append(epistemic_uncertainty.cpu().numpy())
            
            state_pred = torch.cat([state_pred, next_state_mean.unsqueeze(1)], dim=1)
    
    per_step_errors = np.array(per_step_errors)
    per_step_uncertainties = np.array(per_step_uncertainties)
    
    results = []
    for t in range(eval_horizon):
        errors = per_step_errors[t]
        uncertainties = per_step_uncertainties[t]
        
        pearson_r, pearson_p = stats.pearsonr(errors, uncertainties)
        spearman_r, spearman_p = stats.spearmanr(errors, uncertainties)
        
        results.append({
            'prediction_step': t + 1,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(errors),
            'history_horizon': history_horizon,
            'uncertainty_metric': uncertainty_metric,
        })
    
    return pd.DataFrame(results)


def aggregate_results(
    all_results: list[tuple[str, int, int, pd.DataFrame]],
    output_dir: Path,
):
    grouped = defaultdict(lambda: defaultdict(list))
    for condition, seed, step, df in all_results:
        grouped[condition][step].append((seed, df))
    
    for condition, step_data in grouped.items():
        agg_records = []
        
        for step, seed_dfs in step_data.items():
            combined = {}
            for seed, df in seed_dfs:
                for _, row in df.iterrows():
                    pred_step = row['prediction_step']
                    if pred_step not in combined:
                        combined[pred_step] = {
                            'pearson_r': [], 'pearson_p': [], 'spearman_r': [], 'spearman_p': [],
                            'n_samples': [], 'history_horizon': row['history_horizon']
                        }
                    combined[pred_step]['pearson_r'].append(row['pearson_r'])
                    combined[pred_step]['pearson_p'].append(row['pearson_p'])
                    combined[pred_step]['spearman_r'].append(row['spearman_r'])
                    combined[pred_step]['spearman_p'].append(row['spearman_p'])
                    combined[pred_step]['n_samples'].append(row['n_samples'])
            
            uncertainty_metric = seed_dfs[0][1]['uncertainty_metric'].iloc[0]
            
            for pred_step, values in combined.items():
                agg_records.append({
                    'condition': condition,
                    'checkpoint_step': step,
                    'prediction_step': pred_step,
                    'pearson_r_mean': np.mean(values['pearson_r']),
                    'pearson_r_std': np.std(values['pearson_r']),
                    'pearson_p_mean': np.mean(values['pearson_p']),
                    'spearman_r_mean': np.mean(values['spearman_r']),
                    'spearman_r_std': np.std(values['spearman_r']),
                    'spearman_p_mean': np.mean(values['spearman_p']),
                    'n_seeds': len(values['pearson_r']),
                    'n_samples_per_seed': np.mean(values['n_samples']),
                    'history_horizon': values['history_horizon'],
                    'uncertainty_metric': uncertainty_metric,
                })
        
        agg_df = pd.DataFrame(agg_records)
        output_path = output_dir / f"correlation_by_horizon_{condition}.csv"
        agg_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    all_records = []
    for condition, seed, step, df in all_results:
        for _, row in df.iterrows():
            record = row.to_dict()
            record['condition'] = condition
            record['seed'] = seed
            record['checkpoint_step'] = step
            all_records.append(record)
    
    combined_df = pd.DataFrame(all_records)
    output_path = output_dir / "correlation_by_horizon_all.csv"
    
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=['condition', 'seed', 'checkpoint_step', 'prediction_step'],
            keep='last'
        )
    
    combined_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate correlation between prediction error and epistemic uncertainty by prediction step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--checkpoint', type=str, help='Single checkpoint to evaluate')
    input_group.add_argument('--log_dir', type=str, help='Directory containing multiple runs')
    
    parser.add_argument('--filter', type=str, default='finetune',
                        help='Filter runs by pattern (default: "finetune")')
    parser.add_argument('--checkpoint_steps', type=int, nargs='+', default=None,
                        help='Specific checkpoint steps to evaluate (default: last)')
    
    parser.add_argument('--trajectory_data', type=str, default=None,
                        help='Path to pre-recorded trajectory data (.pt file)')
    parser.add_argument('--horizon', type=int, default=200,
                        help='Maximum prediction horizon (default: 200)')
    
    parser.add_argument('--velocity_only', action='store_true',
                        help='Compute errors only on velocity components')
    
    parser.add_argument('--output_dir', type=str, default='results/paper/correlation_by_horizon',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.trajectory_data is None:
        print("ERROR: --trajectory_data is required for offline evaluation.")
        return
    
    all_results = []
    
    if args.checkpoint:
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
        
        df = evaluate_correlation_by_horizon(
            args.checkpoint,
            args.trajectory_data,
            horizon=args.horizon,
            device=args.device,
            velocity_only=args.velocity_only,
        )
        
        if df is not None:
            all_results.append((condition, seed, step, df))
    else:
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
            
            if args.checkpoint_steps is None:
                checkpoints = [checkpoints[-1]]
            
            for step, ckpt_path in checkpoints:
                print(f"  Evaluating checkpoint step {step}")
                
                df = evaluate_correlation_by_horizon(
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
