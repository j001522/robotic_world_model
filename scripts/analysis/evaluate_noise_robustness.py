#!/usr/bin/env python3
"""
Evaluate C: Robustness to Noise (OOD) - Corrected evaluation.

The original `autoregressive_error_noised` metric in training has a bug:
it compares predictions to NOISED targets, not CLEAN targets. This script
provides corrected evaluation that:
1. Adds noise to INPUTS only
2. Compares predictions to CLEAN ground truth

This measures true OOD robustness: how well does the model predict when
given out-of-distribution (noisy) inputs?

Outputs CSVs to the specified output directory. Use plot_noise_robustness.py
to create visualizations from the CSV data.

Usage:
    # Evaluate from saved trajectory data (offline)
    python evaluate_noise_robustness.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/run/model_5000.pt \
        --trajectory_data trajectories.pt \
        --output_dir results/noise_robustness/
    
    # Evaluate multiple runs
    python evaluate_noise_robustness.py \
        --log_dir logs/rsl_rl/anymal_d_flat \
        --filter "finetune" \
        --trajectory_data trajectories.pt \
        --output_dir results/noise_robustness/

    # Extract data from TensorBoard exports (with caveats about the bug)
    python evaluate_noise_robustness.py \
        --from_tensorboard results/tensorboard_export/aggregated \
        --output_dir results/noise_robustness/
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import pandas as pd

# Import shared utilities
from utils import (
    RunMetadata,
    parse_run_name,
    load_checkpoint_and_config,
    infer_dimensions_from_checkpoint,
    ensure_rsl_rl_rwm_in_path,
)


def evaluate_noise_robustness_offline(
    checkpoint_path: str,
    trajectory_data_path: str,
    noise_levels: list = [0.0, 0.1, 0.2, 0.4, 0.5, 0.8],
    horizon: int = 50,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Evaluate noise robustness using pre-recorded trajectory data.
    
    CORRECTED IMPLEMENTATION:
    - Adds noise to INPUTS only (states and actions fed to model)
    - Compares predictions to CLEAN ground truth
    
    Args:
        checkpoint_path: Path to model checkpoint
        trajectory_data_path: Path to .pt file with recorded trajectories
        noise_levels: List of noise scales to evaluate
        horizon: Prediction horizon for evaluation
        device: Device to run evaluation on
    
    Returns:
        DataFrame with error statistics per noise level
    """
    # Convert device string to torch.device, fallback to CPU if CUDA unavailable
    device_str = device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device_str = "cpu"
    device_obj = torch.device(device_str)
    
    # Load trajectory data
    traj_data = torch.load(trajectory_data_path, map_location=device_obj, weights_only=False)
    states = traj_data['states']  # [N, T, state_dim]
    actions = traj_data['actions']  # [N, T, action_dim]
    
    N, T, state_dim = states.shape
    _, _, action_dim = actions.shape
    
    print(f"  Loaded {N} trajectories of length {T}, state_dim={state_dim}, action_dim={action_dim}")
    
    # Load checkpoint and config using shared utility
    checkpoint, agent_cfg, run_dir = load_checkpoint_and_config(checkpoint_path, device_str)
    
    sd_cfg = agent_cfg.get('system_dynamics', {})
    imagination_cfg = agent_cfg.get('imagination', {})
    
    # Import modules using shared utility
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
            'count': torch.tensor(0, dtype=torch.long),
        })
    if 'mean' in action_norm_cfg and 'std' in action_norm_cfg:
        action_normalizer.load_state_dict({
            '_mean': torch.tensor(action_norm_cfg['mean'], device=device_obj).unsqueeze(0),
            '_std': torch.tensor(action_norm_cfg['std'], device=device_obj).unsqueeze(0),
            '_var': torch.square(torch.tensor(action_norm_cfg['std'], device=device_obj).unsqueeze(0)),
            'count': torch.tensor(0, dtype=torch.long),
        })
    
    # Normalize CLEAN data
    states_norm = state_normalizer(states.reshape(-1, state_dim)).reshape(N, T, state_dim)
    actions_norm = action_normalizer(actions.reshape(-1, action_dim)).reshape(N, T, action_dim)
    
    history_horizon = sd_cfg.get('history_horizon', 1)
    eval_horizon = min(horizon, T - history_horizon)
    
    results = []
    
    for noise_scale in noise_levels:
        print(f"    Evaluating noise level: {noise_scale}")
        
        # Add noise to inputs ONLY
        if noise_scale > 0:
            states_noised = states_norm + torch.randn_like(states_norm) * noise_scale
            actions_noised = actions_norm + torch.randn_like(actions_norm) * noise_scale
        else:
            states_noised = states_norm
            actions_noised = actions_norm
        
        # Run open-loop prediction with noised inputs
        all_errors = []
        
        with torch.inference_mode():
            system_dynamics.reset()
            
            # Initialize with noised history
            state_pred = states_noised[:, :history_horizon].clone()
            
            for t in range(history_horizon, history_horizon + eval_horizon):
                # Get input window (from noised predictions)
                if sd_cfg.get('architecture_config', {}).get('type') in ['rnn', 'rssm'] and t > history_horizon:
                    state_input = state_pred[:, -1:]
                    action_input = actions_noised[:, t-1:t]
                else:
                    state_input = state_pred[:, -history_horizon:]
                    action_input = actions_noised[:, t-history_horizon:t]
                
                # Predict next state
                next_state, _, _, _, _, _ = system_dynamics.forward(state_input, action_input)
                
                # Compare to CLEAN ground truth (the key fix!)
                gt_state = states_norm[:, t]  # CLEAN, not noised
                
                # Relative error
                rel_error = (next_state - gt_state).abs().sum(dim=-1) / (gt_state.abs().sum(dim=-1) + 1e-8)
                all_errors.append(rel_error.cpu().numpy())
                
                # Append prediction for next step
                state_pred = torch.cat([state_pred, next_state.unsqueeze(1)], dim=1)
        
        # Aggregate errors
        all_errors = np.concatenate(all_errors)
        
        results.append({
            'noise_level': noise_scale,
            'error_mean': all_errors.mean(),
            'error_std': all_errors.std(),
            'error_min': all_errors.min(),
            'error_max': all_errors.max(),
            'error_median': np.median(all_errors),
        })
    
    return pd.DataFrame(results)


def load_tensorboard_noise_data(data_dir: str) -> dict:
    """
    Load noise robustness data from TensorBoard exports.
    
    NOTE: This data has the bug where predictions are compared to noised targets.
    Use with caution and note this limitation in any figures.
    """
    data_path = Path(data_dir)
    data = {}
    
    for csv_file in data_path.glob('*.csv'):
        condition = csv_file.stem
        df = pd.read_csv(csv_file)
        data[condition] = df
    
    return data


def get_noise_data_from_tensorboard(
    data: dict,
    step: Optional[int] = None,
    noise_levels: list = [0.0, 0.1, 0.2, 0.4, 0.5, 0.8],
) -> dict:
    """
    Extract noise vs error data from TensorBoard exports.
    
    Returns dict mapping condition -> DataFrame with noise_level and error columns.
    """
    result = {}
    
    for condition, df in data.items():
        # Find a valid step with noise data (exclude NaN and inf)
        noise_col = 'System_Dynamics_autoregressive_error_noised_0.1_mean'
        if noise_col not in df.columns:
            print(f"  {condition}: Noise column not found")
            continue
        valid_mask = df[noise_col].notna() & ~np.isinf(df[noise_col])
        
        if not valid_mask.any():
            print(f"  {condition}: No valid noise data found")
            continue
        
        valid_df = df[valid_mask]
        
        if step is not None:
            # Find closest step with valid data
            closest_idx = (valid_df['step'] - step).abs().argmin()
            row = valid_df.iloc[closest_idx]
        else:
            # Use last valid step
            row = valid_df.iloc[-1]
        
        actual_step = row['step']
        
        # Extract noise data
        records = []
        for noise in noise_levels:
            if noise == 0.0:
                mean_col = 'System_Dynamics_autoregressive_error_mean'
                std_col = 'System_Dynamics_autoregressive_error_std'
            else:
                mean_col = f'System_Dynamics_autoregressive_error_noised_{noise}_mean'
                std_col = f'System_Dynamics_autoregressive_error_noised_{noise}_std'
            
            if mean_col in row.index and pd.notna(row[mean_col]):
                records.append({
                    'noise_level': noise,
                    'error_mean': row[mean_col],
                    'error_std': row.get(std_col, 0) if pd.notna(row.get(std_col, np.nan)) else 0,
                })
        
        if records:
            result[condition] = {
                'data': pd.DataFrame(records),
                'step': actual_step,
            }
            print(f"  {condition}: Found data at step {actual_step}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate noise robustness (OOD prediction error) and output CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Input modes
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--from_tensorboard', type=str,
                             help='Load data from TensorBoard exports (with caveats)')
    input_group.add_argument('--checkpoint', type=str,
                             help='Single checkpoint for corrected evaluation')
    input_group.add_argument('--log_dir', type=str,
                             help='Directory with multiple runs for corrected evaluation')
    
    parser.add_argument('--trajectory_data', type=str, default=None,
                        help='Path to trajectory data (required for corrected evaluation)')
    parser.add_argument('--filter', type=str, default='finetune',
                        help='Filter runs by pattern (default: "finetune"). Use --filter "" for no filtering.')
    parser.add_argument('--step', type=int, default=None,
                        help='Training step for TensorBoard data')
    parser.add_argument('--noise_levels', type=float, nargs='+',
                        default=[0.0, 0.1, 0.2, 0.4, 0.5, 0.8],
                        help='Noise levels to evaluate')
    
    parser.add_argument('--output_dir', type=str, default='results/noise_robustness',
                        help='Output directory for CSVs')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.from_tensorboard:
        # Load from TensorBoard exports
        print(f"Loading TensorBoard data from: {args.from_tensorboard}")
        print("NOTE: This data uses the original (buggy) metric that compares to noised targets.")
        
        data = load_tensorboard_noise_data(args.from_tensorboard)
        noise_data = get_noise_data_from_tensorboard(data, step=args.step, noise_levels=args.noise_levels)
        
        if not noise_data:
            print("No valid noise data found!")
            return
        
        # Save data as CSV
        all_records = []
        for condition, info in noise_data.items():
            for _, row in info['data'].iterrows():
                all_records.append({
                    'condition': condition,
                    'step': info['step'],
                    **row.to_dict(),
                })
        
        df = pd.DataFrame(all_records)
        output_path = output_dir / "noise_robustness_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    else:
        # Corrected offline evaluation
        if args.trajectory_data is None:
            print("ERROR: --trajectory_data is required for corrected evaluation")
            return
        
        print("Running CORRECTED noise robustness evaluation")
        print("(Compares predictions to CLEAN targets)")
        
        all_results = []
        
        if args.checkpoint:
            # Single checkpoint
            run_dir = Path(args.checkpoint).parent
            metadata = parse_run_name(str(run_dir))
            condition = metadata.condition_key if metadata else "unknown"
            seed = metadata.seed if metadata else 0
            
            step_match = re.search(r'model_(\d+)\.pt', args.checkpoint)
            step = int(step_match.group(1)) if step_match else 0
            
            print(f"\nEvaluating: {args.checkpoint}")
            df = evaluate_noise_robustness_offline(
                args.checkpoint,
                args.trajectory_data,
                noise_levels=args.noise_levels,
                device=args.device,
            )
            
            for _, row in df.iterrows():
                all_results.append({
                    'condition': condition,
                    'seed': seed,
                    'checkpoint_step': step,
                    **row.to_dict(),
                })
        
        else:
            # Multiple runs
            log_path = Path(args.log_dir)
            run_dirs = sorted([d for d in log_path.iterdir() if d.is_dir()])
            
            for run_dir in run_dirs:
                if args.filter and args.filter not in run_dir.name:
                    continue
                
                metadata = parse_run_name(str(run_dir))
                if metadata is None:
                    continue
                
                # Find last checkpoint
                checkpoints = sorted(run_dir.glob("model_*.pt"))
                if not checkpoints:
                    continue
                
                ckpt_path = str(checkpoints[-1])
                step_match = re.search(r'model_(\d+)\.pt', ckpt_path)
                step = int(step_match.group(1)) if step_match else 0
                
                print(f"\nEvaluating: {run_dir.name}, step {step}")
                
                try:
                    df = evaluate_noise_robustness_offline(
                        ckpt_path,
                        args.trajectory_data,
                        noise_levels=args.noise_levels,
                        device=args.device,
                    )
                    
                    for _, row in df.iterrows():
                        all_results.append({
                            'condition': metadata.condition_key,
                            'seed': metadata.seed,
                            'checkpoint_step': step,
                            **row.to_dict(),
                        })
                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue
        
        if not all_results:
            print("No results!")
            return
        
        # Save all results (per-seed)
        results_df = pd.DataFrame(all_results)
        output_path = output_dir / "noise_robustness_all.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")
        
        # Aggregate by condition
        agg_records = []
        for (condition, noise_level), group in results_df.groupby(['condition', 'noise_level']):
            agg_records.append({
                'condition': condition,
                'noise_level': noise_level,
                'error_mean': group['error_mean'].mean(),
                'error_std': group['error_mean'].std() if len(group) > 1 else group['error_std'].values[0],
                'n_seeds': len(group),
            })
        
        agg_df = pd.DataFrame(agg_records)
        output_path = output_dir / "noise_robustness_aggregated.csv"
        agg_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    print("\nDone!")
    print(f"Use plot_noise_robustness.py --data_dir {output_dir} to create visualizations.")


if __name__ == '__main__':
    main()
