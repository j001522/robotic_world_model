#!/usr/bin/env python3
"""
Plot A: Hallucination Horizon (Dynamics Divergence)

Evaluates how prediction error accumulates over time during open-loop rollouts.
This measures the "hallucination horizon" - how far into the future the model
can predict before diverging significantly from reality.

For each checkpoint:
1. Load real trajectories from replay buffer (or generate from play.py)
2. Reset model to same initial state
3. Open-loop predict forward using same actions
4. Record error at each step t=1 to t=H

Outputs:
- CSV with columns: condition, seed, step, horizon_t, error
- Aggregated CSV with mean/std across seeds

State Structure (AnymalD system_state):
    0-2:   base_lin_vel (v) - Linear velocity
    3-5:   base_ang_vel (w) - Angular velocity  
    6-8:   projected_gravity
    9-20:  joint_pos (q) - Joint positions (12 joints)
    21-32: joint_vel (q_dot) - Joint velocities (12 joints)
    33-44: joint_torque - Joint torques (12 joints)

Usage:
    # Evaluate a single run
    python evaluate_hallucination_horizon.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/2026-02-03_run/model_5000.pt \
        --output_dir results/hallucination_horizon/

    # Evaluate with velocity-only errors (v, w, q_dot)
    python evaluate_hallucination_horizon.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/2026-02-03_run/model_5000.pt \
        --trajectory_data trajectories.pt \
        --velocity_only \
        --output_dir results/hallucination_horizon/

    # Evaluate multiple runs (grouped by condition)
    python evaluate_hallucination_horizon.py \
        --log_dir logs/rsl_rl/anymal_d_flat \
        --filter "finetune" \
        --output_dir results/hallucination_horizon/
        
    # Evaluate specific checkpoints from a run
    python evaluate_hallucination_horizon.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/2026-02-03_run/model_5000.pt \
        --trajectory_source buffer \
        --num_trajectories 100 \
        --horizon 50 \
        --output_dir results/hallucination_horizon/
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


def evaluate_hallucination_horizon_from_checkpoint(
    checkpoint_path: str,
    num_trajectories: int = 100,
    horizon: int = 50,
    device: str = "cuda",
) -> Optional[pd.DataFrame]:
    """
    Evaluate hallucination horizon for a single checkpoint.
    
    This is a standalone evaluation that:
    1. Loads the checkpoint
    2. Generates synthetic test trajectories (since we can't run Isaac Sim here)
    3. Computes per-step prediction errors
    
    For full evaluation with real trajectories, use the Isaac Sim based script.
    
    Returns:
        DataFrame with columns: horizon_t, error_mean, error_std, error_abs_mean, error_abs_std
        or None if evaluation cannot be performed
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    
    checkpoint, agent_cfg, run_dir = load_checkpoint_and_config(checkpoint_path, device)
    
    # Extract system dynamics config
    sd_cfg = agent_cfg.get('system_dynamics', {})
    imagination_cfg = agent_cfg.get('imagination', {})
    
    # Get dimensions from checkpoint
    sd_state_dict = checkpoint['system_dynamics_state_dict']
    
    # Try to infer dimensions from state dict keys
    # This is a heuristic - proper evaluation requires running in Isaac Sim
    print(f"  Checkpoint loaded. System dynamics config: ensemble_size={sd_cfg.get('ensemble_size', 'unknown')}")
    
    # For now, return placeholder - actual evaluation needs Isaac Sim
    # This function will be extended when running within Isaac Sim context
    print(f"  NOTE: Full evaluation requires Isaac Sim environment.")
    print(f"  Use --with_isaac_sim flag to run full evaluation.")
    
    return None


def evaluate_hallucination_horizon_offline(
    checkpoint_path: str,
    trajectory_data_path: str,
    horizon: int = 50,
    device: str = "cuda",
    velocity_only: bool = False,
    save_components: bool = False,
) -> pd.DataFrame:
    """
    Evaluate hallucination horizon using pre-recorded trajectory data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        trajectory_data_path: Path to .pt file with recorded trajectories
            Expected format: {
                'states': Tensor[N, T, state_dim],
                'actions': Tensor[N, T, action_dim],
            }
        horizon: Maximum prediction horizon
        device: Device to run evaluation on
        velocity_only: If True, compute errors only on velocity components (v, w, q_dot)
        save_components: If True, save per-component errors (v, w, q_dot separately)
    
    Returns:
        DataFrame with per-step error statistics
    """
    # Convert device string to torch.device, fallback to CPU if CUDA unavailable
    device_str = device
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
    
    # Reconstruct system dynamics
    # Import here to avoid circular imports when running standalone
    ensure_rsl_rl_rwm_in_path()
    from rsl_rl.modules import SystemDynamicsEnsemble, EmpiricalNormalization
    
    # Infer contact_dim and termination_dim from checkpoint
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
    
    # Normalize data
    states_norm = state_normalizer(states.reshape(-1, state_dim)).reshape(N, T, state_dim)
    actions_norm = action_normalizer(actions.reshape(-1, action_dim)).reshape(N, T, action_dim)
    
    history_horizon = sd_cfg.get('history_horizon', 1)
    eval_horizon = min(horizon, T - history_horizon)
    
    # Determine which state indices to use for error computation
    if velocity_only:
        # Use only velocity-related components: v (0-2), w (3-5), q_dot (21-32)
        # Adjust indices if state_dim doesn't match expected 45
        if state_dim >= 33:
            eval_indices = VELOCITY_INDICES
            print(f"  Using velocity-only indices: v(0-2), w(3-5), q_dot(21-32) = {len(eval_indices)} dims")
        else:
            print(f"  WARNING: state_dim={state_dim} < 33, using all indices")
            eval_indices = list(range(state_dim))
    else:
        eval_indices = list(range(state_dim))
    
    eval_indices_tensor = torch.tensor(eval_indices, device=device_obj)
    
    # Run open-loop prediction
    per_step_errors = []
    per_step_errors_abs = []
    
    # Per-component errors (if save_components is True)
    per_step_errors_v = [] if save_components else None
    per_step_errors_w = [] if save_components else None
    per_step_errors_q_dot = [] if save_components else None
    
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
            
            # Predict next state
            next_state, _, _, _, _, _ = system_dynamics.forward(state_input, action_input)
            
            # Compute error vs ground truth
            gt_state = states_norm[:, t]
            
            # Extract relevant indices for error computation
            pred_selected = next_state[:, eval_indices_tensor]
            gt_selected = gt_state[:, eval_indices_tensor]
            
            # Relative error (same as training metric)
            rel_error = (pred_selected - gt_selected).abs().sum(dim=-1) / (gt_selected.abs().sum(dim=-1) + 1e-8)
            
            # Absolute error (MSE)
            abs_error = (pred_selected - gt_selected).pow(2).sum(dim=-1)
            
            per_step_errors.append(rel_error.cpu().numpy())
            per_step_errors_abs.append(abs_error.cpu().numpy())
            
            # Per-component errors
            if save_components and state_dim >= 33:
                v_indices = torch.tensor(STATE_COMPONENTS['v'], device=device_obj)
                w_indices = torch.tensor(STATE_COMPONENTS['w'], device=device_obj)
                q_dot_indices = torch.tensor(STATE_COMPONENTS['q_dot'], device=device_obj)
                
                v_error = (next_state[:, v_indices] - gt_state[:, v_indices]).abs().sum(dim=-1) / \
                         (gt_state[:, v_indices].abs().sum(dim=-1) + 1e-8)
                w_error = (next_state[:, w_indices] - gt_state[:, w_indices]).abs().sum(dim=-1) / \
                         (gt_state[:, w_indices].abs().sum(dim=-1) + 1e-8)
                q_dot_error = (next_state[:, q_dot_indices] - gt_state[:, q_dot_indices]).abs().sum(dim=-1) / \
                             (gt_state[:, q_dot_indices].abs().sum(dim=-1) + 1e-8)
                
                per_step_errors_v.append(v_error.cpu().numpy())
                per_step_errors_w.append(w_error.cpu().numpy())
                per_step_errors_q_dot.append(q_dot_error.cpu().numpy())
            
            # Append prediction for next step
            state_pred = torch.cat([state_pred, next_state.unsqueeze(1)], dim=1)
    
    # Convert to DataFrame
    per_step_errors = np.array(per_step_errors)  # [horizon, N]
    per_step_errors_abs = np.array(per_step_errors_abs)
    
    results = []
    for t in range(eval_horizon):
        record = {
            'horizon_t': t + 1,
            'error_rel_mean': per_step_errors[t].mean(),
            'error_rel_std': per_step_errors[t].std(),
            'error_abs_mean': per_step_errors_abs[t].mean(),
            'error_abs_std': per_step_errors_abs[t].std(),
        }
        
        # Add per-component errors if available
        if save_components and per_step_errors_v is not None:
            record['error_rel_v_mean'] = np.array(per_step_errors_v)[t].mean()
            record['error_rel_v_std'] = np.array(per_step_errors_v)[t].std()
            record['error_rel_w_mean'] = np.array(per_step_errors_w)[t].mean()
            record['error_rel_w_std'] = np.array(per_step_errors_w)[t].std()
            record['error_rel_q_dot_mean'] = np.array(per_step_errors_q_dot)[t].mean()
            record['error_rel_q_dot_std'] = np.array(per_step_errors_q_dot)[t].std()
        
        results.append(record)
    
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
                        combined[t] = {'rel': [], 'abs': []}
                    combined[t]['rel'].append(row['error_rel_mean'])
                    combined[t]['abs'].append(row['error_abs_mean'])
            
            # Compute mean/std across seeds
            for t, errors in combined.items():
                rel_errors = np.array(errors['rel'])
                abs_errors = np.array(errors['abs'])
                agg_records.append({
                    'condition': condition,
                    'checkpoint_step': step,
                    'horizon_t': t,
                    'error_rel_mean': rel_errors.mean(),
                    'error_rel_std': rel_errors.std(),
                    'error_rel_min': rel_errors.min(),
                    'error_rel_max': rel_errors.max(),
                    'error_abs_mean': abs_errors.mean(),
                    'error_abs_std': abs_errors.std(),
                    'n_seeds': len(rel_errors),
                })
        
        # Save condition-specific file
        agg_df = pd.DataFrame(agg_records)
        output_path = output_dir / f"hallucination_horizon_{condition}.csv"
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
    output_path = output_dir / "hallucination_horizon_all.csv"
    
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
        description='Evaluate hallucination horizon (per-step prediction error)',
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
    parser.add_argument('--save_components', action='store_true',
                        help='Save per-component errors (v, w, q_dot separately)')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='results/hallucination_horizon',
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
        print("\nAlternatively, evaluation can be done within Isaac Sim training loop.")
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
        
        df = evaluate_hallucination_horizon_offline(
            args.checkpoint,
            args.trajectory_data,
            horizon=args.horizon,
            device=args.device,
            velocity_only=args.velocity_only,
            save_components=args.save_components,
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
                
                df = evaluate_hallucination_horizon_offline(
                    ckpt_path,
                    args.trajectory_data,
                    horizon=args.horizon,
                    device=args.device,
                    velocity_only=args.velocity_only,
                    save_components=args.save_components,
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
