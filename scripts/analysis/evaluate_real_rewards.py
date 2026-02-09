#!/usr/bin/env python3
"""
Plot B: Faithfulness Gap - Evaluate real episode rewards from checkpoints.

This script runs policy evaluation on multiple checkpoints to measure
the "faithfulness gap" between imagined rewards (from training logs) and
actual rewards achieved in the real environment.

For each run:
1. Find all checkpoints (or specified ones)
2. Run N episodes and record mean reward
3. Output CSV with real evaluation rewards

Usage:
    # Evaluate all runs in a directory
    python evaluate_real_rewards.py \
        --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
        --log_dir logs/rsl_rl/anymal_d_flat \
        --filter "finetune" \
        --output_dir results/faithfulness_gap/
    
    # Evaluate specific checkpoints
    python evaluate_real_rewards.py \
        --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
        --log_dir logs/rsl_rl/anymal_d_flat \
        --checkpoint_steps 2500 3000 3500 4000 4500 5000 \
        --num_episodes 100 \
        --output_dir results/faithfulness_gap/

Output:
    results/faithfulness_gap/
    ├── real_rewards_all.csv           # All runs and checkpoints
    ├── real_rewards_<condition>.csv   # Per-condition aggregated
    └── faithfulness_comparison.csv    # Combined with imagination rewards
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate real rewards from policy checkpoints.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--log_dir", type=str, required=True, help="Directory containing runs.")
parser.add_argument("--filter", type=str, default="finetune",
                    help='Filter runs by pattern (default: "finetune"). Use --filter "" for no filtering.')
parser.add_argument("--checkpoint_steps", type=int, nargs='+', default=None,
                    help="Specific checkpoint steps to evaluate (default: every 500 steps)")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes per checkpoint.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--output_dir", type=str, default="results/faithfulness_gap",
                    help="Output directory for results.")
parser.add_argument("--imagination_data_dir", type=str, default=None,
                    help="Directory with TensorBoard exports for imagination rewards comparison.")
# Note: --device is provided by AppLauncher, don't add it here

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app (headless)
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners.mbpo_on_policy_runner import MBPOOnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import custom tasks
import mbrl.tasks  # noqa: F401


@dataclass
class RunMetadata:
    """Parsed metadata from a run directory name."""
    full_name: str
    timestamp: str
    phase: str
    method: str
    seed: int
    penalty: Optional[float] = None
    uncertainty_metric: Optional[str] = None
    prior_scale: Optional[float] = None
    bootstrap: Optional[bool] = None
    condition_key: str = field(init=False)
    
    def __post_init__(self):
        parts = [self.phase, self.method]
        if self.penalty is not None:
            parts.append(f"pen{self.penalty}")
        if self.uncertainty_metric:
            parts.append(self.uncertainty_metric)
        if self.prior_scale is not None:
            parts.append(f"prior{self.prior_scale}")
        if self.bootstrap is not None:
            parts.append("boot" if self.bootstrap else "noboot")
        self.condition_key = "-".join(parts)


def parse_run_name(run_dir: str) -> Optional[RunMetadata]:
    """Parse a run directory name to extract metadata."""
    dir_name = os.path.basename(run_dir)
    
    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)', dir_name)
    if not timestamp_match:
        return None
    
    timestamp = timestamp_match.group(1)
    run_name = timestamp_match.group(2)
    
    seed_match = re.search(r'seed(\d+)', run_name)
    if not seed_match:
        return None
    seed = int(seed_match.group(1))
    
    if 'pretrain' in run_name:
        phase = 'pretrain'
    elif 'finetune' in run_name:
        phase = 'finetune'
    else:
        phase = 'unknown'
    
    if '-rp-' in run_name or 'priors' in run_name or 'prior' in run_name:
        method = 'rp'
    elif '-bs-' in run_name or 'bootstrap' in run_name:
        method = 'bs'
    elif 'ensemble' in run_name:
        method = 'ensemble'
    else:
        method = 'unknown'
    
    penalty = None
    penalty_match = re.search(r'pen(\d+)', run_name)
    if penalty_match:
        penalty = int(penalty_match.group(1)) / 100.0
    
    uncertainty_metric = None
    if '-std' in run_name or '-std-' in run_name:
        uncertainty_metric = 'std'
    elif '-var' in run_name or '-var-' in run_name:
        uncertainty_metric = 'var'
    
    prior_scale = None
    prior_match = re.search(r'prior([\d.]+)', run_name)
    if prior_match:
        prior_scale = float(prior_match.group(1))
    
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


def find_checkpoints(run_dir: str, checkpoint_steps: Optional[list] = None) -> list:
    """Find model checkpoints in a run directory."""
    run_path = Path(run_dir)
    checkpoints = []
    
    for f in run_path.glob("model_*.pt"):
        step_match = re.search(r'model_(\d+)\.pt', f.name)
        if step_match:
            step = int(step_match.group(1))
            if checkpoint_steps is None or step in checkpoint_steps:
                checkpoints.append((step, str(f)))
    
    return sorted(checkpoints, key=lambda x: x[0])


def evaluate_checkpoint(
    env,
    checkpoint_path: str,
    agent_cfg,
    num_episodes: int,
    device: str,
) -> dict:
    """
    Evaluate a checkpoint and return episode statistics.
    
    Returns:
        Dictionary with reward statistics
    """
    # Convert config to dict for inspection
    agent_cfg_dict = agent_cfg.to_dict()
    
    # Check if this is an MBPO checkpoint by looking at the agent config class_name
    if agent_cfg.class_name == "MBPOOnPolicyRunner":
        # For evaluation, we don't need to load system dynamics from a separate pretrain checkpoint
        # The checkpoint itself may contain the system dynamics state
        agent_cfg_dict["load_system_dynamics"] = False
        runner = MBPOOnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=device)
    else:
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=device)
    
    runner.load(checkpoint_path, load_optimizer=False)
    
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Collect episode rewards
    episode_rewards = []
    episode_lengths = []
    
    obs = env.get_observations()
    current_rewards = torch.zeros(env.num_envs, device=env.unwrapped.device)
    current_lengths = torch.zeros(env.num_envs, device=env.unwrapped.device)
    
    max_steps = num_episodes * 1000 // env.num_envs + 1000  # Safety margin
    
    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            
            current_rewards += rewards
            current_lengths += 1
            
            # Record completed episodes
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_indices:
                episode_rewards.append(current_rewards[idx].item())
                episode_lengths.append(current_lengths[idx].item())
                current_rewards[idx] = 0
                current_lengths[idx] = 0
            
            if len(episode_rewards) >= num_episodes:
                break
    
    # Trim to exact count
    episode_rewards = episode_rewards[:num_episodes]
    episode_lengths = episode_lengths[:num_episodes]
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'num_episodes': len(episode_rewards),
    }


def aggregate_results(
    all_results: list[dict],
    output_dir: Path,
):
    """Aggregate results across seeds."""
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Save all individual results
    output_path = output_dir / "real_rewards_all.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # Aggregate by condition and step
    grouped = df.groupby(['condition', 'checkpoint_step'])
    
    agg_records = []
    for (condition, step), group in grouped:
        agg_records.append({
            'condition': condition,
            'checkpoint_step': step,
            'real_reward_mean': group['mean_reward'].mean(),
            'real_reward_std': group['mean_reward'].std(),
            'real_reward_min': group['mean_reward'].min(),
            'real_reward_max': group['mean_reward'].max(),
            'real_length_mean': group['mean_length'].mean(),
            'real_length_std': group['mean_length'].std(),
            'n_seeds': len(group),
            'seeds': list(group['seed']),
        })
    
    agg_df = pd.DataFrame(agg_records)
    
    # Save per-condition files
    for condition in agg_df['condition'].unique():
        cond_df = agg_df[agg_df['condition'] == condition]
        output_path = output_dir / f"real_rewards_{condition}.csv"
        cond_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
    
    return agg_df


def load_imagination_rewards(data_dir: str, conditions: list[str]) -> dict:
    """Load imagination rewards from TensorBoard exports."""
    imagination_data = {}
    data_path = Path(data_dir)
    
    for condition in conditions:
        csv_path = data_path / f"{condition}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            imagination_data[condition] = df
    
    return imagination_data


def create_faithfulness_comparison(
    real_rewards_df: pd.DataFrame,
    imagination_data: dict,
    output_dir: Path,
):
    """
    Create comparison CSV with both real and imagination rewards.
    """
    comparison_records = []
    
    for _, row in real_rewards_df.iterrows():
        condition = row['condition']
        step = row['checkpoint_step']
        
        record = {
            'condition': condition,
            'step': step,
            'real_reward_mean': row['real_reward_mean'],
            'real_reward_std': row['real_reward_std'],
            'n_seeds_real': row['n_seeds'],
        }
        
        # Try to find matching imagination reward
        if condition in imagination_data:
            imag_df = imagination_data[condition]
            
            # Find closest step
            closest_idx = (imag_df['step'] - step).abs().argmin()
            imag_row = imag_df.iloc[closest_idx]
            
            # Try different column names for imagination reward
            # Note: Model_Based_per_step_reward_sum_imagination is a SUM over horizon*envs,
            # not comparable to episode rewards. We prefer Train_mean_reward for episode-level
            # comparison, or Train_mean_reward_imagination for per-step imagination rewards.
            for col_name in ['Train_mean_reward_imagination_mean',
                           'Train_mean_reward_mean']:
                if col_name in imag_row.index and pd.notna(imag_row[col_name]):
                    record['imagination_reward_mean'] = imag_row[col_name]
                    std_col = col_name.replace('_mean', '_std')
                    if std_col in imag_row.index:
                        record['imagination_reward_std'] = imag_row[std_col]
                    break
            
            # Compute gap
            if 'imagination_reward_mean' in record:
                record['faithfulness_gap'] = record['imagination_reward_mean'] - record['real_reward_mean']
                record['faithfulness_ratio'] = record['imagination_reward_mean'] / (record['real_reward_mean'] + 1e-8)
        
        comparison_records.append(record)
    
    comparison_df = pd.DataFrame(comparison_records)
    output_path = output_dir / "faithfulness_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    return comparison_df


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Evaluate real rewards from multiple checkpoints."""
    
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure environment
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    
    # Find all runs
    log_path = Path(args_cli.log_dir)
    run_dirs = sorted([d for d in log_path.iterdir() if d.is_dir()])
    
    print(f"[INFO] Found {len(run_dirs)} run directories")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Num episodes per checkpoint: {args_cli.num_episodes}")
    
    # Create environment once
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    all_results = []
    
    for run_dir in run_dirs:
        if args_cli.filter and args_cli.filter not in run_dir.name:
            continue
        
        metadata = parse_run_name(str(run_dir))
        if metadata is None:
            print(f"  Skipping (could not parse): {run_dir.name}")
            continue
        
        print(f"\n[INFO] Processing: {run_dir.name}")
        print(f"  Condition: {metadata.condition_key}, Seed: {metadata.seed}")
        
        checkpoints = find_checkpoints(str(run_dir), args_cli.checkpoint_steps)
        
        if not checkpoints:
            print(f"  No checkpoints found")
            continue
        
        # If no specific steps, sample evenly
        if args_cli.checkpoint_steps is None:
            # Take every 500 steps
            checkpoints = [(s, p) for s, p in checkpoints if s % 500 == 0]
        
        print(f"  Found {len(checkpoints)} checkpoints to evaluate")
        
        for step, ckpt_path in checkpoints:
            print(f"    Evaluating step {step}...")
            
            try:
                stats = evaluate_checkpoint(
                    env,
                    ckpt_path,
                    agent_cfg,
                    args_cli.num_episodes,
                    agent_cfg.device,
                )
                
                result = {
                    'run_name': run_dir.name,
                    'condition': metadata.condition_key,
                    'seed': metadata.seed,
                    'checkpoint_step': step,
                    'checkpoint_path': ckpt_path,
                    **stats,
                }
                
                all_results.append(result)
                print(f"      Mean reward: {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
                
            except Exception as e:
                print(f"      ERROR: {e}")
                continue
    
    env.close()
    
    if not all_results:
        print("\nNo results collected!")
        return
    
    print(f"\n[INFO] Collected {len(all_results)} checkpoint evaluations")
    
    # Aggregate results
    agg_df = aggregate_results(all_results, output_dir)
    
    # Compare with imagination rewards if available
    if args_cli.imagination_data_dir:
        print("\n[INFO] Loading imagination reward data...")
        conditions = list(agg_df['condition'].unique())
        imagination_data = load_imagination_rewards(args_cli.imagination_data_dir, conditions)
        
        if imagination_data:
            create_faithfulness_comparison(agg_df, imagination_data, output_dir)
    
    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
    simulation_app.close()
