#!/usr/bin/env python3
"""
Record trajectories from trained policies for offline evaluation.

This script runs a trained policy in Isaac Sim and records state-action trajectories
that can be used for offline evaluation of world model prediction accuracy.

Usage:
    python record_trajectories.py \
        --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
        --checkpoint logs/rsl_rl/anymal_d_flat/2026-02-03_run/model_5000.pt \
        --output trajectories.pt \
        --num_trajectories 100 \
        --trajectory_length 100

Output format (.pt file):
    {
        'states': Tensor[N, T, state_dim],      # System states
        'actions': Tensor[N, T, action_dim],    # Actions taken
        'rewards': Tensor[N, T],                # Rewards received
        'dones': Tensor[N, T],                  # Episode termination flags
        'metadata': {
            'checkpoint': str,
            'task': str,
            'num_trajectories': int,
            'trajectory_length': int,
        }
    }
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Record trajectories from a trained policy.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--output", type=str, default="trajectories.pt", help="Output file path.")
parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to record.")
parser.add_argument("--trajectory_length", type=int, default=200, help="Length of each trajectory.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
# Note: --device is added by AppLauncher.add_app_launcher_args()

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app (headless by default)
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners.mbpo_on_policy_runner import MBPOOnPolicyRunner

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import custom tasks
import mbrl.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Record trajectories from a trained policy."""
    
    # Configure environment
    num_envs = args_cli.num_envs if args_cli.num_envs else min(args_cli.num_trajectories, 4096)
    env_cfg.scene.num_envs = num_envs
    
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    
    # Device is set by AppLauncher via --device argument
    
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Checkpoint: {args_cli.checkpoint}")
    print(f"[INFO] Num envs: {num_envs}")
    print(f"[INFO] Num trajectories: {args_cli.num_trajectories}")
    print(f"[INFO] Trajectory length: {args_cli.trajectory_length}")
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    # Load policy
    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    
    print(f"[INFO] Loading model from: {resume_path}")
    
    # Check if this is an MBPO checkpoint by looking at the agent config
    agent_cfg_dict = agent_cfg.to_dict()
    is_mbpo = agent_cfg_dict.get("algorithm", {}).get("class_name", "") == "MBPOPPO"
    
    if is_mbpo:
        print("[INFO] Detected MBPOPPO checkpoint, using MBPOOnPolicyRunner")
        # For inference, we don't need to load the system dynamics from a separate pretrain checkpoint
        # The checkpoint itself contains the system dynamics state
        agent_cfg_dict["load_system_dynamics"] = False
        runner = MBPOOnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    else:
        print("[INFO] Using standard OnPolicyRunner")
        runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    # Storage for trajectories
    all_states = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    trajectories_collected = 0
    target_trajectories = args_cli.num_trajectories
    trajectory_length = args_cli.trajectory_length
    
    print(f"\n[INFO] Recording trajectories...")
    
    # Collect trajectories in batches
    while trajectories_collected < target_trajectories:
        # Reset environment
        obs = env.get_observations()
        
        # Current batch storage
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        
        # Collect one trajectory length
        with torch.inference_mode():
            for t in range(trajectory_length):
                # Get system state (the full state used for world model)
                if "system_state" in obs:
                    state = obs["system_state"].clone()
                else:
                    # Fall back to policy observation
                    state = obs["policy"].clone()
                
                # Get action from policy
                action = policy(obs)
                
                # Step environment
                obs, rewards, dones, extras = env.step(action)
                
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(rewards)
                batch_dones.append(dones)
        
        # Stack batch
        batch_states = torch.stack(batch_states, dim=1)  # [num_envs, T, state_dim]
        batch_actions = torch.stack(batch_actions, dim=1)  # [num_envs, T, action_dim]
        batch_rewards = torch.stack(batch_rewards, dim=1)  # [num_envs, T]
        batch_dones = torch.stack(batch_dones, dim=1)  # [num_envs, T]
        
        # Add to collection
        all_states.append(batch_states.cpu())
        all_actions.append(batch_actions.cpu())
        all_rewards.append(batch_rewards.cpu())
        all_dones.append(batch_dones.cpu())
        
        trajectories_collected += num_envs
        print(f"  Collected {min(trajectories_collected, target_trajectories)}/{target_trajectories} trajectories")
    
    # Combine all batches
    all_states = torch.cat(all_states, dim=0)[:target_trajectories]
    all_actions = torch.cat(all_actions, dim=0)[:target_trajectories]
    all_rewards = torch.cat(all_rewards, dim=0)[:target_trajectories]
    all_dones = torch.cat(all_dones, dim=0)[:target_trajectories]
    
    print(f"\n[INFO] Final shapes:")
    print(f"  States: {all_states.shape}")
    print(f"  Actions: {all_actions.shape}")
    print(f"  Rewards: {all_rewards.shape}")
    print(f"  Dones: {all_dones.shape}")
    
    # Save to file
    output_data = {
        'states': all_states,
        'actions': all_actions,
        'rewards': all_rewards,
        'dones': all_dones,
        'metadata': {
            'checkpoint': args_cli.checkpoint,
            'task': args_cli.task,
            'num_trajectories': target_trajectories,
            'trajectory_length': trajectory_length,
        }
    }
    
    torch.save(output_data, args_cli.output)
    print(f"\n[INFO] Saved trajectories to: {args_cli.output}")
    
    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
