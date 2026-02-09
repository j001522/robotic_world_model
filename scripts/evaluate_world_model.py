#!/usr/bin/env python3
"""
Evaluate and visualize world model predictions.

This script loads a pretrained world model and generates trajectory comparison plots
showing real vs predicted states. No Isaac Sim required - runs on CPU/GPU directly.

Usage:
    python scripts/evaluate_world_model.py \
        --checkpoint logs/rsl_rl/anymal_d_flat/<timestamp>/model_2500.pt \
        --output_dir logs/rsl_rl/anymal_d_flat/<timestamp>/eval_plots
"""

import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add rsl_rl to path
sys.path.insert(0, "/gpfs/work4/0/prjs0951/Giacomo/isaac-sim/overlay/rsl_rl_rwm")

from rsl_rl.modules.system_dynamics import SystemDynamicsEnsemble


def load_checkpoint(checkpoint_path, device="cuda"):
    """Load model checkpoint and extract world model."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Print checkpoint keys for debugging
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    return checkpoint


def extract_world_model_config(checkpoint):
    """Extract world model configuration from checkpoint."""
    # Try to find system dynamics config
    if "system_dynamics_state_dict" in checkpoint:
        sd_state = checkpoint["system_dynamics_state_dict"]
        # Infer dimensions from state dict
        # state_base.memory.rnn.weight_ih_l0 shape tells us input dim
        for key, value in sd_state.items():
            print(f"  {key}: {value.shape if hasattr(value, 'shape') else type(value)}")
    return checkpoint


def plot_training_curves(checkpoint_dir, output_dir):
    """Extract and plot training curves from TensorBoard logs."""
    from tensorboard.backend.event_processing import event_accumulator
    
    event_files = [f for f in os.listdir(checkpoint_dir) if 'tfevents' in f]
    if not event_files:
        print("No TensorBoard event files found")
        return
    
    event_file = os.path.join(checkpoint_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot World Model losses
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # State loss
    ax = axes[0, 0]
    try:
        events = ea.Scalars('System Dynamics/state_loss')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, 'b-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('State Loss')
        ax.set_title('World Model State Loss')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    except:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    # Autoregressive error
    ax = axes[0, 1]
    try:
        events = ea.Scalars('System Dynamics/autoregressive_error')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        # Filter out inf values
        valid = [(s, v) for s, v in zip(steps, values) if np.isfinite(v)]
        if valid:
            steps, values = zip(*valid)
            ax.plot(steps, values, 'r-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Autoregressive Error')
        ax.set_title('Multi-step Rollout Error')
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    # Mean reward
    ax = axes[1, 0]
    try:
        events = ea.Scalars('Train/mean_reward')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, 'g-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Policy Training Reward')
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    # Episode length
    ax = axes[1, 1]
    try:
        events = ea.Scalars('Train/mean_episode_length')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        ax.plot(steps, values, 'm-', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Episode Length')
        ax.set_title('Mean Episode Length')
        ax.grid(True, alpha=0.3)
    except:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to: {output_path}")
    plt.close()
    
    # Plot additional WM metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    wm_metrics = [
        ('System Dynamics/bound_loss', 'Bound Loss'),
        ('System Dynamics/kl_loss', 'KL Loss'),
        ('System Dynamics/contact_loss', 'Contact Loss'),
        ('System Dynamics/termination_loss', 'Termination Loss'),
    ]
    
    for idx, (tag, title) in enumerate(wm_metrics):
        ax = axes[idx // 2, idx % 2]
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            ax.plot(steps, values, alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        except:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'wm_auxiliary_losses.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved WM auxiliary losses to: {output_path}")
    plt.close()
    
    # Plot noised autoregressive errors
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    noise_levels = [0.1, 0.2, 0.4, 0.5, 0.8]
    colors = plt.cm.viridis(np.linspace(0, 1, len(noise_levels)))
    
    for noise, color in zip(noise_levels, colors):
        tag = f'System Dynamics/autoregressive_error_noised_{noise}'
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            valid = [(s, v) for s, v in zip(steps, values) if np.isfinite(v)]
            if valid:
                steps, values = zip(*valid)
                ax.plot(steps, values, color=color, alpha=0.7, label=f'noise={noise}')
        except:
            pass
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Autoregressive Error')
    ax.set_title('Autoregressive Error with Different Noise Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'autoregressive_error_noised.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved noised autoregressive errors to: {output_path}")
    plt.close()


def print_summary(checkpoint_dir):
    """Print a summary of training metrics."""
    from tensorboard.backend.event_processing import event_accumulator
    
    event_files = [f for f in os.listdir(checkpoint_dir) if 'tfevents' in f]
    if not event_files:
        print("No TensorBoard event files found")
        return
    
    event_file = os.path.join(checkpoint_dir, event_files[0])
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    print("\n" + "="*60)
    print("WORLD MODEL PRETRAINING SUMMARY")
    print("="*60)
    
    # State loss
    try:
        events = ea.Scalars('System Dynamics/state_loss')
        print(f"\nState Loss:")
        print(f"  Initial: {events[0].value:.4f}")
        print(f"  Final:   {events[-1].value:.4f}")
        print(f"  Reduction: {events[0].value / events[-1].value:.1f}x")
    except:
        pass
    
    # Autoregressive error
    try:
        events = ea.Scalars('System Dynamics/autoregressive_error')
        valid = [e for e in events if np.isfinite(e.value)]
        if valid:
            print(f"\nAutoregressive Error:")
            print(f"  Best:  {min(e.value for e in valid):.4f}")
            print(f"  Final: {valid[-1].value:.4f}")
    except:
        pass
    
    # Mean reward
    try:
        events = ea.Scalars('Train/mean_reward')
        print(f"\nPolicy Reward:")
        print(f"  Initial: {events[0].value:.2f}")
        print(f"  Final:   {events[-1].value:.2f}")
    except:
        pass
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model predictions")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint (model_*.pt)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: <checkpoint_dir>/eval_plots)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    
    args = parser.parse_args()
    
    # Resolve paths
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    if args.output_dir is None:
        output_dir = os.path.join(checkpoint_dir, "eval_plots")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")
    
    # Print summary
    print_summary(checkpoint_dir)
    
    # Generate plots
    plot_training_curves(checkpoint_dir, output_dir)
    
    print(f"\nDone! Plots saved to: {output_dir}")
    print("\nTo view plots, copy to local machine:")
    print(f"  scp -r snellius:{output_dir}/*.png .")


if __name__ == "__main__":
    main()
