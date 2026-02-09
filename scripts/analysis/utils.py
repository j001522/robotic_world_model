#!/usr/bin/env python3
"""
Shared utilities for analysis scripts.

This module provides:
- RunMetadata dataclass for parsed run directory names
- parse_run_name() function to extract metadata from run directory names
- find_checkpoints() function to locate model checkpoints
- State component definitions for AnymalD
- Checkpoint loading utilities

Usage:
    from utils import RunMetadata, parse_run_name, find_checkpoints, STATE_COMPONENTS
"""

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml


# =============================================================================
# State Component Definitions for AnymalD
# =============================================================================
# These indices define which parts of the system_state correspond to which
# physical quantities. Based on flat_env_cfg.py::ObservationsCfg_PRETRAIN.SystemStateCfg
#
# State structure (45 dimensions total):
#   0-2:   base_lin_vel (v) - Base linear velocity in body frame
#   3-5:   base_ang_vel (w) - Base angular velocity in body frame
#   6-8:   projected_gravity - Gravity vector in body frame
#   9-20:  joint_pos (q) - 12 joint positions (relative to default)
#   21-32: joint_vel (q_dot) - 12 joint velocities
#   33-44: joint_torque - 12 joint torques/efforts

STATE_COMPONENTS = {
    'v': list(range(0, 3)),       # Base linear velocity
    'w': list(range(3, 6)),       # Base angular velocity
    'gravity': list(range(6, 9)), # Projected gravity
    'q': list(range(9, 21)),      # Joint positions
    'q_dot': list(range(21, 33)), # Joint velocities
    'torque': list(range(33, 45)),# Joint torques
}

# Velocity-related components (v, w, q_dot)
VELOCITY_INDICES = STATE_COMPONENTS['v'] + STATE_COMPONENTS['w'] + STATE_COMPONENTS['q_dot']


# =============================================================================
# Run Metadata
# =============================================================================

@dataclass
class RunMetadata:
    """
    Parsed metadata from a run directory name.
    
    Run directory names follow the pattern:
        {timestamp}_{phase}-{method}-pen{penalty}-{uncertainty_metric}_seed{seed}
    
    Examples:
        2026-02-03_15-30-00_finetune-rp-pen025-std_seed42
        2026-02-03_15-30-00_finetune-bs-nopen_seed1
    
    Attributes:
        full_name: The complete directory name
        timestamp: Timestamp string (YYYY-MM-DD_HH-MM-SS)
        phase: Training phase ('pretrain' or 'finetune')
        method: Method used ('rp' for random priors, 'bs' for bootstrap, 'ensemble')
        seed: Random seed
        penalty: Penalty coefficient (e.g., 0.25) or None if no penalty
        uncertainty_metric: 'std' or 'var' for uncertainty computation, or None
        prior_scale: Prior scale for random priors method, or None
        bootstrap: Whether bootstrap is enabled, or None if not applicable
        condition_key: Generated key for grouping runs by condition (auto-computed)
    """
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
        """Generate condition_key from other fields."""
        # For condition_key, we use a simplified format suitable for paper figures
        # e.g., "bs-nopen", "bs-pen025", "rp-pen025-std"
        parts = [self.method]
        
        if self.penalty is not None:
            # Convert penalty to string without decimal point (e.g., 0.25 -> "025")
            pen_str = str(self.penalty).replace('.', '').replace('0', '', 1) if self.penalty < 1 else str(int(self.penalty))
            # Ensure consistent formatting: 0.25 -> "025", 0.06 -> "006"
            pen_int = int(self.penalty * 100)
            pen_str = f"{pen_int:03d}" if pen_int < 100 else str(pen_int)
            parts.append(f"pen{pen_str}")
        else:
            parts.append("nopen")
        
        if self.uncertainty_metric:
            parts.append(self.uncertainty_metric)
        
        self.condition_key = "-".join(parts)
    
    def to_full_condition_key(self) -> str:
        """
        Generate a full condition key including phase.
        
        Returns key like "finetune-rp-pen025-std" instead of just "rp-pen025-std".
        """
        parts = [self.phase, self.method]
        
        if self.penalty is not None:
            pen_int = int(self.penalty * 100)
            pen_str = f"{pen_int:03d}" if pen_int < 100 else str(pen_int)
            parts.append(f"pen{pen_str}")
        else:
            parts.append("nopen")
        
        if self.uncertainty_metric:
            parts.append(self.uncertainty_metric)
        
        if self.prior_scale is not None:
            parts.append(f"prior{self.prior_scale}")
        
        if self.bootstrap is not None:
            parts.append("boot" if self.bootstrap else "noboot")
        
        return "-".join(parts)


def parse_run_name(run_dir: str) -> Optional[RunMetadata]:
    """
    Parse a run directory name to extract metadata.
    
    Args:
        run_dir: Path to run directory (can be full path or just directory name)
    
    Returns:
        RunMetadata object if parsing succeeds, None otherwise
    
    Examples:
        >>> meta = parse_run_name("2026-02-03_15-30-00_finetune-rp-pen025-std_seed42")
        >>> meta.method
        'rp'
        >>> meta.penalty
        0.25
        >>> meta.condition_key
        'rp-pen025-std'
    """
    dir_name = os.path.basename(run_dir)
    
    # Match timestamp at the start
    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(.+)', dir_name)
    if not timestamp_match:
        return None
    
    timestamp = timestamp_match.group(1)
    run_name = timestamp_match.group(2)
    
    # Extract seed (required)
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
        method = 'rp'
    elif '-bs-' in run_name or 'bootstrap' in run_name:
        method = 'bs'
    elif 'ensemble' in run_name:
        method = 'ensemble'
    else:
        method = 'unknown'
    
    # Extract penalty value
    penalty = None
    penalty_match = re.search(r'pen(\d+)', run_name)
    if penalty_match:
        # Convert e.g., "025" to 0.25, "006" to 0.06
        penalty = int(penalty_match.group(1)) / 100.0
    
    # Extract uncertainty metric
    uncertainty_metric = None
    if '-std' in run_name or '-std-' in run_name or run_name.endswith('-std'):
        uncertainty_metric = 'std'
    elif '-var' in run_name or '-var-' in run_name or run_name.endswith('-var'):
        uncertainty_metric = 'var'
    
    # Extract prior scale (optional)
    prior_scale = None
    prior_match = re.search(r'prior([\d.]+)', run_name)
    if prior_match:
        prior_scale = float(prior_match.group(1))
    
    # Extract bootstrap flag (optional)
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


# =============================================================================
# Checkpoint Discovery
# =============================================================================

def find_checkpoints(run_dir: str, checkpoint_steps: Optional[list] = None) -> list[tuple[int, str]]:
    """
    Find all model checkpoints in a run directory.
    
    Args:
        run_dir: Path to run directory
        checkpoint_steps: Optional list of specific steps to find. If None, returns all.
    
    Returns:
        List of (step, checkpoint_path) tuples, sorted by step
    
    Example:
        >>> checkpoints = find_checkpoints("logs/rsl_rl/anymal_d_flat/2026-02-03_run")
        >>> checkpoints
        [(1000, '/path/to/model_1000.pt'), (2000, '/path/to/model_2000.pt'), ...]
    """
    run_path = Path(run_dir)
    checkpoints = []
    
    for f in run_path.glob("model_*.pt"):
        step_match = re.search(r'model_(\d+)\.pt', f.name)
        if step_match:
            step = int(step_match.group(1))
            if checkpoint_steps is None or step in checkpoint_steps:
                checkpoints.append((step, str(f)))
    
    return sorted(checkpoints, key=lambda x: x[0])


def find_latest_checkpoint(run_dir: str) -> Optional[tuple[int, str]]:
    """
    Find the latest (highest step) checkpoint in a run directory.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Tuple of (step, checkpoint_path) or None if no checkpoints found
    """
    checkpoints = find_checkpoints(run_dir)
    return checkpoints[-1] if checkpoints else None


# =============================================================================
# Condition Discovery
# =============================================================================

def discover_conditions(
    log_dir: str,
    filter_pattern: Optional[str] = None,
    phase_filter: Optional[str] = None,
    min_seeds: int = 1,
) -> dict[str, list[RunMetadata]]:
    """
    Discover all conditions (method + penalty + uncertainty combinations) in a log directory.
    
    Groups runs by condition_key and optionally filters by pattern, phase, or minimum seeds.
    
    Args:
        log_dir: Directory containing run directories
        filter_pattern: Optional pattern to filter run names (substring match)
        phase_filter: Optional phase to filter by ('pretrain' or 'finetune')
        min_seeds: Minimum number of seeds required to include a condition
    
    Returns:
        Dictionary mapping condition_key -> list of RunMetadata for that condition
    
    Example:
        >>> conditions = discover_conditions("logs/rsl_rl/anymal_d_flat", phase_filter="finetune", min_seeds=3)
        >>> conditions.keys()
        dict_keys(['bs-nopen', 'bs-pen025', 'rp-pen025-std'])
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # Group runs by condition
    conditions: dict[str, list[RunMetadata]] = {}
    
    for run_dir in sorted(log_path.iterdir()):
        if not run_dir.is_dir():
            continue
        
        # Apply filter pattern
        if filter_pattern and filter_pattern not in run_dir.name:
            continue
        
        metadata = parse_run_name(str(run_dir))
        if metadata is None:
            continue
        
        # Apply phase filter
        if phase_filter and metadata.phase != phase_filter:
            continue
        
        # Group by condition
        if metadata.condition_key not in conditions:
            conditions[metadata.condition_key] = []
        conditions[metadata.condition_key].append(metadata)
    
    # Filter by minimum seeds
    if min_seeds > 1:
        conditions = {k: v for k, v in conditions.items() if len(v) >= min_seeds}
    
    return conditions


# =============================================================================
# Checkpoint Loading
# =============================================================================

def load_checkpoint_and_config(
    checkpoint_path: str,
    device: str = "cuda",
) -> tuple[dict, dict, Path]:
    """
    Load a checkpoint and its configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load checkpoint on
    
    Returns:
        Tuple of (checkpoint_dict, agent_config_dict, run_dir_path)
    
    Raises:
        FileNotFoundError: If checkpoint or config files don't exist
    """
    # Handle device
    if device == "cuda" and not torch.cuda.is_available():
        print("  WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config from params directory
    run_dir = Path(checkpoint_path).parent
    agent_cfg_path = run_dir / "params" / "agent.yaml"
    
    if not agent_cfg_path.exists():
        raise FileNotFoundError(f"Agent config not found: {agent_cfg_path}")
    
    with open(agent_cfg_path, 'r') as f:
        agent_cfg = yaml.safe_load(f)
    
    return checkpoint, agent_cfg, run_dir


def infer_dimensions_from_checkpoint(checkpoint: dict) -> tuple[int, int]:
    """
    Infer contact_dim and termination_dim from a checkpoint's state dict.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
    
    Returns:
        Tuple of (contact_dim, termination_dim)
    """
    sd_state = checkpoint.get('system_dynamics_state_dict', {})
    contact_dim = 0
    termination_dim = 0
    
    for key in sd_state.keys():
        if 'contact_layers' in key and key.endswith('.bias') and '.0.' in key:
            if '2.bias' in key:
                contact_dim = sd_state[key].shape[0]
                break
    
    for key in sd_state.keys():
        if 'termination_layers' in key and key.endswith('.bias') and '.0.' in key:
            if '2.bias' in key:
                termination_dim = sd_state[key].shape[0]
                break
    
    return contact_dim, termination_dim


# =============================================================================
# Path Utilities
# =============================================================================

def get_rsl_rl_rwm_path() -> Path:
    """
    Get the path to the rsl_rl_rwm module.
    
    Returns:
        Path to the rsl_rl_rwm directory
    """
    # Relative path from scripts/analysis/ to isaac-sim/overlay/rsl_rl_rwm
    return Path(__file__).parent.parent.parent.parent / "isaac-sim" / "overlay" / "rsl_rl_rwm"


def ensure_rsl_rl_rwm_in_path():
    """
    Ensure rsl_rl_rwm is in sys.path for importing modules.
    
    This is needed when running analysis scripts outside of Isaac Sim.
    """
    rwm_path = str(get_rsl_rl_rwm_path())
    if rwm_path not in sys.path:
        sys.path.insert(0, rwm_path)
