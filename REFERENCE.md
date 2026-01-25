# Technical Reference

Technical details, configuration options, and research notes for Robotic World Model.

---

## Table of Contents

1. [RWM vs U-RWM Quick Reference](#rwm-vs-u-rwm-quick-reference)
2. [Configuration Parameters](#configuration-parameters)
3. [System Dynamics Ensemble Details](#system-dynamics-ensemble-details)
4. [Uncertainty Quantification](#uncertainty-quantification)
5. [Research: Ensemble Improvements](#research-ensemble-improvements)
6. [File Locations](#file-locations)

---

## RWM vs U-RWM Quick Reference

### Key Insight

**RWM and U-RWM use the SAME codebase** - only configuration differs!

### Configuration Differences

| Parameter | RWM | U-RWM |
|-----------|-----|-------|
| **ensemble_size** | 1 | 5 |
| **Training Mode** | Online (with simulator) | Offline (dataset only) |
| **Epistemic Uncertainty** | Always 0 (disabled) | Active (ensemble disagreement) |
| **uncertainty_penalty_weight** | -0.0 | -1.0 |
| **Config File** | `rsl_rl_ppo_cfg.py` | `anymal_d_flat_cfg.py` |
| **Script** | `rsl_rl/train.py` | `model_based/train.py` |

### Shared Components

Both use the **exact same code**:
- ✅ `SystemDynamicsEnsemble` class
- ✅ `MBPOPPO` algorithm
- ✅ Network architectures (RNN, MLP heads)
- ✅ Training procedures (bootstrap sampling)

**Location**: `isaac-sim/overlay/rsl_rl_rwm/rsl_rl/`

### Research Options

**Option 1: Online Ensemble** ⭐ Recommended

Combine RWM + U-RWM benefits:

```python
# In rsl_rl_ppo_cfg.py
ensemble_size = 5  # Enable ensemble
uncertainty_penalty_weight = -0.1  # Enable uncertainty penalty
```

**Best for**: Uncertainty modeling research, safer online learning

---

## Configuration Parameters

### System Dynamics Configuration

**File**: `source/mbrl/mbrl/tasks/.../config/anymal_d/agents/rsl_rl_ppo_cfg.py`

```python
@configclass
class RslRlSystemDynamicsCfg:
    # Core Settings
    enabled: bool = True
    ensemble_size: int = 5                # 1=RWM, 5=U-RWM
    history_horizon: int = 32             # State-action history length
    forecast_horizon: int = 8             # Multi-step prediction steps
    
    # Architecture
    architecture_config: Dict = {
        "type": "rnn",                    # Architecture type
        "rnn_type": "gru",                # GRU or LSTM
        "rnn_num_layers": 2,              # RNN depth
        "rnn_hidden_size": 256,           # RNN hidden size
        "state_mean_shape": [128],        # MLP head layers
        "state_logstd_shape": [128],
    }
    
    # Training
    train_interval: int = 24              # Steps before model update
    train_batch_size: int = 256           # Batch size
    train_epochs: int = 50                # Epochs per update
    learning_rate: float = 1e-4           # Learning rate
    
    # Loss Functions
    regression_loss_type: str = "nll"    # "mse" or "nll"
    state_min_logstd: float = -5.0       # Min log std
    state_max_logstd: float = 2.0        # Max log std
    
    # Observation Groups
    obs_groups: dict = {
        "state": ["base_lin_vel", "base_ang_vel", "projected_gravity",
                  "joint_pos", "joint_vel", "joint_torque"]
    }
```

### Imagination Configuration

```python
@configclass
class RslRlMbrlImaginationCfg:
    num_envs: int = 4096                  # Parallel imagination envs
    num_steps_per_env: int = 24           # Rollout length
    uncertainty_penalty_weight: float = -0.0  # Uncertainty penalty
    use_bootstrap_sampling: bool = True   # Bootstrap for diversity
```

### PPO Configuration

```python
@configclass
class RslRlMbrlPpoAlgorithmCfg:
    # Standard PPO
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 1e-3
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
```

---

## System Dynamics Ensemble Details

### Architecture

```
SystemDynamicsEnsemble (N=5 members)
├── State Branch (per member)
│   ├── RNN Encoder: GRU(57 → 256, layers=2)
│   │   └── Input: concat([state(45), action(12)])
│   │
│   └── State Head: MLP(256 → 128 → 45)
│       ├── Mean: Residual prediction (Δstate)
│       └── LogStd: Aleatoric uncertainty (bounded)
│
└── Auxiliary Branch (per member)
    ├── RNN Encoder: GRU(separate)
    │
    └── Auxiliary Heads
        ├── Contacts: MLP → 8 dims (foot/thigh)
        └── Termination: MLP → 1 dim
```

### Forward Pass

**Input:**
- `x_state_batch`: (batch, history=32, state_dim=45)
- `x_action_batch`: (batch, history=32, action_dim=12)

**Output:**
- `state_means`: (batch, 45) - Predicted next state
- `aleatoric_uncertainty`: (batch,) - Data noise
- `epistemic_uncertainty`: (batch,) - Model disagreement
- `contacts`: (batch, 8) - Contact probabilities
- `termination`: (batch, 1) - Termination probability

### Training Loss

For each ensemble member (with bootstrap sampling):

```python
# State prediction loss
if regression_loss_type == "nll":
    state_loss = GaussianNLL(state_mean, state_target, state_std^2)
else:
    state_loss = MSE(state_mean, state_target)

# Auxiliary losses
contact_loss = BCEWithLogits(contact_pred, contact_target)
termination_loss = BCEWithLogits(term_pred, term_target)

# Bound loss (regularize logstd)
bound_loss = mean(max_logstd) - mean(min_logstd)

# Total loss
total_loss = state_loss + contact_loss + termination_loss + bound_loss
```

**Bootstrap sampling**: Each ensemble member sees different random samples from the same batch → diversity → epistemic uncertainty.

### Autoregressive Training

Multi-step forecasting to handle compounding errors:

```python
for i in range(forecast_horizon=8):
    # Predict next state
    state_mean, state_std = model.forward(...)
    
    # Sample from prediction (adds robustness)
    sampled_state = randn() * state_std + state_mean
    
    # Use as input for next prediction
    state_history = cat([state_history[:, 1:], sampled_state.unsqueeze(1)])
```

---

## Uncertainty Quantification

### Aleatoric Uncertainty

**Definition**: Inherent randomness in the data (irreducible)

**Computation**:
```python
aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
# Dimension flow: (ensemble, batch, state_dim) → (batch, state_dim) → (batch,)
```

**Interpretation**: Average of predicted standard deviations across ensemble

**Captures**: Sensor noise, contact randomness, etc.

### Epistemic Uncertainty

**Definition**: Model uncertainty due to limited data (reducible with more data)

**Computation**:
```python
epistemic_uncertainty = state_means.std(dim=0).sum(dim=1) if ensemble_size > 1 else 0
# Dimension flow: (ensemble, batch, state_dim) → (batch, state_dim) → (batch,)
```

**Interpretation**: Standard deviation of mean predictions (ensemble disagreement)

**Captures**: Out-of-distribution states, model uncertainty

### Usage in Imagination

Uncertainty penalty encourages conservative exploration:

```python
reward_imagination = reward_base - uncertainty_penalty_weight * epistemic_uncertainty
# uncertainty_penalty_weight = -0.1 typical
```

**Effect**: Policy avoids high-uncertainty regions (safer exploration)

---

## Research: Ensemble Improvements

### Current Limitations

1. **Simple bootstrap may be insufficient** - Only ~37% data difference per member
2. **No diversity regularization** - Models can converge to similar solutions
3. **No calibration metrics** - Uncertainty not explicitly calibrated to errors

### Proposed Improvements

#### 1. Randomized Priors (HIGH PRIORITY)

**Concept**: Initialize each ensemble member with different random seeds

**Implementation** (very simple):

```python
# In SystemDynamicsEnsemble._init_networks()
for i in range(self.ensemble_size):
    torch.manual_seed(42 + i * 1000)  # Different seed per member
    self.state_heads.append(MLPStateHead(...))
torch.manual_seed(torch.initial_seed())  # Reset global seed
```

**Expected benefits**:
- ✅ Increased diversity without changing training
- ✅ Better epistemic uncertainty in OOD regions
- ✅ Prevents mode collapse
- ✅ Minimal overhead

**Reference**: *Osband et al. 2018 - "Randomized Prior Functions for Deep RL"*

#### 2. Diversity Loss

Explicitly penalize ensemble members for making similar predictions:

```python
def compute_diversity_loss(state_means):
    # state_means: (ensemble_size, batch, state_dim)
    diversity_loss = 0
    for i in range(ensemble_size):
        for j in range(i+1, ensemble_size):
            similarity = F.cosine_similarity(
                state_means[i].flatten(1),
                state_means[j].flatten(1),
                dim=1
            ).mean()
            diversity_loss -= similarity  # Negative = encourage diversity
    
    return diversity_loss / (ensemble_size * (ensemble_size - 1) / 2)

# Add to total loss in mbpo_ppo.py
loss = state_loss + ... + 0.01 * diversity_loss
```

#### 3. Uncertainty Calibration

Compute calibration metrics:

```python
def compute_calibration_error(predictions, targets, uncertainties):
    errors = (predictions - targets).abs()
    return F.mse_loss(uncertainties, errors.mean(dim=-1))
```

Add to training loop to monitor calibration.

### Research Roadmap

**Phase 1: Improve Diversity** (2-3 weeks)
1. Implement randomized priors
2. Add diversity loss
3. Compare epistemic uncertainty distributions

**Phase 2: Calibration** (2-3 weeks)
1. Implement calibration metrics
2. Tune uncertainty bounds
3. Validate on test set

**Phase 3: Advanced Features** (4-6 weeks)
1. Multi-modal failure prediction
2. Time-to-failure estimation
3. Real robot validation

See `ensemble_uncertainty.md` for full research notes (753 lines of detailed analysis).

---

## File Locations

### Core Algorithm Implementations

**World Model** (`isaac-sim/overlay/rsl_rl_rwm/rsl_rl/`):

| File | Purpose | Key Functions |
|------|---------|---------------|
| `modules/system_dynamics.py` | Ensemble dynamics model | `forward()`, `compute_loss()` |
| `algorithms/mbpo_ppo.py` | MBPOPPO algorithm | `update_system_dynamics()` |
| `runners/mbpo_on_policy_runner.py` | Training loop | `learn()`, `imagine()` |
| `modules/architectures/rnn.py` | RNN memory management | `RNNBase`, `Memory` |
| `modules/architectures/mlp.py` | State/auxiliary heads | `MLPStateHead`, `MLPAuxiliaryHead` |

### Environment & Task Definitions

**RWM Extension** (`robotic_world_model/source/mbrl/mbrl/`):

| File | Purpose |
|------|---------|
| `envs/manager_based_mbrl_env.py` | Base imagination environment |
| `tasks/.../flat_env_cfg.py` | Environment configuration |
| `tasks/.../rsl_rl_ppo_cfg.py` | Algorithm configuration |
| `tasks/.../anymal_d_manager_based_mbrl_env.py` | ANYmal D MBRL environment |

### Training Scripts

| File | Purpose |
|------|---------|
| `scripts/reinforcement_learning/rsl_rl/train.py` | Online training |
| `scripts/reinforcement_learning/rsl_rl/play.py` | Policy evaluation |
| `scripts/reinforcement_learning/rsl_rl/visualize.py` | World model visualization |
| `scripts/reinforcement_learning/model_based/train.py` | Offline training |

---

## Development: Modifying World Model

### Change Ensemble Size

```python
# In rsl_rl_ppo_cfg.py
system_dynamics = RslRlSystemDynamicsCfg(
    ensemble_size=7,  # was 5
)
```

No code changes needed - automatically creates 7 heads.

### Modify RNN Architecture

```python
architecture_config: Dict = {
    "type": "rnn",
    "rnn_type": "lstm",        # was "gru"
    "rnn_num_layers": 3,       # was 2
    "rnn_hidden_size": 512,    # was 256
}
```

### Change Uncertainty Computation

Edit `system_dynamics.py:125-126`:

```python
# Current: sum of disagreements
epistemic_uncertainty = state_means.std(dim=0).sum(dim=1)

# Alternative: max disagreement
epistemic_uncertainty = state_means.std(dim=0).max(dim=1)[0]

# Alternative: weighted by importance
weights = torch.tensor([1.0, 1.0, 0.5, ...])  # Per state dimension
epistemic_uncertainty = (state_means.std(dim=0) * weights).sum(dim=1)
```

### Modify Loss Function

Edit `system_dynamics.py:270-299`:

```python
def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target):
    # Add custom loss
    custom_loss = my_custom_loss_fn(state_mean_pred, state_target)
    return custom_loss, sequence_loss
```

---

## Quick Reference Tables

### Observation Dimensions

| Component | Dims | Description |
|-----------|------|-------------|
| **Policy observations** | 48 | For actor-critic network |
| `base_lin_vel` | 3 | Body linear velocity |
| `base_ang_vel` | 3 | Body angular velocity |
| `projected_gravity` | 3 | Gravity in body frame |
| `velocity_commands` | 3 | Desired velocity |
| `joint_pos` | 12 | Joint positions |
| `joint_vel` | 12 | Joint velocities |
| `last_actions` | 12 | Previous actions |
| **World model state** | 45 | For dynamics prediction |
| (same as policy) | 42 | Minus commands, plus torques |
| `joint_torque` | 12 | Applied joint torques |

### Training Performance (A100-40GB)

| Environments | Data Collection | Model Training | Imagination | PPO Update | Total/iter |
|--------------|----------------|----------------|-------------|------------|------------|
| 4096 | ~5 sec | ~30 sec | ~10 sec | ~5 sec | ~50 sec |
| 2048 | ~3 sec | ~20 sec | ~5 sec | ~3 sec | ~30 sec |
| 1024 | ~2 sec | ~10 sec | ~3 sec | ~2 sec | ~15 sec |

**Full training (5000 iterations)**:
- 4096 envs: ~70 hours
- 2048 envs: ~42 hours
- 1024 envs: ~21 hours

---

## Resources

### Papers

- **RWM**: [arXiv:2501.10100](https://arxiv.org/abs/2501.10100)
- **U-RWM**: [arXiv:2504.16680](https://arxiv.org/abs/2504.16680)
- **Randomized Priors**: [Osband et al. 2018, arXiv:1806.03335](https://arxiv.org/abs/1806.03335)

### Code

- **RSL_RL RWM**: [github.com/leggedrobotics/rsl_rl_rwm](https://github.com/leggedrobotics/rsl_rl_rwm)
- **Robotic World Model**: [github.com/leggedrobotics/robotic_world_model](https://github.com/leggedrobotics/robotic_world_model)
- **Isaac Lab**: [isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab/)

### Full Documentation

- **ensemble_uncertainty.md**: Detailed research notes on ensemble improvements (753 lines)
- **U-RWM_TRAINING_WORKFLOW.md**: Complete RWM vs U-RWM analysis (1131 lines)
- Both files remain in `robotic_world_model/` for comprehensive technical reference

---

**For installation, see** `/INSTALLATION.md`  
**For training workflows, see** `TRAINING.md`  
**For code structure, see** `README.md`
