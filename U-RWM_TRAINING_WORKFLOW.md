# RWM vs U-RWM: Complete Analysis & Training Workflow

## Overview

This document explains:
1. **How RWM and U-RWM are connected** (same codebase, different modes)
2. **How they are separated** (configuration-based differences)
3. **How to train world models** for U-RWM (two-stage process)
4. **Implementation details** (where the code differs and where it's shared)

---

## Table of Contents

1. [RWM vs U-RWM: Conceptual Differences](#rwm-vs-u-rwm-conceptual-differences)
2. [Implementation: Same Codebase, Different Modes](#implementation-same-codebase-different-modes)
3. [U-RWM Training Workflow](#u-rwm-training-workflow)
4. [How to Train World Models](#how-to-train-world-models)
5. [Your Development Options](#your-development-options)

---

# Part 1: RWM vs U-RWM Comparison

## RWM vs U-RWM: Conceptual Differences

### **Research Papers**

**RWM (Robotic World Model)**
- **Paper**: [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)
- **Algorithm**: MBPO-PPO (Model-Based Policy Optimization with PPO)
- **Key Innovation**: Neural network simulator for robust policy optimization
- **Training Paradigm**: Online learning (continuous environment interaction)
- **Uncertainty Modeling**: ❌ No ensemble, no epistemic uncertainty

**U-RWM (Uncertainty-Aware Robotic World Model)**
- **Paper**: [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)
- **Algorithm**: MOPO-PPO (Model-based Offline Policy Optimization with PPO)
- **Key Innovation**: Ensemble uncertainty for safe offline RL
- **Training Paradigm**: Offline learning (no environment during policy training)
- **Uncertainty Modeling**: ✅ 5-member ensemble, epistemic uncertainty active

### **Algorithmic Differences**

| Aspect | RWM (MBPO-PPO) | U-RWM (MOPO-PPO) |
|--------|----------------|------------------|
| **World Model** | Single neural network | 5-member ensemble |
| **Uncertainty** | Only aleatoric (data noise) | Aleatoric + Epistemic (model disagreement) |
| **Training** | Online (policy + model updated together) | Offline (frozen model, policy only) |
| **Exploration** | Standard PPO exploration | Uncertainty-penalized exploration |
| **Use Case** | Sample-efficient online RL | Safe offline RL, real robot deployment |
| **Safety** | No uncertainty awareness | Uncertainty-aware, avoids OOD states |
| **Computational Cost** | Lower (single model) | Higher (5x model inference) |
| **Data Requirements** | Generates own data | Requires pre-collected dataset |

### **Mathematical Formulation**

#### **RWM: Standard Model-Based RL**

**World Model** (single model):
```
s_{t+1} = f_θ(s_t, a_t) + ε_aleatoric
where ε_aleatoric ~ N(0, σ²) (learned data noise)
```

**Policy Objective**:
```
max_π E[Σ r_t]
where experience comes from: real env + imagination rollouts
```

**Imagination Reward**:
```
r_imagination = r_base(s_t, a_t)
(no uncertainty penalty)
```

#### **U-RWM: Uncertainty-Aware Offline RL**

**World Model** (ensemble of K=5 models):
```
s_{t+1}^(i) = f_θ_i(s_t, a_t) + ε_aleatoric^(i)  for i = 1,...,5
```

**Uncertainty Decomposition**:
```
Aleatoric:  σ_aleatoric = mean_i(σ_i)  (average predicted noise)
Epistemic:  σ_epistemic = std_i(μ_i)   (model disagreement)
```

**Policy Objective** (uncertainty-penalized):
```
max_π E[Σ (r_t - λ * σ_epistemic(s_t))]
where:
  - λ > 0 (uncertainty penalty weight)
  - All experience from imagination (no real env)
```

**Imagination Reward**:
```
r_imagination = r_base(s_t, a_t) - λ * σ_epistemic(s_t)
                                    ↑
                        penalizes uncertain states
```

**Key Insight**: The uncertainty penalty makes the policy **conservative**, avoiding states where the ensemble disagrees (likely out-of-distribution).

---

## Implementation: Same Codebase, Different Modes

### **Critical Discovery**

RWM and U-RWM are **NOT separate implementations**. They share the **exact same codebase** with **configuration-based** mode switching.

### **Shared Core Components**

All located in: `isaac-sim/overlay/rsl_rl_rwm/rsl_rl/`

#### **1. System Dynamics Ensemble** (`modules/system_dynamics.py`)

```python
class SystemDynamicsEnsemble(nn.Module):
    def __init__(self, ..., ensemble_size: int = 1, ...):
        # Works for BOTH RWM (ensemble_size=1) and U-RWM (ensemble_size=5)
        self.ensemble_size = ensemble_size
        
        # Create ensemble_size independent models
        self.state_heads = nn.ModuleList([
            MLPStateHead(...) for _ in range(self.ensemble_size)
        ])
        
        self.auxiliary_heads = nn.ModuleList([
            MLPAuxiliaryHead(...) for _ in range(self.ensemble_size)
        ])
    
    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        # ... forward through all ensemble members ...
        
        # Aleatoric uncertainty (always computed)
        aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
        
        # Epistemic uncertainty (only if ensemble_size > 1)
        epistemic_uncertainty = (
            state_means.std(dim=0).sum(dim=1)  # Model disagreement
            if self.ensemble_size > 1          # ← U-RWM mode
            else torch.zeros(..., device=self.device)  # ← RWM mode
        )
        
        return ..., aleatoric_uncertainty, epistemic_uncertainty, ...
```

**Key Implementation Detail**:
- **RWM** (`ensemble_size=1`): `epistemic_uncertainty` is always **zero**
- **U-RWM** (`ensemble_size=5`): `epistemic_uncertainty` reflects ensemble disagreement
- **Same code**, different behavior based on `ensemble_size`!

#### **2. MBPO-PPO Algorithm** (`algorithms/mbpo_ppo.py`)

```python
class MBPOPPO(PPO):
    """Unified algorithm for both RWM and U-RWM"""
    
    def __init__(self, system_dynamics: SystemDynamicsEnsemble, ...):
        self.system_dynamics = system_dynamics  # Works with any ensemble_size
        self.system_replay_buffer = ReplayBuffer(...)
        self.system_dynamics_optimizer = optim.Adam(...)
    
    def update_system_dynamics(self):
        """Trains all ensemble members with bootstrap sampling"""
        for system_state_batch, ... in system_generator:
            self.system_dynamics.reset()
            
            # Bootstrap sampling creates ensemble diversity
            state_loss, ... = self.system_dynamics.compute_loss(
                system_state_batch, ..., 
                bootstrap=True  # ← Key for ensemble diversity!
            )
            
            # Update all ensemble members
            self.system_dynamics_optimizer.zero_grad()
            state_loss.backward()
            self.system_dynamics_optimizer.step()
```

**Bootstrap Sampling** (creates ensemble diversity):
```python
# In system_dynamics.py:138-150
def compute_loss(self, state_batch, action_batch, ..., bootstrap=True):
    for i in range(self.ensemble_size):
        if bootstrap:
            # Each model sees DIFFERENT random samples
            ids = torch.randint(0, batch_size, (batch_size,))
        else:
            ids = torch.arange(0, batch_size)
        
        # Train model i on its unique data sample
        state_loss, ... = self.compute_state_loss(
            self.state_heads[i], 
            state_batch[ids],    # Different data per model!
            action_batch[ids]
        )
```

**Result**: Even with the same dataset, each model learns slightly different dynamics → disagreement → epistemic uncertainty!

#### **3. Online Runner** (`runners/mbpo_on_policy_runner.py`)

```python
class MBPOOnPolicyRunner(OnPolicyRunner):
    """Supports both RWM and U-RWM in online mode"""
    
    def learn(self, num_learning_iterations):
        for it in range(start_iter, tot_iter):
            # 1. Collect real environment data
            for i in range(self.num_steps_per_env):
                actions = self.alg.act(obs)
                obs, rewards, dones, extras = self.env.step(actions)  # REAL ENV
                self.alg.fill_history_buffer(obs)
            
            # 2. Update system dynamics (works for ensemble_size=1 or 5)
            mean_system_state_loss, ... = self.alg.update_system_dynamics()
            
            # 3. Generate imagination rollouts (if enabled)
            if self.num_imagination_envs > 0:
                imagination_obs, ... = self.imagine()
                loss_dict = self.alg.update(imagination=True)
            else:
                loss_dict = self.alg.update()  # Standard PPO
```

---

### **Configuration-Based Separation**

The **ONLY** differences between RWM and U-RWM are **configuration parameters**:

#### **RWM Configuration** (Online, No Ensemble)

**File**: `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py`

```python
@configclass
class AnymalDFlatPPOPretrainRunnerCfg(AnymalDFlatPPORunnerCfg):
    class_name: str = "MBPOOnPolicyRunner"
    
    # ========== KEY DIFFERENCE #1: No Ensemble ==========
    system_dynamics = RslRlSystemDynamicsCfg(
        ensemble_size=1,              # ← Single model (NO ensemble)
        history_horizon=32,
        architecture_config = {
            "type": "rnn",            # Same RNN architecture
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
        },
        freeze_auxiliary=False,
    )
    
    # ========== KEY DIFFERENCE #2: No Imagination Initially ==========
    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,                   # ← Disabled during pretrain
        num_steps_per_env=0,
        uncertainty_penalty_weight=-0.0,  # ← No uncertainty penalty
    )

@configclass
class AnymalDFlatPPOFinetuneRunnerCfg(AnymalDFlatPPOPretrainRunnerCfg):
    # ========== Finetune: Enable Imagination ==========
    def __post_init__(self):
        super().__post_init__()
        self.imagination.num_envs = 8192       # Enable imagination
        self.imagination.num_steps_per_env = 24
        self.imagination.uncertainty_penalty_weight = -0.0  # Still no penalty
```

**Training Mode**: **Online** (continuous real environment interaction via Isaac Lab)

#### **U-RWM Configuration** (Offline, With Ensemble)

**File**: `scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py`

```python
@dataclass
class AnymalDFlatConfig(BaseConfig):
    experiment_name: str = "offline"   # ← Explicitly offline
    
    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        # ========== KEY DIFFERENCE #1: Full Ensemble ==========
        ensemble_size: int = 5         # ← 5-member ensemble
        history_horizon: int = 32
        architecture_config: Dict = {
            "type": "rnn",             # Same RNN architecture
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
        }
        resume_path: str = "assets/models/pretrain_rnn_ens.pt"  # Pre-trained!
    
    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        # ========== KEY DIFFERENCE #2: Uncertainty Penalty Enabled ==========
        uncertainty_penalty_weight: float = -1.0  # ← Strong penalty
        command_resample_interval_range: List = [100, 120]
```

**Training Mode**: **Offline** (no real environment, pure imagination)

---

### **Side-by-Side Comparison**

| Configuration Parameter | RWM (MBPO) | U-RWM (MOPO) | Impact |
|-------------------------|------------|--------------|--------|
| **ensemble_size** | **1** | **5** | Determines if epistemic uncertainty exists |
| **uncertainty_penalty_weight** | **-0.0** | **-1.0** | Whether to discourage uncertain states |
| **Training script** | `rsl_rl/train.py` | `model_based/train.py` | Online vs offline execution |
| **Config file** | `rsl_rl_ppo_cfg.py` | `anymal_d_flat_cfg.py` | Different parameter sets |
| **Training mode** | **Online** | **Offline** | Real env vs pure imagination |
| **Model updates** | **Continuous** | **None (frozen)** | Whether dynamics can improve |
| **Data source** | Live simulation | Static CSV + frozen model | Where experience comes from |
| **resume_path** | Optional | **Required** (pretrained ensemble) | Model initialization |

### **Code Paths**

#### **RWM Online Training**

```
scripts/reinforcement_learning/rsl_rl/train.py
    ↓
source/mbrl/mbrl/tasks/.../rsl_rl_ppo_cfg.py (ensemble_size=1)
    ↓
isaac-sim/overlay/rsl_rl_rwm/rsl_rl/runners/mbpo_on_policy_runner.py
    ↓
isaac-sim/overlay/rsl_rl_rwm/rsl_rl/algorithms/mbpo_ppo.py
    ↓
isaac-sim/overlay/rsl_rl_rwm/rsl_rl/modules/system_dynamics.py (ensemble_size=1)
```

#### **U-RWM Offline Training**

```
scripts/reinforcement_learning/model_based/train.py
    ↓
scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py (ensemble_size=5)
    ↓
scripts/reinforcement_learning/model_based/policy_training.py
    ↓
isaac-sim/overlay/rsl_rl_rwm/rsl_rl/algorithms/mbpo_ppo.py (SAME!)
    ↓
isaac-sim/overlay/rsl_rl_rwm/rsl_rl/modules/system_dynamics.py (ensemble_size=5, SAME!)
```

**Key Insight**: Both paths converge to the **same core algorithm and model classes**!

---

# Part 2: U-RWM Training Workflow

## Your Question

> "If U-RWM was developed to fully work offline, how do I train the world models? I read they should be trainable offline in a supervised manner on a dataset? But the pretrain world model script looks for online interaction doesn't it?"

**Excellent question!** You've identified a critical distinction. Let me clarify the complete workflow.

---

## The Complete U-RWM Workflow

### **Two Separate Stages**

U-RWM (Uncertainty-aware Robotic World Model) involves **TWO distinct stages** with different training paradigms:

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: World Model Training (ONLINE or SUPERVISED)      │
│  - Uses real environment OR logged dataset                  │
│  - Trains ensemble dynamics model                           │
│  - Output: pretrain_rnn_ens.pt (frozen ensemble)           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: Policy Training (PURE OFFLINE)                    │
│  - NO real environment interaction                           │
│  - Uses frozen ensemble from Stage 1                         │
│  - Trains policy purely in imagination                       │
│  - Output: Trained policy checkpoint                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Stage 1: World Model Training

### **Two Approaches Available**

The world model (ensemble) can be trained in **two different ways**:

#### **Approach A: Online Training (What the Pretrain Script Does)**

**Script**: `scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0`

**Process**:
```python
# Pseudocode for online world model training
for iteration in range(num_iterations):
    # 1. Collect data from REAL environment (Isaac Lab simulation)
    for step in range(num_steps_per_env):
        action = policy.act(observation)
        next_obs, reward, done = env.step(action)  # ← REAL ENVIRONMENT
        replay_buffer.add(obs, action, next_obs, reward, done)
    
    # 2. Train ensemble on collected data (SUPERVISED LEARNING)
    for batch in replay_buffer:
        for i in range(ensemble_size):
            # Bootstrap sampling for diversity
            if bootstrap:
                ids = torch.randint(0, batch_size, (batch_size,))
            
            # Supervised loss: predict next_state from (state, action)
            predicted_next_state = ensemble_member_i(state[ids], action[ids])
            loss = MSELoss(predicted_next_state, true_next_state[ids])
            loss.backward()
    
    # 3. Optionally: Update policy with PPO (side effect)
    policy.update(rollout_data)
```

**Key characteristics**:
- ✅ Uses Isaac Lab simulator for data collection
- ✅ Continuously collects new experience
- ✅ Trains ensemble in **supervised manner** on collected transitions
- ✅ Can optionally train a policy alongside (but policy is not the goal)
- ✅ Output: `pretrain_rnn_ens.pt` (trained ensemble checkpoint)

**Why it looks "online"**: Because it interacts with the simulator. BUT the world model training itself is **supervised learning** (predicting next states from current states and actions).

---

#### **Approach B: Supervised Training on Static Dataset** (Alternative)

**Script**: ❌ **NOT currently implemented in the repository!**

**What it would look like**:
```python
# Hypothetical supervised world model training script
# (not in the current repository)

# 1. Load pre-collected dataset
dataset = load_csv("assets/data/state_action_data_0.csv")
# Dataset contains: (state_t, action_t, next_state_t+1, reward, contact, termination)

# 2. Train ensemble in supervised manner
for epoch in range(num_epochs):
    for batch in dataset:
        for i in range(ensemble_size):
            # Bootstrap sampling
            ids = torch.randint(0, batch_size, (batch_size,))
            
            # Supervised prediction
            predicted_next_state = ensemble_member_i(state[ids], action[ids])
            loss = MSELoss(predicted_next_state, true_next_state[ids])
            loss.backward()

# 3. Save trained ensemble
save("pretrain_rnn_ens.pt", ensemble)
```

**Key characteristics**:
- ✅ Uses pre-collected CSV dataset (no simulator needed)
- ✅ Pure supervised learning (predict transitions)
- ✅ No environment interaction
- ✅ Output: `pretrain_rnn_ens.pt`

**Status**: ⚠️ This approach is **NOT implemented** in the current repository. The pretrained model `assets/models/pretrain_rnn_ens.pt` was likely trained using **Approach A** (online pretraining) and then provided as a checkpoint for offline policy training.

---

## Stage 2: Offline Policy Training (Pure Offline)

**Script**: `scripts/reinforcement_learning/model_based/train.py --task anymal_d_flat`

**Process**:
```python
# From scripts/reinforcement_learning/model_based/train.py

# 1. Load FROZEN pretrained ensemble
system_dynamics = SystemDynamicsEnsemble(...)
system_dynamics.load_state_dict(torch.load("assets/models/pretrain_rnn_ens.pt"))
system_dynamics.eval()  # Frozen - no training!

# 2. Load initial states dataset (for branching imagination)
dataset = load_csv("assets/data/state_action_data_0.csv")
# Only used to get initial states, not for training world model

# 3. Train policy PURELY in imagination
for iteration in range(num_iterations):
    # Sample initial states from dataset
    init_states = dataset.sample_batch(num_envs)
    
    # Run imagination rollouts using FROZEN ensemble
    for step in range(num_steps_per_env):
        action = policy.act(imagination_obs)
        
        # Step through LEARNED dynamics (NOT real env!)
        next_state, uncertainty = system_dynamics(state, action)
        reward = compute_reward(next_state) - penalty * uncertainty
        
        # Store imagined experience
        rollout_buffer.add(imagination_obs, action, reward, ...)
    
    # Update policy on imagined experience
    policy.update(rollout_buffer)

# NO WORLD MODEL TRAINING IN THIS STAGE!
```

**Key characteristics**:
- ❌ NO real environment interaction
- ❌ NO world model training (ensemble is FROZEN)
- ✅ Policy learns purely from imagined rollouts
- ✅ Uses static dataset only for initial states
- ✅ Epistemic uncertainty guides safe exploration in imagination

---

## Understanding the Workflow

### **What the README Says**

From `README.md` lines 137-138:
> - **Option 1: Train policy in imagination *online***, where additional environment interactions are continually collected using the latest policy to update the dynamics model (as implemented with RWM and MBPO-PPO)
> - **Option 2: Train policy in imagination *offline*** where no additional environment interactions are collected and the policy has to rely on the static dynamics model (as implemented with RWM-U and MOPO-PPO)

**Key insight**: The "offline" in U-RWM refers to **Stage 2** (policy training), NOT Stage 1 (world model training).

---

### **The Confusion Explained**

| Component | Training Mode | Uses Real Env? | Uses Dataset? |
|-----------|---------------|----------------|---------------|
| **World Model (Ensemble)** | **Supervised Learning** | Yes (Approach A) OR No (Approach B) | Yes (logged transitions) |
| **Policy (U-RWM)** | **Pure Offline RL** | ❌ No | Yes (for initial states) |

**The confusion arises because**:
1. The **world model can be trained online** (with real env) even though it learns via **supervised prediction**
2. The **policy is trained offline** (no real env) using the frozen world model
3. "Offline" in U-RWM primarily refers to the **policy training stage**, not necessarily the world model training

---

## Provided Assets

### **Pretrained World Model**

**File**: `assets/models/pretrain_rnn_ens.pt`

```bash
$ file assets/models/pretrain_rnn_ens.pt
assets/models/pretrain_rnn_ens.pt: ASCII text

$ cat assets/models/pretrain_rnn_ens.pt
version https://git-lfs.github.com/spec/v1
oid sha256:2ac8686c8c7736287853c0fe1438c379320068adf97be71609b5f1afb52c6e5a
size 25054639
```

**Status**: Stored with Git LFS (Large File Storage)
- **Actual size**: ~25 MB (contains full 5-member ensemble)
- **Likely trained using**: Online pretrain script (Approach A)
- **Purpose**: Ready-to-use frozen ensemble for offline policy training

---

### **Dataset for Policy Initialization**

**File**: `assets/data/state_action_data_0.csv`

```bash
$ wc -l assets/data/state_action_data_0.csv
10000 assets/data/state_action_data_0.csv
```

**Structure**: 10,000 timesteps with 65 columns
- Columns 0-44: State (45 dims)
- Columns 45-56: Action (12 dims)
- Columns 57-64: Extension + Contact + Termination (8 dims)

**Purpose**: Provides initial states for branching imagination rollouts during offline policy training

**Source**: Likely collected from:
1. Random policy rollouts, OR
2. Pretrained policy rollouts, OR
3. Mixture of exploration behaviors

---

## How to Train Your Own World Model

### **Option 1: Use Online Pretrain (Current Approach)**

This is what the repository currently supports:

```bash
# Train ensemble online using Isaac Lab simulation
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless
```

**What happens**:
1. Launches Isaac Lab simulator
2. Collects data with random/learning policy
3. Trains 5-member ensemble on collected data (supervised learning)
4. Saves checkpoint: `logs/rsl_rl/anymal_d_flat/<timestamp>_pretrain_rnn/model_<iter>.pt`

**Configuration for ensemble**:
```python
# In rsl_rl_ppo_cfg.py
system_dynamics = RslRlSystemDynamicsCfg(
    ensemble_size=5,  # ← Change from 1 to 5 for ensemble
    history_horizon=32,
    architecture_config={
        "type": "rnn",
        "rnn_type": "gru",
        "rnn_num_layers": 2,
        "rnn_hidden_size": 256,
    },
)
```

**Output**: Trained ensemble ready for offline policy training

---

### **Option 2: Train Supervised on CSV Dataset** (Not Implemented)

If you want to train the world model **purely from logged data** without running the simulator, you would need to implement this yourself:

```python
# Hypothetical supervised training script (YOU WOULD NEED TO WRITE THIS)

import torch
from rsl_rl.modules import SystemDynamicsEnsemble
import pandas as pd

# 1. Load CSV dataset
data = pd.read_csv("assets/data/state_action_data_0.csv", header=None)
states = torch.tensor(data.iloc[:, :45].values)  # State dims
actions = torch.tensor(data.iloc[:, 45:57].values)  # Action dims
next_states = torch.cat([states[1:], states[-1:]], dim=0)  # Shift by 1

# 2. Create ensemble
ensemble = SystemDynamicsEnsemble(
    state_dim=45,
    action_dim=12,
    ensemble_size=5,
    ...
)

# 3. Train with bootstrap sampling
optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for batch_idx in range(0, len(states), batch_size):
        batch_states = states[batch_idx:batch_idx+batch_size]
        batch_actions = actions[batch_idx:batch_idx+batch_size]
        batch_next_states = next_states[batch_idx:batch_idx+batch_size]
        
        # Compute loss with bootstrap
        loss = ensemble.compute_loss(
            batch_states, 
            batch_actions, 
            batch_next_states,
            bootstrap=True  # ← Key for ensemble diversity
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Save trained ensemble
torch.save({
    'system_dynamics_state_dict': ensemble.state_dict(),
    'iter': num_epochs,
}, "my_trained_ensemble.pt")
```

**Benefits**:
- ✅ No simulator required
- ✅ Can train on real robot logged data
- ✅ Reproducible (same dataset → same model)

**Challenges**:
- ❌ Requires high-quality logged dataset
- ❌ Need to collect dataset first (chicken-and-egg problem)
- ❌ Not currently implemented in repository

---

## Summary: The Two-Stage Process

### **For U-RWM Paper Results**

```
STAGE 1: World Model Training (Supervised Learning)
┌─────────────────────────────────────────────────────┐
│ Approach A (Current Repo):                         │
│ - Run online pretrain with Isaac Lab               │
│ - Ensemble learns to predict transitions           │
│ - Output: pretrain_rnn_ens.pt                      │
│                                                     │
│ Approach B (Not Implemented):                      │
│ - Load logged CSV dataset                          │
│ - Train ensemble supervised on dataset             │
│ - Output: pretrain_rnn_ens.pt                      │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
STAGE 2: Offline Policy Training (Pure Offline RL)
┌─────────────────────────────────────────────────────┐
│ - Load FROZEN ensemble from Stage 1                │
│ - Load initial states dataset                      │
│ - Train policy in imagination only                 │
│ - NO real environment                              │
│ - NO world model updates                           │
│ - Output: Trained policy                           │
└─────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### **1. "Offline" Refers to Policy Training, Not Necessarily World Model Training**

- ✅ **World model**: Can be trained online (with sim) OR offline (with dataset)
- ✅ **Policy (U-RWM)**: Always trained offline (no sim, frozen model)

### **2. World Model Training is Always Supervised**

Whether trained online or offline, the world model learns via **supervised learning**:
- **Input**: (state, action) pairs
- **Output**: Predicted next state
- **Loss**: MSE(predicted_next_state, true_next_state)
- **Diversity**: Bootstrap sampling creates ensemble disagreement

### **3. Current Repository Workflow**

```bash
# Step 1: Train ensemble (online pretrain with Isaac Lab)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless

# Step 2: Train policy offline (pure imagination)
python scripts/reinforcement_learning/model_based/train.py \
    --task anymal_d_flat
```

### **4. The Pretrained Model**

The provided `assets/models/pretrain_rnn_ens.pt`:
- ✅ Is a **Git LFS file** (25 MB actual size)
- ✅ Contains a **5-member ensemble**
- ✅ Was likely trained using **online pretrain** (Approach A)
- ✅ Is **frozen** during offline policy training
- ✅ Provides uncertainty quantification for safe offline RL

### **5. The Dataset**

The provided `assets/data/state_action_data_0.csv`:
- ✅ Contains **10,000 timesteps** of logged experience
- ✅ Is used **only for initial states** in offline policy training
- ✅ Is **NOT used to train the world model** in the current workflow
- ✅ Could be used to train a world model if you implement Approach B

---

# Part 3: Your Development Options

## What You Need for Your Research

Based on your goal to develop **uncertainty modeling** for the robotic world model, you now understand that RWM and U-RWM share the same codebase. This gives you several development paths:

---

## Option 1: Online Ensemble Training (Recommended)

**What it is**: Combine RWM's online training with U-RWM's ensemble uncertainty

**Configuration Changes**:
```python
# In source/mbrl/mbrl/tasks/.../rsl_rl_ppo_cfg.py

# Change 1: Enable ensemble
system_dynamics = RslRlSystemDynamicsCfg(
    ensemble_size=5,  # Instead of 1
    ...
)

# Change 2: Enable uncertainty penalty (finetune stage)
self.imagination.uncertainty_penalty_weight = -0.1  # Instead of -0.0
```

**Training**:
```bash
# Pretrain with ensemble
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless

# Finetune with imagination + uncertainty
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --headless \
    --checkpoint <pretrain_checkpoint> \
    --system_dynamics_load_path <ensemble_checkpoint>
```

**Characteristics**:
- ✅ Uncertainty quantification during training
- ✅ Continuous model improvement (not frozen)
- ✅ Online data collection from Isaac Lab
- ✅ Safer exploration via uncertainty penalty
- ✅ All your improvements from `ensemble_uncertainty.md` are compatible
- ⚠️  +50-100% training time vs single model

**Best for**: 
- Safety-critical online learning
- Continuous uncertainty monitoring
- Research on uncertainty-aware exploration
- Development and testing of ensemble improvements

---

## Option 2: Standard U-RWM Workflow (Offline)

**What it is**: Follow the U-RWM paper approach exactly

**Stage 1 - Train Ensemble**:
```bash
# Option A: Use provided pretrained ensemble
# (already in assets/models/pretrain_rnn_ens.pt)

# Option B: Train your own (requires changing ensemble_size to 5)
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless
```

**Stage 2 - Offline Policy Training**:
```bash
python scripts/reinforcement_learning/model_based/train.py \
    --task anymal_d_flat
```

**Characteristics**:
- ✅ Pure offline RL (no simulator during policy training)
- ✅ Frozen ensemble (no model updates)
- ✅ Uncertainty-penalized exploration in imagination
- ✅ Matches U-RWM paper exactly
- ❌ Ensemble doesn't improve after Stage 1
- ❌ Requires good pretrained model

**Best for**:
- Pure offline RL research
- Testing with frozen dynamics
- Real robot deployment (if you have logged data)
- Reproducing U-RWM paper results

---

## Option 3: Standard RWM (Baseline)

**What it is**: Original RWM without ensemble or uncertainty

**Configuration**: Default settings (ensemble_size=1)

**Training**:
```bash
# Pretrain
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless

# Finetune
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --headless
```

**Characteristics**:
- ✅ Fastest training (single model)
- ✅ Lowest computational cost
- ✅ Online learning with imagination rollouts
- ❌ No uncertainty quantification
- ❌ No safety monitoring
- ❌ No OOD detection

**Best for**:
- Baseline comparisons
- Quick prototyping
- Computational budget constraints
- Non-safety-critical applications

---

## Option 4: Pure Offline World Model Training (Custom)

**What it is**: Train ensemble from CSV dataset without simulator

**Status**: ⚠️ **NOT currently implemented** - you would need to write this

**What you'd need to implement**:

```python
# hypothetical_supervised_training.py (YOU WRITE THIS)

import torch
from rsl_rl.modules import SystemDynamicsEnsemble
import pandas as pd

# 1. Load dataset
data = pd.read_csv("your_dataset.csv")
states = torch.tensor(data.iloc[:, :45].values)
actions = torch.tensor(data.iloc[:, 45:57].values)
next_states = torch.cat([states[1:], states[-1:]], dim=0)

# 2. Create ensemble
ensemble = SystemDynamicsEnsemble(
    state_dim=45,
    action_dim=12,
    ensemble_size=5,
    history_horizon=32,
    architecture_config={"type": "rnn", ...},
    device="cuda",
)

# 3. Train with bootstrap sampling
optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for batch in dataloader:
        # Compute loss with bootstrap (creates diversity)
        loss = ensemble.compute_loss(
            batch_states, 
            batch_actions,
            batch_next_states,
            bootstrap=True  # ← KEY!
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Save
torch.save({
    'system_dynamics_state_dict': ensemble.state_dict(),
    'iter': num_epochs,
}, "my_trained_ensemble.pt")

# 5. Use for offline policy training
# (same as Option 2, Stage 2)
```

**Characteristics**:
- ✅ No simulator required for model training
- ✅ Can train on real robot logged data
- ✅ Reproducible (same data → same model)
- ❌ Requires implementation work
- ❌ Need high-quality dataset first
- ❌ Chicken-and-egg problem (where does dataset come from?)

**Best for**:
- Real robot data (if you have it)
- Reproducible research
- Environments where simulation is unavailable/expensive

---

## Comparison Matrix

| Approach | Ensemble | Online/Offline | Model Updates | Uncertainty | Complexity | Best Use Case |
|----------|----------|----------------|---------------|-------------|------------|---------------|
| **Option 1: Online Ensemble** | ✅ 5 | Online | ✅ Continuous | ✅ Active | Medium | **Your research** (uncertainty modeling) |
| **Option 2: U-RWM Workflow** | ✅ 5 | Offline (policy) | ❌ Frozen | ✅ Active | Medium | Offline RL research |
| **Option 3: Standard RWM** | ❌ 1 | Online | ✅ Continuous | ❌ None | Low | Baseline / fast prototyping |
| **Option 4: Custom Offline** | ✅ 5 | Pure offline | ❌ Frozen | ✅ Active | High | Real robot / custom datasets |

---

## Recommended Path for Your Research

Based on your stated goals to develop **uncertainty modeling improvements** (from `ensemble_uncertainty.md`):

### **Phase 1: Start with Option 1 (Online Ensemble)**

**Why**:
1. ✅ Gets uncertainty quantification working immediately
2. ✅ Continuous model improvement helps debug ensemble behavior
3. ✅ Easy to implement (just change config parameters)
4. ✅ Compatible with all your proposed improvements
5. ✅ Can test both online and offline modes later

**Steps**:
1. **Enable ensemble in pretrain config**:
   ```python
   # source/mbrl/mbrl/tasks/.../rsl_rl_ppo_cfg.py
   ensemble_size = 5  # Line 22
   ```

2. **Train ensemble online**:
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train.py \
       --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
       --headless
   ```

3. **Monitor epistemic uncertainty** in logs:
   - Should start high (~0.5-1.0)
   - Should decrease as model learns (~0.1-0.3)
   - Should NOT collapse to near-zero (<0.01)

4. **Implement Phase 1 improvements** (randomized priors):
   ```python
   # In system_dynamics.py:_init_networks()
   for i in range(self.ensemble_size):
       torch.manual_seed(42 + i * 1000)  # Different seed per member
       self.state_heads.append(MLPStateHead(...))
   torch.manual_seed(torch.initial_seed())  # Reset
   ```

### **Phase 2: Add Uncertainty Penalty**

**Enable in finetune config**:
```python
# source/mbrl/mbrl/tasks/.../rsl_rl_ppo_cfg.py:126
self.imagination.uncertainty_penalty_weight = -0.1
```

**Test different penalty values**:
- `-0.05`: Mild discouragement
- `-0.1`: Moderate (recommended)
- `-0.3`: Strong (conservative)
- `-1.0`: Very strong (offline-like)

### **Phase 3: Implement Advanced Features**

Follow your `ensemble_uncertainty.md` roadmap:
1. **Failure prediction** (2-3 weeks)
2. **Diversity regularization** (1-2 weeks)
3. **Uncertainty calibration** (1-2 weeks)

### **Phase 4: Compare All Modes**

Once you have improvements working, compare:
1. **Baseline RWM** (ensemble_size=1, no uncertainty)
2. **Online Ensemble** (your improved version)
3. **U-RWM Offline** (freeze your trained ensemble)

This gives you a complete story for your research!

---

## Summary Table: What Changes Between Modes

| Component | RWM | Online Ensemble | U-RWM Offline | Custom Offline |
|-----------|-----|-----------------|---------------|----------------|
| **Config File** | `rsl_rl_ppo_cfg.py` | `rsl_rl_ppo_cfg.py` | `anymal_d_flat_cfg.py` | Custom script |
| **ensemble_size** | 1 | **5** | 5 | 5 |
| **Training Script** | `rsl_rl/train.py` | `rsl_rl/train.py` | `model_based/train.py` | Custom script |
| **Stage 1 (Model)** | Online pretrain | **Online pretrain** | Online pretrain | **CSV dataset** |
| **Stage 2 (Policy)** | Online finetune | **Online finetune** | **Offline** | Offline |
| **uncertainty_penalty** | -0.0 | **-0.1** | -1.0 | -1.0 |
| **Model Updates** | ✅ Yes | **✅ Yes** | ❌ No | ❌ No |
| **Epistemic Uncertainty** | ❌ Always 0 | **✅ Active** | ✅ Active | ✅ Active |
| **Code Changes** | None | **Config only** | None | **New script** |

---

## Final Recommendations

### **For Your Uncertainty Modeling Research**

1. ✅ **Start with Online Ensemble** (Option 1)
   - Easiest to implement
   - Provides immediate uncertainty feedback
   - Allows iterative development

2. ✅ **Implement improvements incrementally**
   - Phase 1: Randomized priors
   - Phase 2: Failure prediction
   - Phase 3: Advanced techniques

3. ✅ **Test both online and offline modes**
   - Online: During development
   - Offline: For final evaluation (freeze your trained ensemble)

4. ✅ **Use RWM baseline for comparison**
   - Shows value of uncertainty modeling
   - Demonstrates safety improvements

### **Key Insight**

**You don't have to choose between RWM and U-RWM** - they're the same code! You can:
- Develop with **Online Ensemble** (best for research)
- Deploy as **U-RWM Offline** (best for real robots)
- Compare against **RWM Baseline** (shows your contribution)

All from the **same codebase** with **configuration changes only**!

---

## Conclusion

**To directly answer your question**:

> "How do I train the world models offline in a supervised manner on a dataset?"

**Answer**: The current repository **does NOT** implement pure offline world model training from CSV datasets. The pretrain script uses **online data collection** (Isaac Lab simulation) but trains the ensemble in a **supervised manner** (predicting transitions).

**The workflow is**:
1. **Stage 1** (World Model): Train ensemble **supervised** on data collected **online** from simulator
2. **Stage 2** (Policy): Train policy **offline** using **frozen** ensemble from Stage 1

**"Offline" in U-RWM** primarily refers to **Stage 2** (no environment during policy training), not necessarily Stage 1 (world model can be trained with or without environment, but always supervised on transition data).

If you want to train the world model **purely from a static CSV dataset** without running the simulator, you would need to implement that functionality yourself (see Option 2 above).
