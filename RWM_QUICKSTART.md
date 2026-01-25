# RWM Training Quick Start Guide

Practical guide for reproducing Robotic World Model paper results on Snellius HPC.

---

## Training Overview

RWM (Robotic World Model) training has **two phases**:

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRETRAIN (World Model + Policy)                       │
│  ─────────────────────────────────────────                      │
│  • Train PPO policy from scratch                                │
│  • Collect real experience → replay buffer                      │
│  • Train world model (RNN) on collected data                    │
│  • NO imagination rollouts (imagination disabled)               │
│  • Duration: ~5-6 hours (2500 iterations)                       │
│  • Output: model_2500.pt (policy + world model)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: FINETUNE (Model-Based PPO)                            │
│  ────────────────────────────────────                           │
│  • Load pretrained world model + policy                         │
│  • Generate imagination rollouts (4096 envs × 100 steps)        │
│  • Train policy on BOTH real + imagined experience              │
│  • Continue updating world model with new data                  │
│  • Duration: ~10-15 hours (2500 iterations)                     │
│  • Output: Finetuned policy with better sample efficiency       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start Commands

### Phase 1: Pretrain World Model (Classic RWM)

```bash
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Submit pretrain job
sbatch slurm_train_rwm_benchmark.sh

# Monitor progress
tail -f logs/slurm/rwm-benchmark-<JOB_ID>.out
```

**What it runs:**
```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
  --num_envs 4096 \
  --max_iterations 2500 \
  --headless
```

### Phase 2: Finetune with MBPPO

```bash
# 1. Edit slurm_finetune_rwm.sh and update PRETRAIN_DIR:
#    PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-24_17-10-17"

# 2. Submit finetune job
sbatch slurm_finetune_rwm.sh
```

### Alternative: U-RWM Ensemble Training

For uncertainty-aware training with 5-model ensemble:

```bash
sbatch slurm_pretrain_rwm_ensemble.sh
```

---

## Available SLURM Scripts

| Script | Purpose | Duration |
|--------|---------|----------|
| `slurm_train_rwm_benchmark.sh` | Phase 1: Pretrain WM (single model) | ~10-12h |
| `slurm_finetune_rwm.sh` | Phase 2: MBPPO finetuning | ~15-20h |
| `slurm_pretrain_rwm_ensemble.sh` | Pretrain with 5-model ensemble | ~15-20h |

---

## Configuration Files

### Where to Set Parameters

**1. Agent/Algorithm Config** (main training parameters):
```
source/mbrl/mbrl/tasks/.../config/anymal_d/agents/rsl_rl_ppo_cfg.py
```

Key classes:
- `AnymalDFlatPPOPretrainRunnerCfg` - Pretrain settings
- `AnymalDFlatPPOFinetuneRunnerCfg` - Finetune settings

**2. Environment Config** (observations, rewards):
```
source/mbrl/mbrl/tasks/.../config/anymal_d/flat_env_cfg.py
```

**3. CLI Overrides** (runtime):
```bash
--num_envs 4096
--max_iterations 5000
agent.system_dynamics.ensemble_size=5  # Hydra override
```

### Key Parameters

| Parameter | Pretrain | Finetune | Description |
|-----------|----------|----------|-------------|
| `imagination.num_envs` | 0 | 4096 | Imagination rollout environments |
| `imagination.num_steps` | 0 | 100 | Imagination rollout length |
| `system_dynamics.ensemble_size` | 1 | 1 | Number of WM models (5 for U-RWM) |
| `system_dynamics_warmup_iterations` | 0 | 500 | Iterations before imagination |
| `uncertainty_penalty_weight` | 0.0 | 0.0 | Penalty for high uncertainty (-0.1 typical) |

---

## Registered Tasks

| Task ID | Purpose | Runner |
|---------|---------|--------|
| `Template-Isaac-Velocity-Flat-Anymal-D-Init-v0` | Model-free PPO baseline | `OnPolicyRunner` |
| `Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0` | **Phase 1: Pretrain WM** | `MBPOOnPolicyRunner` |
| `Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0` | **Phase 2: MBPPO finetune** | `MBPOOnPolicyRunner` |
| `Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0` | Visualize WM predictions | Special |
| `Isaac-Velocity-Flat-Anymal-D-Play-v0` | Evaluate trained policy | Play mode |

---

## Training Outputs

Checkpoints saved to: `logs/rsl_rl/anymal_d_flat/<timestamp>/`

```
logs/rsl_rl/anymal_d_flat/2026-01-24_17-10-17/
├── model_0.pt              # Initial checkpoint
├── model_500.pt            # Periodic checkpoints
├── model_1000.pt
├── ...
├── model_2500.pt           # Final checkpoint (includes WM + policy)
├── params/
│   ├── env.yaml            # Environment config snapshot
│   └── agent.yaml          # Agent config snapshot
└── events.out.tfevents.*   # TensorBoard logs
```

**Note**: The world model is saved inside `model_*.pt`, not as a separate file.

---

## Monitoring Training

### Key Metrics to Watch

**Phase 1 (Pretrain):**
```
Mean reward:              Should improve from -0.4 → positive
Mean episode length:      Should increase (robot survives longer)
System Dynamics/state_loss:  Should decrease (WM learning)
Metrics/error_vel_xy:     Should decrease (tracking improves)
```

**Phase 2 (Finetune):**
```
Train/mean_reward_imagination:  Reward from imagined rollouts
Model Based/epistemic_uncertainty:  Should be low if WM is good
```

### Commands

```bash
# Watch SLURM output
tail -f logs/slurm/rwm-benchmark-<JOB_ID>.out

# TensorBoard (if available)
tensorboard --logdir logs/rsl_rl/anymal_d_flat/

# Check GPU usage (SSH to compute node)
squeue -u $USER  # Get node name
ssh <node>
nvidia-smi -l 1
```

---

## Evaluation

### Play Trained Policy

```bash
# Inside container or with proper environment
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-Play-v0 \
  --checkpoint logs/rsl_rl/anymal_d_flat/<timestamp>/model_2500.pt \
  --num_envs 32

# With video recording
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Anymal-D-Play-v0 \
  --checkpoint model_2500.pt \
  --video --video_length 400 --headless
```

### Visualize World Model

Compare real robot vs imagination side-by-side:

```bash
python scripts/reinforcement_learning/rsl_rl/visualize.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \
  --checkpoint model_2500.pt \
  --system_dynamics_load_path model_2500.pt \
  --video --headless
```

---

## Typical Training Timeline

| Time | Iteration | What's Happening |
|------|-----------|------------------|
| 0h | 0 | Random policy, WM random |
| 1h | ~400 | Robot learning to stand |
| 2h | ~800 | Basic locomotion emerging |
| 3h | ~1200 | Decent walking, WM accurate |
| 5h | ~2000 | Good tracking performance |
| 6h | 2500 | Pretrain complete |

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| CUDA OOM | Too many envs | Reduce `--num_envs` to 2048 |
| Reward stuck negative | Learning rate too high | Reduce `policy_learning_rate` |
| WM loss not decreasing | Buffer too small | Increase `replay_buffer_size` |
| Finetune fails to load | Wrong checkpoint path | Verify `PRETRAIN_DIR` in script |
| Job killed (time limit) | Training too slow | Request more time or fewer iterations |

---

## File Reference

```
robotic_world_model/
├── slurm_train_rwm_benchmark.sh      # Phase 1: Pretrain (single model)
├── slurm_finetune_rwm.sh             # Phase 2: MBPPO finetuning
├── slurm_pretrain_rwm_ensemble.sh    # Ensemble pretrain (U-RWM)
│
├── scripts/reinforcement_learning/
│   ├── rsl_rl/
│   │   ├── train.py                  # Main training script
│   │   ├── play.py                   # Policy evaluation
│   │   └── visualize.py              # WM visualization
│   └── model_based/
│       └── train.py                  # Offline training (U-RWM Stage 2)
│
├── source/mbrl/mbrl/
│   ├── rl/rsl_rl/rl_cfg.py           # Config schema definitions
│   └── tasks/.../config/anymal_d/
│       ├── __init__.py               # Task registration
│       ├── flat_env_cfg.py           # Environment configs
│       └── agents/rsl_rl_ppo_cfg.py  # Algorithm configs
│
└── logs/
    ├── slurm/                        # SLURM job outputs
    └── rsl_rl/anymal_d_flat/         # Training checkpoints
```

---

## Next Steps After Training

1. **Evaluate** pretrain policy to establish baseline
2. **Run finetune** (Phase 2) to improve with imagination
3. **Compare** pretrain vs finetune performance
4. **Visualize** world model predictions
5. **Export** policy for deployment (TorchScript)

---

## Your Successful Run

Your pretrain run is at:
```
logs/rsl_rl/anymal_d_flat/2026-01-24_17-10-17/
```

To finetune from this:
1. Edit `slurm_finetune_rwm.sh`
2. Set `PRETRAIN_DIR="logs/rsl_rl/anymal_d_flat/2026-01-24_17-10-17"`
3. Run `sbatch slurm_finetune_rwm.sh`

---

## Paper Hyperparameters (Tables S10 & S11)

These are the hyperparameters from the official RWM paper supplementary materials.

### Table S10: RWM (World Model) Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `step_time` | 0.02s | Simulation step time |
| `max_iterations` | 2500 | Training iterations for pretrain |
| `learning_rate` | 1e-4 | World model learning rate |
| `weight_decay` | 1e-5 | Weight regularization |
| `batch_size` | 1024 | Batch size for WM training |
| `history_horizon` | 32 | Steps of history for WM input |
| `forecast_horizon` | 8 | Steps to predict ahead |
| `forecast_decay` | 1.0 | Decay for multi-step loss |

### Table S11: MBPO-PPO (Policy) Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `imagination_envs` | 4096 | Number of imagination environments |
| `imagination_steps` | 100 | Rollout length in imagination |
| `learning_rate` | 0.001 | Policy learning rate |
| `learning_epochs` | 5 | PPO epochs per update |
| `mini_batches` | 4 | Mini-batches per epoch |
| `KL_target` | 0.01 | Target KL divergence |
| `gamma` | 0.99 | Discount factor |
| `clip` | 0.2 | PPO clipping parameter |
| `entropy_coef` | 0.005 | Entropy regularization |

### How to Apply These via Hydra Overrides

```bash
# Pretrain (Phase 1) - key parameters
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
  --max_iterations 2500 \
  --num_envs 4096 \
  agent.algorithm.system_dynamics_learning_rate=1e-4 \
  agent.algorithm.system_dynamics_weight_decay=1e-5 \
  agent.algorithm.system_dynamics_mini_batch_size=1024 \
  agent.algorithm.policy_learning_rate=0.001

# Finetune (Phase 2) - key parameters  
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
  --max_iterations 2500 \
  agent.imagination.num_envs=4096 \
  agent.imagination.num_steps=100 \
  agent.algorithm.policy_learning_rate=0.001
```

### Notes on Paper Parameters

1. **imagination_steps=100**: The paper uses 100-step imagination rollouts, which is significantly longer than the default 24 steps. This provides more training signal per iteration.

2. **imagination_envs=4096**: Paper uses 4096 imagination envs (not 8192). Combined with 100 steps, this gives 409,600 imagined transitions per iteration.

3. **max_iterations=2500**: Paper trains for 2500 iterations (not 5000). With 4096 real envs, this is ~10M real environment steps.

4. **learning_rate difference**: WM uses 1e-4, policy uses 1e-3 (10x higher for policy).
