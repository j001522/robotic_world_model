# Quick Reference: RWM vs U-RWM

## Key Insight

**RWM and U-RWM are the SAME codebase with different configurations!**

---

## What Changes Between Them

| Parameter | RWM | U-RWM |
|-----------|-----|-------|
| **ensemble_size** | 1 | 5 |
| **Training Mode** | Online | Offline (policy only) |
| **Epistemic Uncertainty** | Always 0 | Active |
| **uncertainty_penalty_weight** | -0.0 | -1.0 |
| **Config File** | `rsl_rl_ppo_cfg.py` | `anymal_d_flat_cfg.py` |
| **Script** | `rsl_rl/train.py` | `model_based/train.py` |

---

## Shared Components

Both use the **exact same**:
- ✅ `SystemDynamicsEnsemble` class
- ✅ `MBPOPPO` algorithm
- ✅ Network architectures
- ✅ Training procedures

**Location**: `isaac-sim/overlay/rsl_rl_rwm/rsl_rl/`

---

## U-RWM Training (Two Stages)

### Stage 1: Train World Model
```bash
# Uses simulator + supervised learning
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless
```
**Output**: `pretrain_rnn_ens.pt` (5-member ensemble)

### Stage 2: Train Policy Offline
```bash
# Pure imagination, no simulator
python scripts/reinforcement_learning/model_based/train.py \
    --task anymal_d_flat
```
**Input**: Frozen ensemble from Stage 1  
**Output**: Trained policy

---

## Your Research Options

### Option 1: Online Ensemble ⭐ (Recommended)
**What**: RWM + U-RWM ensemble = uncertainty-aware online learning

**Config Changes**:
```python
# rsl_rl_ppo_cfg.py
ensemble_size = 5  # Line 22
uncertainty_penalty_weight = -0.1  # Line 126
```

**Best for**: Your uncertainty modeling research

### Option 2: Standard U-RWM
**What**: Follow paper exactly (offline policy training)

**Best for**: Pure offline RL research

### Option 3: Standard RWM (Baseline)
**What**: No ensemble (ensemble_size=1)

**Best for**: Comparison baseline

---

## Quick Start for Uncertainty Research

1. **Enable ensemble**:
   ```python
   # source/mbrl/mbrl/tasks/.../rsl_rl_ppo_cfg.py:22
   ensemble_size = 5
   ```

2. **Train**:
   ```bash
   python scripts/reinforcement_learning/rsl_rl/train.py \
       --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
       --headless
   ```

3. **Monitor** `epistemic_uncertainty_mean` in logs

4. **Implement** improvements from `ensemble_uncertainty.md`

---

## Important Files

- **Full Analysis**: `U-RWM_TRAINING_WORKFLOW.md`
- **Research Plan**: `ensemble_uncertainty.md`
- **RWM Config**: `source/mbrl/mbrl/tasks/.../rsl_rl_ppo_cfg.py`
- **U-RWM Config**: `scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py`
- **Core Ensemble**: `isaac-sim/overlay/rsl_rl_rwm/rsl_rl/modules/system_dynamics.py`
- **Core Algorithm**: `isaac-sim/overlay/rsl_rl_rwm/rsl_rl/algorithms/mbpo_ppo.py`

---

## Summary

**Don't choose between RWM and U-RWM** - they're configuration modes of the same system!

You can:
- ✅ Develop with **Online Ensemble** (best for research)
- ✅ Deploy as **U-RWM Offline** (best for real robots)  
- ✅ Compare against **RWM Baseline** (shows your contribution)

All from **one codebase**! 🚀
