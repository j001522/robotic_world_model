# Manipulation Task Integration with Robotic World Model — Feasibility Analysis

## 1. Current Setup (ANYmal-D Velocity Tracking)

The RWM currently trains on **ANYmal-D velocity tracking** (locomotion). The architecture:

- **World model**: GRU-based `SystemDynamicsEnsemble` (2 layers, 256 hidden) predicting next state (45D), contacts (8D), terminations (1D)
- **Training pipeline**: Pretrain (real + WM training) -> Finetune (real + imagination rollouts, 8192 imagined envs x 24 steps)
- **Key abstraction**: `ManagerBasedMBRLEnv` — 7 abstract methods that define how a task maps to/from the world model's flat state vector

### ANYmal-D Dimensions
| Component | Dimension |
|---|---|
| system_state | 45 (3 base_lin_vel + 3 base_ang_vel + 3 gravity + 12 joint_pos + 12 joint_vel + 12 joint_torque) |
| system_action | 12 (joint position targets) |
| system_contact | 8 (4 thigh + 4 foot binary) |
| system_termination | 1 (base contact) |
| policy obs | 48 (state without torques + 3 velocity command + 12 last_action) |

---

## 2. Available Manipulation Tasks in IsaacLab v2.3.1

The installed IsaacLab has **70+ manipulation environments**:

| Category | Tasks | Robots |
|---|---|---|
| **Reach** | End-effector pose tracking | Franka Panda, UR10, UR10e |
| **Lift** | Pick up cube to goal position | Franka Panda |
| **Cabinet** | Open drawer | Franka Panda |
| **Stack** | Stack cubes | Franka, Galbot, UR10 |
| **Place** | Place object in container | Agibot |
| **In-hand** | Cube reorientation | Allegro Hand |
| **DexSuite** | Dexterous lift/reorient | Kuka iiwa7 + Allegro |
| **Factory** | Peg insert, gear mesh, nut thread | Franka Panda |

### Available Robot Arm Assets
| Robot | DOF | Gripper | Config |
|---|---|---|---|
| Franka Emika Panda | 7+2 | Panda Hand | `FRANKA_PANDA_CFG` |
| UR10 | 6 | None | `UR10_CFG` |
| UR10e | 6 | None/Robotiq | `UR10e_CFG` |
| Kinova JACO2 | 7+6 | 3-finger | `KINOVA_JACO2_N7S300_CFG` |
| Kinova Gen3 | 7 | None | `KINOVA_GEN3_N7_CFG` |
| Sawyer | 7+1 | None | `SAWYER_CFG` |
| Kuka iiwa7 + Allegro | 7+16 | Allegro 4-finger | `KUKA_ALLEGRO_CFG` |

---

## 3. Recommendation: Start with Franka Reach, Then Graduate to Franka Lift

### Why Reach First
1. **Simplest manipulation task** — no object dynamics, no grasping, no contact-rich interactions
2. **Clean mapping to RWM**: state = joint positions/velocities/torques + EE position -> smooth dynamics the world model can learn
3. **No gripper** — action dim is just 7 (joint positions), similar in structure to locomotion's 12 joint targets
4. **Rewards are purely kinematic** — position error, orientation error, action rate — all trivially computable from predicted state in imagination
5. **No termination complexity** — only timeout, no early termination
6. **Existing env is manager-based** — same paradigm as your locomotion task

### Why Lift Second
1. **Tests object dynamics prediction** — world model must predict both robot and object state
2. **Tests contact prediction** — fingertip contacts, object-table contacts become meaningful
3. **Tests termination prediction** — object dropping is a real early termination
4. **More representative of real-robot thesis work** — actual manipulation with grasping
5. **Still relatively simple** — single rigid cube, binary gripper, moderate reward complexity

---

## 4. Understanding the Reward Model (Two Locations)

This is a critical architectural point. Rewards are defined in **two separate places**:

### 4.1 Real Environment Rewards
Defined in the IsaacLab env config via `RewardsCfg`. These use **simulator queries** (e.g., `env.scene["robot"].data.body_pos_w`) and are computed by IsaacLab's reward manager during real rollouts.

For Franka Reach, these are:
- `end_effector_position_tracking` (weight=-0.2): L2 distance between EE and goal
- `end_effector_position_tracking_fine_grained` (weight=0.1): `1 - tanh(distance / 0.1)`
- `end_effector_orientation_tracking` (weight=-0.1): quaternion error
- `action_rate` (weight=-0.0001, curriculum to -0.005): action smoothness
- `joint_vel` (weight=-0.0001, curriculum to -0.001): joint velocity penalty

### 4.2 Imagination Rewards
Manually reimplemented in `_compute_imagination_reward_terms()` inside the `ManagerBasedMBRLEnv` subclass. During imagination, there is **no simulator** — only the world model's predicted flat state vector. So every reward term must be rewritten as a pure tensor operation on that predicted state.

**The two must compute the same reward, just from different data sources.** The real reward queries the sim; the imagination reward queries the predicted state vector.

The term names in `imagination_reward_per_step` must exactly match the names in `RewardsCfg`, because `_post_imagination_step()` looks up the weight via `self.reward_manager.get_term_cfg(term)`.

---

## 5. What Needs to Be Built (Franka Reach)

### 5.1 Directory Structure
```
robotic_world_model/source/mbrl/mbrl/tasks/manager_based/manipulation/
    __init__.py
    reach/
        __init__.py
        config/
            __init__.py
            franka/
                __init__.py          # Task registration (gym.register)
                flat_env_cfg.py      # Env config + observation groups
                agents/
                    __init__.py
                    rsl_rl_ppo_cfg.py # Agent config with normalizers
                envs/
                    __init__.py
                    franka_reach_mbrl_env.py      # MBRL env (7 abstract methods)
                    franka_reach_visualize_env.py  # Visualization env
        mdp/
            __init__.py
            observations.py          # Custom observation functions (ee_pos)
```

### 5.2 Observation Groups for World Model

**system_state (what the WM predicts):**
| Term | Function | Dim |
|---|---|---|
| joint_pos | `mdp.joint_pos_rel` | 7 (arm joints only, excl. fingers) |
| joint_vel | `mdp.joint_vel_rel` | 7 |
| joint_torque | `mdp.joint_effort` | 7 |
| ee_pos | custom `ee_position_in_robot_root_frame` | 3 |
| **Total** | | **24** |

Note: EE position is included in system_state so the world model learns to predict it, avoiding the need to reimplement forward kinematics in PyTorch during imagination.

**system_action:** 7 dims (last_action for 7 arm joints)
**system_contact:** Not needed for reach (no contacts relevant to reward)
**system_termination:** Not needed for reach (only timeout termination)

**policy obs (same as base IsaacLab Reach):** 32 dims
- joint_pos_rel(9) + joint_vel_rel(9) + pose_command(7) + last_action(7)

### 5.3 Imagination Reward Reimplementation

For the reach task, every reward term is a simple function of the predicted state:

```python
# ee_pos is predicted as part of system_state (dims 21:24)
ee_pos = parsed_imagination_states["ee_pos"]  # (N, 3)
goal_pos = self.ee_pose[:, :3]  # command (N, 3)

# Position tracking: L2 distance
position_error = torch.norm(ee_pos - goal_pos, dim=1)
end_effector_position_tracking = position_error

# Fine-grained position tracking: tanh kernel
end_effector_position_tracking_fine_grained = 1 - torch.tanh(position_error / 0.1)

# Orientation tracking: not predicted in state -> set to 0 or include in state
# (simplified: skip orientation tracking initially)

# Action rate: ||a_t - a_{t-1}||^2
action_rate = torch.sum(torch.square(last_action - rollout_action), dim=1)

# Joint velocity: ||joint_vel||^2
joint_vel = torch.sum(torch.square(parsed_imagination_states["joint_vel"]), dim=1)
```

### 5.4 State Normalizer

Must be estimated from real rollout data. Initial estimates for Franka:
| State component | Mean | Std (estimated) |
|---|---|---|
| joint_pos_rel (7) | 0.0 | 0.5 |
| joint_vel_rel (7) | 0.0 | 1.0 |
| joint_torque (7) | 0.0 | 10.0 |
| ee_pos (3) | [0.5, 0.0, 0.3] | [0.15, 0.2, 0.15] |

These are rough estimates. After collecting real rollout data during pretraining, they should be refined.

---

## 6. Feasibility Assessment

| Dimension | Reach | Lift (future) |
|---|---|---|
| **Implementation effort** | ~2-3 days | ~4-5 days (on top of Reach) |
| **World model learnability** | High — smooth dynamics | Medium — contact-rich |
| **Reward reimplementation** | Trivial — distance functions | Moderate — gated rewards |
| **Contact prediction** | Not needed | Needed (3 signals) |
| **Termination prediction** | Not needed | Needed (object drop) |
| **Training data requirement** | Low | Higher — needs diverse grasps |
| **Risk of WM failure** | Low | Medium — discontinuities |

### Key Risks
1. **Normalizer statistics** must be carefully estimated; poor normalization -> poor WM
2. **Forward kinematics**: Solved by including ee_pos in system_state
3. **IsaacLab version mismatch**: RWM references IsaacLab 2.1.0 / IsaacSim 4.5.0, but installed is v2.3.1 / IsaacSim 5.1.0. May have API differences.
4. **Orientation tracking**: Including quaternion orientation in system_state adds complexity (quaternion normalization). Can be skipped initially or handled with Euler angles.

---

## 7. File Reference

### Existing Files to Follow as Templates
- `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py` — task registration pattern
- `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py` — env config with system observation groups
- `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/anymal_d_manager_based_mbrl_env.py` — MBRL env implementation (7 abstract methods)
- `source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py` — agent config with normalizers

### IsaacLab Files Being Extended
- `isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py` — base reach config
- `isaaclab_tasks/manager_based/manipulation/reach/config/franka/joint_pos_env_cfg.py` — Franka specialization
- `isaaclab_tasks/manager_based/manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py` — base Franka PPO config
- `isaaclab_assets/robots/franka.py` — Franka Panda robot asset config

---

## Appendix A: Implementation Progress

Tracks what has been built so far. If a crash occurs, resume from the first incomplete item.

### Phase 1: Core Infrastructure (COMPLETE)

| # | Task | Status | Files |
|---|------|--------|-------|
| 1 | Directory structure with `__init__.py` files | DONE | `manipulation/__init__.py`, `reach/__init__.py`, `reach/config/__init__.py`, `reach/config/franka/envs/__init__.py`, `reach/config/franka/agents/__init__.py` |
| 2 | Custom EE observation function | DONE | `reach/mdp/observations.py` — `ee_position_in_robot_root_frame()` |
| 3 | MDP `__init__.py` re-exports | DONE | `reach/mdp/__init__.py` — re-exports from isaaclab_tasks.reach.mdp + mbrl.envs.mdp + local observations |
| 4 | Environment config (`flat_env_cfg.py`) | DONE | All 5 variants: `FrankaReachFlatEnvCfg`, `_INIT`, `_PRETRAIN` (with 24D system_state obs group), `_FINETUNE`, `_VISUALIZE` |
| 5 | MBRL env (`franka_reach_mbrl_env.py`) | DONE | `FrankaReachManagerBasedMBRLEnv` — all 7+ abstract methods implemented (system state 24D, action 7D, no contacts/termination, 5 imagination reward terms) |
| 6 | Visualization env (`franka_reach_visualize_env.py`) | DONE | `FrankaReachManagerBasedVisualizeEnv` — `_reset_imagination_sim()` writes predicted joint states to sim |

### Phase 2: Agent Config & Registration (COMPLETE)

| # | Task | Status | Files |
|---|------|--------|-------|
| 7 | `SampleUniformPoseCommand` class | DONE | `mbrl/envs/mdp/commands/pose_command.py` — `sample_command(num_envs)` for imagination rollouts |
| 8 | Commands `__init__.py` updated | DONE | `mbrl/envs/mdp/commands/__init__.py` — exports `SampleUniformPoseCommand` |
| 9 | PPO runner configs (`rsl_rl_ppo_cfg.py`) | DONE | `reach/config/franka/agents/rsl_rl_ppo_cfg.py` — 4 configs: `FrankaReachFlatPPORunnerCfg`, `PretrainRunnerCfg` (24D normalizer, GRU WM, contact/termination loss=0), `FinetuneRunnerCfg` (8192 imagined envs), `VisualizeRunnerCfg` |
| 10 | Gym registration (`franka/__init__.py`) | DONE | 4 envs registered: `Template-Isaac-Reach-Franka-{Init,Pretrain,Finetune,Visualize}-v0` |
| 11 | Finetune & visualize command overrides | DONE | `flat_env_cfg.py` — `FINETUNE` uses `SampleUniformPoseCommand`, `VISUALIZE` uses `UniformPoseCommand_Visualize` |

### Phase 3: Testing & Validation (PENDING)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 12 | Verify gym registration loads | PENDING | Run `python scripts/environments/list_envs.py` and check for `Template-Isaac-Reach-Franka-*` |
| 13 | Test Init training (standard RL) | PENDING | `python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Reach-Franka-Init-v0` |
| 14 | Test Pretrain (WM training) | PENDING | `python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Reach-Franka-Pretrain-v0` |
| 15 | Verify system_state dimensions match | PENDING | Check that obs manager reports 24D for system_state during pretrain |
| 16 | Refine normalizer statistics | PENDING | After collecting pretrain data, update mean/std in rsl_rl_ppo_cfg.py |
| 17 | Test Finetune (imagination) | PENDING | Requires successful pretrain; `--task Template-Isaac-Reach-Franka-Finetune-v0` |
| 18 | Test Visualize | PENDING | Requires trained WM; `--task Template-Isaac-Reach-Franka-Visualize-v0` |
| 19 | Validate imagination rewards | PENDING | Compare imagination reward terms against real env rewards during finetune |

### Implementation Notes & Decisions

1. **System state is 24D** (not 45D like locomotion): 7 joint_pos_rel + 7 joint_vel_rel + 7 joint_torque + 3 ee_pos. No base velocity/gravity (fixed base robot).
2. **No contacts or terminations** in system_state: Reach has no contact-based rewards and only timeout termination. WM loss weights for contact and termination set to 0.
3. **EE position in system_state**: Avoids reimplementing FK in PyTorch during imagination. The WM learns to predict ee_pos directly.
4. **Orientation tracking set to 0**: The `end_effector_orientation_tracking` reward term is zeroed during imagination because quaternion orientation is not included in system_state. This simplification trades ~10% reward accuracy for significant implementation simplicity. Can be added later if needed.
5. **Policy obs padded with finger zeros**: The policy expects 9D joint_pos/vel (7 arm + 2 finger), but system_state only has 7D arm joints. During imagination, finger joints are padded with zeros.
6. **SampleUniformPoseCommand**: Created because `UniformPoseCommand` from IsaacLab lacks `sample_command()` needed by `ManagerBasedMBRLEnv._init_imagination_command()`. Follows the same pattern as `SampleUniformVelocityCommand` in locomotion.
7. **Normalizer statistics are initial estimates**: mean=[0]*21+[0.5,0,0.3], std=[0.5]*7+[1.0]*7+[10.0]*7+[0.15,0.2,0.15]. Should be refined after collecting real rollout data.

### Phase 3.5: SLURM Scripts & Configs (DONE)

| # | Task | Status | Files |
|---|------|--------|-------|
| 20 | Pretrain latent config | DONE | `slurm/configs/franka_reach_pretrain_latent.yaml` — 1024 envs, 2000 iters, latent WM |
| 21 | Pretrain ensemble config | DONE | `slurm/configs/franka_reach_pretrain_ensemble.yaml` — 1024 envs, 2000 iters, raw state |
| 22 | Finetune latent config | DONE | `slurm/configs/franka_reach_finetune_latent.yaml` — 1024 real + 4096 imag envs, update checkpoint paths before use |
| 23 | Finetune ensemble config | DONE | `slurm/configs/franka_reach_finetune_ensemble.yaml` — same structure, update checkpoint paths |
| 24 | Test script | DONE | `slurm/slurm_test_franka_reach.sh` — verifies gym registration + 2-iteration pretrain smoke test |
| 25 | README updated | DONE | `slurm/configs/README.md` — added Franka Reach configs section |

**How to run:**
```bash
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/slurm

# Test everything works
sbatch --job-name=fr-test slurm_test_franka_reach.sh

# Pretrain (latent)
sbatch --job-name=fr-pretrain-latent --partition=gpu_a100 --time=12:00:00 \
       slurm_train.sh configs/franka_reach_pretrain_latent.yaml

# Finetune (update checkpoint paths in YAML first!)
sbatch --job-name=fr-finetune-latent --partition=gpu_a100 --time=24:00:00 \
       slurm_train.sh configs/franka_reach_finetune_latent.yaml
```
