# Reward Analysis Report: Franka Reach vs ANYmal D Locomotion

## Executive Summary

The Franka Reach manipulation task exhibits diminishing reward trends during training. After a thorough code inspection comparing it to the working ANYmal D locomotion task, I identified **one critical bug, two significant design issues, and two minor inconsistencies**. The critical bug alone is very likely the primary cause of the diminishing rewards.

### Issues Found (Severity Order)

| # | Issue | Severity | Impact |
|---|---|---|---|
| 1 | **Curriculum causes reward weight explosion mid-training** | CRITICAL | Penalty weights jump 50x during training; imagination rewards collapse |
| 2 | **Reward scale is net-negative by design** | HIGH | Total reward is structurally dominated by penalties; policy cannot achieve positive returns |
| 3 | **Orientation reward zeroed = 10% reward capacity lost** | MEDIUM | Reduced reward signal compared to real env |
| 4 | **`joint_vel` computed over 7 joints (imagination) vs 9 joints (real)** | LOW | Minimal practical impact (finger velocities ~0 in reach) |
| 5 | **Normalizer statistics are rough estimates** | LOW | May degrade world model prediction quality |

---

## 1. CRITICAL: Curriculum Causes Reward Collapse Mid-Training

### The Problem

The IsaacLab `ReachEnvCfg` defines a `CurriculumCfg` that abruptly changes reward weights after 4500 real environment steps:

```
action_rate:  -0.0001  -->  -0.005   (50x increase in penalty)
joint_vel:    -0.0001  -->  -0.001   (10x increase in penalty)
```

**File:** `IsaacLab/.../manipulation/reach/reach_env_cfg.py` (CurriculumCfg, lines 171-180)

This curriculum is inherited by `FrankaReachFlatEnvCfg` and is **never disabled**. The ANYmal D task, by contrast, has **no reward-weight curriculum** at all (its only curriculum was terrain levels, which is disabled for the flat variant).

### Why This Breaks MBRL Training

The curriculum runs during real environment resets (via `ManagerBasedRLEnv._reset_idx()` -> `self.curriculum_manager.compute()`). Once `common_step_counter > 4500`, it mutates the reward weights stored in `self.reward_manager`. Critically, **imagination rollouts read from the same `reward_manager`** via `self.reward_manager.get_term_cfg(term)` in `_post_imagination_step()` (line 145 of `manager_based_mbrl_env.py`).

This means:
1. Before step 4500: action_rate contributes ~ `-0.0001 * value * dt` per step
2. After step 4500: action_rate contributes ~ `-0.005 * value * dt` per step (50x larger)
3. The policy was optimized under the old weights; the sudden change invalidates the value function
4. Both real and imagined rewards are affected simultaneously

At 1024 environments with episode length ~500 steps, step 4500 is reached within the first ~9 PPO iterations. So the weight explosion happens very early.

### The Fix

Disable the curriculum in `FrankaReachFlatEnvCfg.__post_init__()`:

```python
# In flat_env_cfg.py, FrankaReachFlatEnvCfg.__post_init__():
self.curriculum.action_rate = None
self.curriculum.joint_vel = None
```

And set the desired final weights directly in the reward config instead. If you want the stronger penalties from the start:
```python
self.rewards.action_rate.weight = -0.005
self.rewards.joint_vel.weight = -0.001
```

Or keep them small:
```python
self.rewards.action_rate.weight = -0.0001
self.rewards.joint_vel.weight = -0.0001
```

**File to edit:** `source/mbrl/mbrl/tasks/manager_based/manipulation/reach/config/franka/flat_env_cfg.py`

---

## 2. HIGH: Reward Scale Is Net-Negative by Design

### Comparing Reward Structure

**ANYmal D Locomotion** has a well-balanced reward that is structurally net-positive when the robot performs well:

| Term | Weight | Range | Typical Contribution | Type |
|---|---|---|---|---|
| `track_lin_vel_xy_exp` | **+1.0** | [0, 1] | **+1.0** (at perfect tracking) | Reward |
| `track_ang_vel_z_exp` | **+0.5** | [0, 1] | **+0.5** | Reward |
| `feet_air_time` | **+0.5** | [0, ~2] | **+0.25** | Reward |
| `lin_vel_z_l2` | -2.0 | [0, inf) | -0.02 | Penalty |
| `ang_vel_xy_l2` | -0.05 | [0, inf) | -0.01 | Penalty |
| `dof_torques_l2` | -2.5e-5 | [0, inf) | -0.1 | Penalty |
| `dof_acc_l2` | -2.5e-7 | [0, inf) | -0.001 | Penalty |
| `action_rate_l2` | -0.01 | [0, inf) | -0.05 | Penalty |
| `flat_orientation_l2` | -5.0 | [0, ~0.04] | -0.01 | Penalty |
| `undesired_contacts` | -1.0 | [0, 4] | -0.1 | Penalty |
| `stand_still` | -1.0 | [0, ~12] | -0.0 (when moving) | Penalty |
| | | | **Net: ~+1.5** | |

The positive reward terms dominate. A well-performing policy can achieve **net positive returns**.

**Franka Reach** has a structurally different picture:

| Term | Weight | Range | Typical Contribution | Type |
|---|---|---|---|---|
| `ee_position_tracking` | **-0.2** | [0, ~1] | **-0.1** (at moderate distance) | Penalty |
| `ee_position_tracking_fine_grained` | **+0.1** | [0, 1] | **+0.07** (moderate tracking) | Reward |
| `ee_orientation_tracking` | **-0.1** | [0, pi] | **0.0** (zeroed in imagination) | Penalty |
| `action_rate` | -0.0001 | [0, inf) | -0.001 | Penalty |
| `joint_vel` | -0.0001 | [0, inf) | -0.001 | Penalty |
| | | | **Net: ~-0.03** | |

The **only positive term** (`ee_position_tracking_fine_grained`, weight +0.1) can at most contribute +0.1 per step, but the negative position tracking term (weight -0.2) always contributes a negative amount unless distance is exactly zero. Even at perfect tracking (distance=0), the fine-grained term gives +0.1 while position tracking gives 0.0, for a total of +0.1. But during training, the robot is rarely at distance=0.

After curriculum kicks in (step 4500), the penalties become even more dominant:
- `action_rate`: -0.005 (50x increase)
- `joint_vel`: -0.001 (10x increase)

This means the total reward trends **increasingly negative over training** as penalties grow.

### Comparison to ANYmal D Design Philosophy

The ANYmal D task follows the principle that **positive rewards for task achievement should dominate** and penalties should be small shaping terms. The Franka Reach task inverts this: the dominant term is a **negative penalty** (-0.2 * distance), with only a small positive bonus (+0.1 * tanh_kernel).

### Suggested Rebalancing

Option A: Increase the positive reward weight substantially:
```python
self.rewards.end_effector_position_tracking.weight = -0.2
self.rewards.end_effector_position_tracking_fine_grained.weight = 1.0  # was 0.1
```

Option B: Use only the tanh-kernel reward (purely positive) and remove the raw distance penalty:
```python
self.rewards.end_effector_position_tracking.weight = 0.0     # disable raw penalty
self.rewards.end_effector_position_tracking_fine_grained.weight = 1.0  # dominant positive reward
```

Option C: Add an exponential tracking reward (like ANYmal D's `track_lin_vel_xy_exp`):
```python
# In imagination reward:
position_tracking_exp = torch.exp(-distance / 0.05)  # 1.0 at d=0, ~0.14 at d=0.1m
```

---

## 3. MEDIUM: Orientation Tracking Reward Is Zeroed

The `end_effector_orientation_tracking` reward term is set to `torch.zeros(...)` during imagination because quaternion orientation is not included in the system state. In the real environment, this term has weight `-0.1` and penalizes orientation error.

**Impact:** During imagination, the policy receives no signal about end-effector orientation. This means:
- ~10% of the real reward signal is missing in imagination
- The policy may learn orientations in the real env that it cannot be rewarded for in imagination
- Creates a systematic gap between real and imagination rewards (faithfulness gap)

**Mitigation:** For a basic reach task where orientation matters less, this is acceptable. But if you need orientation tracking, you would need to either:
1. Include EE orientation (as quaternion or Euler angles) in the system state
2. Implement forward kinematics in PyTorch to derive orientation from joint positions

---

## 4. LOW: `joint_vel` Dimension Mismatch

**Real env:** `joint_vel_l2` uses `SceneEntityCfg("robot")` with default `joint_ids=slice(None)`, summing squared velocities over **all 9 joints** (7 arm + 2 finger).

**Imagination:** Sums squared velocities over **7 arm joints only** (from system_state indices [7:14]).

**Practical impact:** Minimal. The finger joints are unactuated in the reach task (`gripper_action=None`), so their velocities are near-zero. The 2 extra dimensions contribute negligibly to the real reward.

**Fix (if desired):** In the real env config, restrict `joint_vel` to arm joints only:
```python
joint_vel = RewTerm(
    func=mdp.joint_vel_l2,
    weight=-0.0001,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
)
```

---

## 5. LOW: Normalizer Statistics Are Rough Estimates

The state normalizer in `rsl_rl_ppo_cfg.py` uses hand-estimated statistics:

| Component | Mean | Std | Concern |
|---|---|---|---|
| joint_pos_rel (7D) | 0.0 | 0.5 | Reasonable for relative positions |
| joint_vel_rel (7D) | 0.0 | 1.0 | May underestimate (Franka max vel ~2.6 rad/s) |
| joint_torque (7D) | 0.0 | 10.0 | Franka max torques range 12-87 Nm; uniform 10.0 may be too low for base joints |
| ee_pos (3D) | [0.5, 0.0, 0.3] | [0.15, 0.2, 0.15] | Depends on workspace; reasonable for typical reach goals |

The ANYmal D normalizer, by contrast, uses **carefully tuned per-joint statistics** (e.g., different std for hip vs knee vs ankle joints, different torque means per joint).

**Recommendation:** After running pretrain with data collection, compute the actual mean/std from the collected trajectories and update the normalizer. Poor normalization degrades world model prediction quality, which compounds errors during imagination rollouts.

---

## Detailed Reward Architecture Reference

### How Rewards Flow Through the System

Rewards are defined in **two places** and must agree:

```
                        REAL ENVIRONMENT                    IMAGINATION
                        ================                    ===========

Source of truth:        IsaacLab RewardsCfg                 _compute_imagination_reward_terms()
                        (reward_manager computes terms)     (manual tensor operations)

Reward function:        mdp.position_command_error(env)     torch.norm(ee_pos - goal_pos)
                        (queries simulator via env.scene)   (queries predicted state vector)

Weight lookup:          reward_manager stores weights       reward_manager.get_term_cfg(term).weight
                                                           (reads from SAME reward_manager)

Aggregation:            sum(weight * term_value)            sum(weight * term_value * dt)
                        (done by reward_manager)            (done by _post_imagination_step)
```

Key architectural fact: the `weight` used in imagination comes from the real environment's `reward_manager`. This means any mutation of weights (e.g., by curriculum) affects both real and imagination rewards.

### _post_imagination_step() Reward Aggregation (Base Class)

```python
# manager_based_mbrl_env.py, lines 136-148
def _post_imagination_step(self):
    rewards = torch.zeros(self.num_imagination_envs, ...)
    for term in self.imagination_episode_sums.keys():
        if term == "uncertainty":
            rewards += uncertainty_penalty_weight * epistemic_uncertainty * dt
        else:
            term_cfg = self.reward_manager.get_term_cfg(term)    # gets weight
            term_value = self.imagination_reward_per_step[term]   # computed by subclass
            rewards += term_cfg.weight * term_value * dt          # weight * value * dt
    ...
```

Note the `* dt` factor. This scales the reward by the simulation timestep, which is standard for continuous-time reward formulations. Both ANYmal D and Franka Reach use this same aggregation.

### ANYmal D: Why It Works Well

1. **Net-positive reward structure:** Tracking rewards (+1.0, +0.5) dominate over penalties
2. **No curriculum on reward weights:** Weights are constant throughout training
3. **Rich shaping signals:** 12 reward terms covering velocity tracking, orientation, contacts, air time, torques, accelerations
4. **Carefully tuned normalizer:** Per-joint statistics that match the robot's actual operating range
5. **Contact and termination prediction:** The world model predicts contacts (8D) and base-contact termination (1D), giving the imagination environment accurate signals for contact-dependent rewards

### Franka Reach: Why It Struggles

1. **Net-negative reward structure:** The dominant term is a penalty (-0.2 * distance)
2. **Curriculum causes reward weight explosion:** Penalty weights jump 50x at step 4500
3. **Limited positive reward signal:** Only one positive term (+0.1), and it saturates near the goal
4. **Missing orientation signal:** 10% of reward capacity zeroed during imagination
5. **Rough normalizer estimates:** May cause poor world model predictions, especially for torques
6. **No curriculum in ANYmal D to compare against:** The working task avoids this class of issues entirely

---

## Implementing More Complex Manipulation Tasks: What's Needed

This section documents the requirements for extending the MBRL framework to more complex manipulation tasks (lift, stack, in-hand manipulation, etc.), building on lessons from the reach implementation.

### Requirement 1: Object State in the World Model

For tasks involving objects (lift, push, stack), the world model must predict **object state** in addition to robot state.

**What to add to system_state:**
```
object_pos_rel (3D):   object position in robot root frame
object_quat (4D):      object orientation (quaternion)
object_lin_vel (3D):   object linear velocity
object_ang_vel (3D):   object angular velocity
```

**Challenges:**
- Quaternion prediction: The world model outputs raw values, but quaternions must be normalized (||q||=1). Options:
  - Predict unnormalized, then normalize after prediction
  - Predict Euler angles instead (avoids normalization but has gimbal lock)
  - Predict rotation matrix elements (6D representation from Zhou et al. 2019)
- Object dynamics are **discontinuous** at contact events (grasp, release, collision), making them harder for the GRU-based world model to learn
- System state dimension grows: 24D (reach) -> 37D+ (lift with object state)

**Implementation pattern:**
```python
# In flat_env_cfg.py SystemStateCfg:
object_pos = ObsTerm(
    func=mdp.object_position_in_robot_root_frame,
    params={"object_cfg": SceneEntityCfg("cube")},
)
object_vel = ObsTerm(func=mdp.root_lin_vel_b, params={"asset_cfg": SceneEntityCfg("cube")})

# In franka_lift_mbrl_env.py _parse_imagination_states():
parsed["object_pos"] = states[:, 24:27]
parsed["object_quat"] = F.normalize(states[:, 27:31], dim=1)  # normalize quaternion
parsed["object_vel"] = states[:, 31:34]
```

### Requirement 2: Contact Prediction

Contact-dependent rewards (grasping force, object contact with table, fingertip contacts) require the world model to predict contact states.

**What to add:**
```python
# In flat_env_cfg.py SystemContactCfg:
fingertip_contact = ObsTerm(
    func=mdp.body_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="panda_.*finger.*"), "threshold": 1.0}
)
object_table_contact = ObsTerm(
    func=mdp.body_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="cube"), "threshold": 1.0}
)
```

**In the MBRL env:**
```python
def _parse_contacts(self, contacts):
    fingertip_contact = torch.sigmoid(contacts[:, 0:2]).round()
    object_table_contact = torch.sigmoid(contacts[:, 2:3]).round()
    return {"fingertip": fingertip_contact, "object_table": object_table_contact}
```

**Agent config:** Set `system_dynamics_loss_weights["contact"] = 1.0` (currently 0.0 for reach).

### Requirement 3: Termination Prediction

Tasks with failure conditions (object dropped, robot in collision) need termination prediction.

**Example for lift task:**
```python
# In flat_env_cfg.py SystemTerminationCfg:
object_dropped = ObsTerm(
    func=mdp.body_contact,
    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="table"), "threshold": 0.5}
)
# (or a custom function that checks if object_z < threshold)
```

**In the MBRL env:**
```python
def _parse_terminations(self, terminations):
    return torch.sigmoid(terminations).squeeze(-1).round().bool()
```

**Agent config:** Set `system_dynamics_loss_weights["termination"] = 1.0` (currently 0.0 for reach).

### Requirement 4: Gripper Action Space

Tasks requiring grasping need the gripper action in the action space.

**Changes needed:**
- Action dimension: 7 -> 8 or 9 (depending on gripper type)
  - Binary gripper (Franka): +1D (open/close command)
  - Parallel jaw: +1D (width target)
  - Dexterous hand (Allegro): +16D
- `system_action` observation must include gripper action
- Action normalizer must be updated (8D or 9D instead of 7D)
- Policy observation `last_action` dimension increases accordingly

**Implementation:**
```python
# In joint_pos_env_cfg.py:
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger_joint.*"],
    open_command_expr={"panda_finger_joint.*": 0.04},
    close_command_expr={"panda_finger_joint.*": 0.0},
)
```

### Requirement 5: Gated / Multi-Phase Rewards

Complex tasks have rewards that are conditional on task phase (e.g., lift reward only activates after grasp is achieved).

**Example for lift task rewards:**
```python
def _compute_imagination_reward_terms(self, states, action, extensions, contacts):
    object_pos = states["object_pos"]
    fingertip_contact = contacts["fingertip"]

    # Phase 1: Reaching (always active)
    reach_distance = torch.norm(ee_pos - object_pos, dim=1)
    reaching_reward = 1 - torch.tanh(reach_distance / 0.1)

    # Phase 2: Grasping (active when fingertips contact object)
    is_grasping = fingertip_contact.sum(dim=1) >= 2  # both fingers in contact
    grasp_reward = is_grasping.float()

    # Phase 3: Lifting (active only when grasping)
    lift_height = object_pos[:, 2] - self.table_height
    lifting_reward = is_grasping.float() * torch.clamp(lift_height - self.goal_height, min=0.0)

    # Phase 4: Goal tracking (active when lifted)
    goal_distance = torch.norm(object_pos - goal_pos, dim=1)
    goal_reward = is_grasping.float() * (1 - torch.tanh(goal_distance / 0.05))

    self.imagination_reward_per_step = {
        "reaching": reaching_reward,
        "grasping": grasp_reward,
        "lifting": lifting_reward,
        "goal_tracking": goal_reward,
        "action_rate": ...,
        "joint_vel": ...,
    }
```

**Key challenge:** Gated rewards depend on contact predictions being accurate. If the world model mispredicts contacts, the gating conditions will be wrong, and the policy will receive incorrect reward signals during imagination.

### Requirement 6: Reward Weight Balancing Checklist

Based on the comparison between ANYmal D (works) and Franka Reach (doesn't), follow these guidelines:

1. **Positive task rewards should dominate.** The sum of positive rewards at good performance should be at least 3-5x larger than the sum of penalties at good performance.

2. **Avoid curriculum on reward weights in MBRL.** The shared reward_manager means curriculum affects both real and imagination rewards. If you need curriculum, implement it as a separate mechanism in the MBRL env that doesn't mutate the reward_manager.

3. **Scale penalties to be small shaping terms.** Penalties (action_rate, joint_vel, torques) should be 10-100x smaller in magnitude than the primary task reward.

4. **Test reward balance before training:**
   ```python
   # Compute expected reward at random policy vs expert policy
   random_reward = sum(weight * E[term_value_random] for all terms)
   expert_reward = sum(weight * E[term_value_expert] for all terms)
   assert expert_reward > 0, "Expert policy should achieve positive reward"
   assert expert_reward > 5 * abs(random_reward), "Expert should be clearly better"
   ```

5. **Use exponential or tanh kernels for positive rewards** (like ANYmal D's `exp(-error / std)`), not raw distance penalties. These have bounded range [0, 1] and provide clear gradient signal.

### Requirement 7: Normalizer Refinement Pipeline

For any new task, follow this process:

1. Run pretrain for ~200 iterations with rough estimates
2. Extract system_state statistics from collected trajectories:
   ```python
   states = replay_buffer.get_all_states()  # (N, state_dim)
   per_dim_mean = states.mean(dim=0)
   per_dim_std = states.std(dim=0).clamp(min=1e-6)
   ```
3. Update the normalizer in `rsl_rl_ppo_cfg.py`
4. Restart pretrain with refined statistics

### Requirement 8: Imagination Reward Validation

Always validate that imagination rewards match real environment rewards. The framework already has infrastructure for this (see `scripts/analysis/evaluate_real_rewards.py`).

**Validation procedure:**
1. Run a pretrained policy in the real environment
2. Record system_state, action, and real reward per term at each step
3. Feed the same system_state through `_compute_imagination_reward_terms()`
4. Compare real vs imagination reward per term
5. The **faithfulness gap** (difference) should be < 5% for all terms

---

## Summary of Recommended Fixes for Franka Reach

### Fix 1: Disable curriculum (CRITICAL)

**File:** `source/mbrl/mbrl/tasks/manager_based/manipulation/reach/config/franka/flat_env_cfg.py`

In `FrankaReachFlatEnvCfg.__post_init__()`:
```python
def __post_init__(self):
    super().__post_init__()
    # Disable reward weight curriculum -- incompatible with MBRL
    # (curriculum mutates shared reward_manager, causing 50x penalty spike at step 4500)
    self.curriculum.action_rate = None
    self.curriculum.joint_vel = None
    # Set desired penalty weights directly
    self.rewards.action_rate.weight = -0.005
    self.rewards.joint_vel.weight = -0.001
```

### Fix 2: Rebalance reward weights (HIGH)

**File:** Same as Fix 1

```python
def __post_init__(self):
    super().__post_init__()
    # ... curriculum fix above ...
    # Rebalance: make positive reward dominant
    self.rewards.end_effector_position_tracking_fine_grained.weight = 1.0  # was 0.1
    # Optionally reduce raw distance penalty
    self.rewards.end_effector_position_tracking.weight = -0.1              # was -0.2
```

### Fix 3: Restrict joint_vel to arm joints (LOW)

**File:** Same as Fix 1

```python
from isaaclab.managers import SceneEntityCfg

def __post_init__(self):
    super().__post_init__()
    # ... fixes above ...
    # Match imagination (7 arm joints only)
    self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
```

---

## Appendix A: Complete Reward Weight Tables

### Franka Reach (After Fixes -- Applied 2026-03-10)

| Term | Weight | Imagination Formula | Notes |
|---|---|---|---|
| `end_effector_position_tracking` | **-0.1** | `norm(ee_pos - goal_pos)` | Reduced from -0.2 (Fix 2) |
| `end_effector_position_tracking_fine_grained` | **+1.0** | `1 - tanh(distance / 0.1)` | Increased from +0.1 (Fix 2) -- now dominant |
| `end_effector_orientation_tracking` | -0.1 | `0.0` (zeroed) | Not in system state |
| `action_rate` | **-0.005** | `sum(square(a_t - a_{t-1}))` | Fixed weight, curriculum disabled (Fix 1) |
| `joint_vel` | **-0.001** | `sum(square(joint_vel))` 7 arm joints | Fixed weight, curriculum disabled (Fix 1), arm-only (Fix 3) |

**Estimated net reward at moderate performance (d~0.05m): ~+0.53** (structurally positive)

### ANYmal D Flat (Current -- Working)

| Term | Weight | Imagination Formula |
|---|---|---|
| `track_lin_vel_xy_exp` | **+1.0** | `exp(-lin_vel_error / 0.25)` |
| `track_ang_vel_z_exp` | **+0.5** | `exp(-ang_vel_error / 0.25)` |
| `lin_vel_z_l2` | -2.0 | `square(base_lin_vel_z)` |
| `ang_vel_xy_l2` | -0.05 | `sum(square(base_ang_vel_xy))` |
| `dof_torques_l2` | -2.5e-5 | `sum(square(joint_torque))` |
| `dof_acc_l2` | -2.5e-7 | `sum(square(joint_acc))` |
| `action_rate_l2` | -0.01 | `sum(square(a_t - a_{t-1}))` |
| `feet_air_time` | **+0.5** | `sum((air_time - 0.5) * first_contact) * moving` |
| `undesired_contacts` | -1.0 | `sum(thigh_contact)` |
| `stand_still` | -1.0 | `sum(abs(joint_pos)) * (cmd < 0.05)` |
| `flat_orientation_l2` | -5.0 | `sum(square(projected_gravity_xy))` |
| `dof_pos_limits` | 0.0 | `0` (disabled) |

## Appendix B: File Reference

| File | Purpose |
|---|---|
| `source/mbrl/mbrl/tasks/.../manipulation/reach/config/franka/flat_env_cfg.py` | Env config with reward weights |
| `source/mbrl/mbrl/tasks/.../manipulation/reach/config/franka/envs/franka_reach_mbrl_env.py` | Imagination reward computation |
| `source/mbrl/mbrl/tasks/.../manipulation/reach/config/franka/agents/rsl_rl_ppo_cfg.py` | PPO + WM config, normalizers |
| `source/mbrl/mbrl/tasks/.../manipulation/reach/mdp/observations.py` | Custom EE observation |
| `source/mbrl/mbrl/mbrl/envs/manager_based_mbrl_env.py` | Base MBRL env with reward aggregation |
| `source/mbrl/mbrl/tasks/.../locomotion/velocity/config/anymal_d/flat_env_cfg.py` | ANYmal D env config (reference) |
| `source/mbrl/mbrl/tasks/.../locomotion/velocity/config/anymal_d/envs/anymal_d_manager_based_mbrl_env.py` | ANYmal D imagination rewards (reference) |
| `IsaacLab/.../manipulation/reach/reach_env_cfg.py` | Base reach config with curriculum |
| `IsaacLab/.../manipulation/reach/mdp/rewards.py` | Real reward function implementations |
