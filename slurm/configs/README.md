# RWM Training Configuration System

This directory contains a unified training launcher that replaces the 24+ separate slurm scripts.

## Quick Start

```bash
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/slurm

# Pretrain with latent space (single seed for quick test)
sbatch --job-name=pretrain-latent --partition=gpu_a100 --time=24:00:00 \
       --array=0-0 slurm_train.sh configs/pretrain_latent.yaml

# Pretrain with multiple seeds (for experiments)
# Config has seeds: [42, 123, 1001], so array range is 0-2
sbatch --job-name=pretrain-latent --partition=gpu_a100 --time=24:00:00 \
       --array=0-2 slurm_train.sh configs/pretrain_latent_multi.yaml

# Finetune from multi-seed pretrain (auto-matching seeds)
sbatch --job-name=finetune-latent --partition=gpu_a100 --time=24:00:00 \
       --array=0-2 slurm_train.sh configs/finetune_latent_multi.yaml

# Monitor array job status
squeue -u $USER  # Shows all jobs and array tasks
# Check individual array task logs
ls logs/slurm/*-{JOB_ID}-{TASK_ID}.out
```

## Available Configs

| Config | Mode | Description |
|--------|------|-------------|
| `pretrain_ensemble.yaml` | Pretrain | Standard 5-member ensemble, raw state |
| `pretrain_latent.yaml` | Pretrain | Phase 1+2: Latent space with EMA target encoder |
| `pretrain_latent_multi.yaml` | Pretrain | Same as pretrain_latent, but runs 3 seeds |
| `pretrain_bootstrap.yaml` | Pretrain | Bootstrap sampling ensemble |
| `finetune_ensemble.yaml` | Finetune | From ensemble pretrain (std metric) |
| `finetune_latent.yaml` | Finetune | From latent pretrain (manual checkpoint paths) |
| `finetune_latent_multi.yaml` | Finetune | From latent pretrain, 3 seeds (auto-matching) |
| `pretrain_latent_rv.yaml` | Pretrain | Phase 3: Latent space + reward & value heads |
| `finetune_latent_rv.yaml` | Finetune | From latent+reward/value pretrain |
| `finetune_bootstrap_variance.yaml` | Finetune | Bootstrap with variance metric |

### Franka Reach (Manipulation)

| Config | Mode | Description |
|--------|------|-------------|
| `franka_reach_pretrain_latent.yaml` | Pretrain | Latent-space WM for Franka Reach (24D state) |
| `franka_reach_pretrain_ensemble.yaml` | Pretrain | Standard ensemble for Franka Reach |
| `franka_reach_pretrain_multi.yaml` | Pretrain | Standard ensemble with 3 seeds (use `--array=0-2`) |
| `franka_reach_finetune_latent.yaml` | Finetune | From latent pretrain (update checkpoint paths!) |
| `franka_reach_finetune_ensemble.yaml` | Finetune | From ensemble pretrain (update checkpoint paths!) |
| `franka_reach_finetune_ensemble_multi.yaml` | Finetune | Seed-aware finetune from ensemble pretrain (use `--array=0-2`) |

**Test script:** `slurm/slurm_test_franka_reach.sh` — verifies gym registration + runs 2-iteration pretrain

## Creating Custom Configs

Copy an existing config and modify:

```bash
cp configs/pretrain_latent.yaml configs/pretrain_latent_custom.yaml
# Edit pretrain_latent_custom.yaml
sbatch slurm_train.sh configs/pretrain_latent_custom.yaml
```

## Config Structure

```yaml
mode: pretrain              # or 'finetune'
task: <task_name>
max_iterations: 2500
seeds: [42, 123, 1001]      # Optional: list of seeds (default: [42])

model:
  ensemble_size: 5
  latent_mode: true/false
  latent_dim: 256
  # ... other model params

training:
  num_envs: 4096
  wm_learning_rate: 1e-4
  # ... other training params

reward_value:               # Phase 3 (optional, disabled by default)
  reward_head_enabled: false
  value_head_enabled: false
  value_ensemble_size: 5
  reward_hidden_dims: [256, 256]
  value_hidden_dims: [256, 256]
  reward_prior_scale: 5.0
  value_prior_scale: 5.0
  value_gamma: 0.99
  reward_loss_weight: 0.5
  value_loss_weight: 0.1

imagination:                # Only for finetune mode
  num_envs: 4096
  num_steps: 100
  uncertainty_penalty_weight: -0.1

checkpoint:                 # Only for finetune mode
  resume: true

  # Option 1: Seed-aware loading (recommended for multi-seed)
  pretrain_run_name: "pretrain-latent-rec_only_bi"
  pretrain_logs_dir: "logs/rsl_rl/anymal_d_flat/"  # Optional: auto-detected for Anymal/Franka
  checkpoint: model_2500.pt
  # load_run and system_dynamics_load_path are auto-generated

  # Option 2: Manual loading (for single seed or custom paths)
  load_run: "<pretrain_dir>"
  system_dynamics_load_path: "<full_path>"
  checkpoint: model_2500.pt

logging:
  logger: tensorboard       # or 'wandb'
  run_name: <run_name>
```

## Important Notes

1. **Finetune configs must update the checkpoint paths** to point to your actual pretrain runs
2. **Model parameters must match** between pretrain and finetune (ensemble_size, latent_mode, priors)
3. **Phase 3 reward/value heads** are behind feature flags — when disabled (default), behavior is identical to Phase 2
4. **Job resources** (partition, time, gpus) are set via sbatch flags, not in config
5. **All 20+ old slurm scripts** are still available but deprecated - prefer this system
6. **Job arrays**: When using multiple seeds, submit with `--array=0-(N-1)` where N is the number of seeds in your config. Each seed runs in parallel on different nodes.
7. **Single-seed tests**: Use `--array=0-0` for quick tests with a single seed
8. **Output files**: Each array task produces its own `.out` and `.err` file: `{job_name}-{job_id}-{task_id}.{out,err}`
9. **Multi-seed experiments**: Add `seeds: [42, 123, 1001]` to your config and submit with `--array=0-2`. Training runs in parallel for each seed, with `-seed<N>` appended to run_name.

## Seed-Aware Finetuning with Job Arrays

When using multiple seeds with job arrays, each array task automatically matches its finetune seed to the corresponding pretrain checkpoint.

**Setup in finetune config:**
```yaml
seeds: [42, 123, 1001]  # Must match pretrain seeds

checkpoint:
  resume: true
  pretrain_run_name: "pretrain-latent-rec_only_bi"  # Base name, no seed suffix
  pretrain_logs_dir: "logs/rsl_rl/anymal_d_flat/"  # Optional: auto-detected for Anymal/Franka
  checkpoint: model_2500.pt
```

**Submit with matching array range:**
```bash
sbatch --job-name=finetune-latent --partition=gpu_a100 --time=24:00:00 \
       --array=0-2 slurm_train.sh configs/finetune_latent_multi.yaml
```

**How it works:**
- Array task 0 (seed 42) → loads from `*_pretrain-latent-rec_only_bi-seed42/model_2500.pt`
- Array task 1 (seed 123) → loads from `*_pretrain-latent-rec_only_bi-seed123/model_2500.pt`
- Array task 2 (seed 1001) → loads from `*_pretrain-latent-rec_only_bi-seed1001/model_2500.pt`

All finetune tasks run in parallel, each on its own node, automatically loading the correct checkpoint based on its seed.

**Auto-detection:**
- `*-Anymal-D-*` tasks → `logs/rsl_rl/anymal_d_flat/`
- `*-Franka-*` tasks → `logs/rsl_rl/franka_reach/`
- Other tasks → specify `pretrain_logs_dir` explicitly or let the script search

The script searches the logs directory and finds the most recent matching pretrain for each seed. No manual path updates needed!

## Migration from Old System

Old way (24 separate files):
```bash
sbatch slurm_pretrain_rwm_latent.sh
sbatch slurm_finetune_rwm_ensemble.sh
```

New way (1 template + configs):
```bash
sbatch --array=0-0 slurm_train.sh configs/pretrain_latent.yaml
sbatch --array=0-0 slurm_train.sh configs/finetune_ensemble.yaml
```

Old way (multi-seed with array jobs):
```bash
# Had to use slurm_pretrain_rwm_ensemble_multi.sh with --array=0-2
```

New way (multi-seed with job arrays):
```bash
# Add seeds: [42, 123, 1001] to your config, submit with matching array range
sbatch --array=0-2 slurm_train.sh configs/pretrain_latent_multi.yaml
```
