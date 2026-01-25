# Experiment Tracking

Simple experiment tracking for RWM runs.

## Directory Structure

```
experiments/
├── README.md           # This file
├── experiments.csv     # Experiment log (append-only)
└── completed/          # Summaries of completed runs
```

## Experiment Log Format

The `experiments.csv` file tracks all runs:

| Column | Description |
|--------|-------------|
| `id` | Unique experiment ID (e.g., exp001) |
| `date` | Start date (YYYY-MM-DD) |
| `slurm_job` | SLURM job ID |
| `type` | pretrain-single, pretrain-ensemble, finetune-single, finetune-ensemble |
| `status` | running, completed, failed, cancelled |
| `log_dir` | Path to training logs |
| `description` | What this experiment tests |
| `key_params` | Changed parameters (vs defaults) |
| `final_reward` | Final mean reward (filled after completion) |
| `notes` | Any observations |

## Quick Commands

```bash
# Check running experiments
squeue -u $USER

# Add new experiment (after submitting)
echo "exp002,2026-01-25,18669465,pretrain-single,running,logs/rsl_rl/anymal_d_flat/2026-01-25_11-36-30_pretrain,Paper params baseline,max_iter=2500,," >> experiments/experiments.csv

# Update status after completion
# Edit experiments.csv and change status to completed, add final_reward

# View experiment log
column -t -s',' experiments/experiments.csv
```

## Naming Convention

Run directories follow the pattern:
```
logs/rsl_rl/anymal_d_flat/<timestamp>_<run_name>/
```

Where `run_name` is set in SLURM scripts via `--run_name` flag.
