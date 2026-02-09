#!/bin/bash
#SBATCH --job-name=rwm-bs-pretrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2               # <--- CRITICAL: Runs 3 jobs (ID 0, 1, 2)
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-bs-pretrain-%A_%a.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-bs-pretrain-%A_%a.err

# =============================================================================
# U-RWM: Ensemble World Model Pretraining (Bootstrap / 3 Seeds)
# =============================================================================

# --- SEED CONFIGURATION ---
SEEDS=(42 123 1001)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Training parameters
MAX_ITERATIONS=2500
NUM_ENVS=4096
ENSEMBLE_SIZE=5
UNCERTAINTY_PENALTY=0.0
UNCERTAINTY_METRIC="std"

# Logging
LOGGER="tensorboard" # or "wandb"
WANDB_PROJECT="rwm-anymal"
BASE_RUN_NAME="pretrain-ensemble-bs"
RUN_NAME="${BASE_RUN_NAME}-seed${SEED}"  # <--- Unique name per seed

# =============================================================================

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "SELECTED SEED: $SEED"
echo "========================================"

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

echo "Configuration:"
echo "  - Ensemble Size: $ENSEMBLE_SIZE"
echo "  - Seed: $SEED"
echo "  - Run Name: $RUN_NAME"

# Create logs directory
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run training inside container
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
    --headless \
    --seed $SEED \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --logger $LOGGER \
    --log_project_name $WANDB_PROJECT \
    --run_name $RUN_NAME \
    agent.system_dynamics.ensemble_size=$ENSEMBLE_SIZE \
    agent.system_dynamics.uncertainty_metric=$UNCERTAINTY_METRIC \
    agent.algorithm.system_dynamics_learning_rate=1e-4 \
    agent.algorithm.system_dynamics_weight_decay=1e-5 \
    agent.algorithm.system_dynamics_mini_batch_size=1024 \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo "End Time: $(date)"
exit $EXITCODE