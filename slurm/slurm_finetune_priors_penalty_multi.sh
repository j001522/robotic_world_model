#!/bin/bash
#SBATCH --job-name=ft-rp-pen
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/ft-rp-pen-%A_%a.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/ft-rp-pen-%A_%a.err

# =============================================================================
# Finetune: Randomized Priors (No Bootstrap), WITH Uncertainty Penalty -0.25 (3 Seeds)
# =============================================================================
# Loads from: Randomized Priors pretrain (slurm_pretrain_rwm_priors_nobs_multi.sh)
# Config: uncertainty_penalty = -0.25, prior_scale = 1.0, bootstrap = False
# =============================================================================

# --- SEED CONFIGURATION (must match pretrain seeds!) ---
SEEDS=(42 123 1001)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# --- PRETRAIN DIRECTORIES (UPDATE AFTER PRETRAIN COMPLETES!) ---
# Format: PRETRAIN_DIRS[seed_index]="logs/rsl_rl/anymal_d_flat/<timestamp>_pretrain-ensemble-prior1.0-noboot-seed<SEED>"
declare -A PRETRAIN_DIRS
PRETRAIN_DIRS[0]="logs/rsl_rl/anymal_d_flat/2026-02-02_19-35-08_pretrain-ensemble-prior1.0-noboot-seed42"   # UPDATE THIS
PRETRAIN_DIRS[1]="logs/rsl_rl/anymal_d_flat/2026-02-02_19-35-28_pretrain-ensemble-prior1.0-noboot-seed123"  # UPDATE THIS
PRETRAIN_DIRS[2]="logs/rsl_rl/anymal_d_flat/2026-02-02_19-35-13_pretrain-ensemble-prior1.0-noboot-seed1001" # UPDATE THIS

PRETRAIN_DIR=${PRETRAIN_DIRS[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT_ITER=2500

# Finetuning parameters
MAX_ITERATIONS=2500
NUM_ENVS=4096
IMAGINATION_ENVS=4096
IMAGINATION_STEPS=100

# Ensemble parameters (MUST match pretrain!)
ENSEMBLE_SIZE=5
PRIOR_SCALE=1.0          # Randomized Priors enabled
PRIOR_HIDDEN_DIV=4
BOOTSTRAP=False          # No bootstrap (priors provide diversity)

# Uncertainty penalty
UNCERTAINTY_PENALTY=-0.06  # PENALTY ENABLED
UNCERTAINTY_METRIC="std"  

# Logging
LOGGER="tensorboard"
WANDB_PROJECT="rwm-anymal"
RUN_NAME="finetune-rp-pen006-${UNCERTAINTY_METRIC}-seed${SEED}"

# =============================================================================

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "SEED: $SEED"
echo "Pretrain Dir: $PRETRAIN_DIR"
echo "Uncertainty Penalty: $UNCERTAINTY_PENALTY"
echo "Prior Scale: $PRIOR_SCALE"
echo "Uncertainty Metric: $UNCERTAINTY_METRIC"
echo "========================================"

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Construct checkpoint path
CHECKPOINT_PATH="${PRETRAIN_DIR}/model_${CHECKPOINT_ITER}.pt"

# Validate checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "ERROR: Checkpoint not found at: $CHECKPOINT_PATH"
    echo "Please update PRETRAIN_DIRS in this script after pretrain completes."
    echo ""
    echo "Available runs:"
    ls -la logs/rsl_rl/anymal_d_flat/ 2>/dev/null | grep "pretrain-ensemble-prior" || echo "  No matching runs found"
    exit 1
fi

# Extract load_run name from PRETRAIN_DIR (everything after the last /)
LOAD_RUN=$(basename "$PRETRAIN_DIR")

mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run finetuning
apptainer exec --nv \
  --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
  --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
  --bind $HOME:$HOME \
  --env OMNI_KIT_ACCEPT_EULA=YES \
  --env GIT_PYTHON_REFRESH=quiet \
  /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
  /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 \
    --headless \
    --seed $SEED \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --resume True \
    --load_run "$LOAD_RUN" \
    --checkpoint model_${CHECKPOINT_ITER}.pt \
    --system_dynamics_load_path $CHECKPOINT_PATH \
    --logger $LOGGER \
    --log_project_name $WANDB_PROJECT \
    --run_name $RUN_NAME \
    agent.system_dynamics.ensemble_size=$ENSEMBLE_SIZE \
    agent.system_dynamics.uncertainty_metric=$UNCERTAINTY_METRIC \
    agent.system_dynamics.prior_scale=$PRIOR_SCALE \
    agent.system_dynamics.prior_hidden_div=$PRIOR_HIDDEN_DIV \
    agent.system_dynamics.bootstrap=$BOOTSTRAP \
    agent.imagination.num_envs=$IMAGINATION_ENVS \
    agent.imagination.num_steps=$IMAGINATION_STEPS \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo "========================================"
echo "Finetune Completed - Seed $SEED"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

exit $EXITCODE
