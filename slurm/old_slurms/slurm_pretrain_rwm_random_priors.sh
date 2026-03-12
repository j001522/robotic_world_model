#!/bin/bash
#SBATCH --job-name=rwm-prior-pretrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-prior-pretrain-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-prior-pretrain-%j.err

# =============================================================================
# RWM with Randomized Priors: Ensemble Pretraining
# =============================================================================
# This script trains an ensemble world model with RANDOMIZED PRIORS.
#
# Randomized priors (Osband et al., 2018) prevent ensemble collapse by adding
# a frozen random network output to each ensemble member's predictions.
# This maintains diversity even when all models see the same data.
#
# Key differences from standard ensemble:
#   - Each head has a frozen prior network (Xavier initialized)
#   - Prior output is added to trainable mean predictions (not variances)
#   - Diversity comes from priors, not just data bootstrapping
#
# IMPORTANT: This requires the random_priors branch of rsl_rl_rwm!
#   cd /projects/0/prjs0951/Giacomo/isaac-sim/overlay/rsl_rl_rwm
#   git checkout random_priors
#
# Usage:
#   sbatch slurm_pretrain_rwm_random_priors.sh
# =============================================================================

# ======================= CONFIGURATION =======================
# Training parameters (from Table S10)
MAX_ITERATIONS=2500
NUM_ENVS=4096
ENSEMBLE_SIZE=5

# Randomized Prior parameters
PRIOR_SCALE=1.0        # Prior strength: 0=disabled, 1.0=standard
PRIOR_HIDDEN_DIV=4     # Prior network hidden dim = main_hidden / div

# Uncertainty metric: "std" or "variance"
UNCERTAINTY_METRIC="variance"

# Uncertainty penalty (0.0 for pretrain)
UNCERTAINTY_PENALTY=0.0
# =============================================================

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

# ======================= LOGGING =======================
LOGGER="tensorboard"
WANDB_PROJECT="rwm-anymal"
RUN_NAME="pretrain-ensemble-prior${PRIOR_SCALE}"
# =======================================================

# Print environment info
echo ""
echo "Environment Information:"
echo "  Working directory: $(pwd)"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  Container: /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif"
echo ""

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

echo "========================================"
echo "RWM with Randomized Priors: Ensemble Pretraining"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0"
echo ""
echo "Configuration (from Table S10/S11 + Random Priors):"
echo "  - Ensemble Size: $ENSEMBLE_SIZE"
echo "  - Real Envs: $NUM_ENVS"
echo "  - Max Iterations: $MAX_ITERATIONS"
echo ""
echo "Randomized Prior Settings (Osband et al., 2018):"
echo "  - Prior Scale: $PRIOR_SCALE (0=disabled, 1.0=standard)"
echo "  - Prior Hidden Divisor: $PRIOR_HIDDEN_DIV"
echo "  - Prior applied to: MEANS only (not variances)"
echo ""
echo "Uncertainty Settings:"
echo "  - Uncertainty Metric: $UNCERTAINTY_METRIC"
echo "  - Uncertainty Penalty: $UNCERTAINTY_PENALTY (0 for pretrain)"
echo ""
echo "World Model Hyperparameters:"
echo "  - WM Learning Rate: 1e-4"
echo "  - WM Weight Decay: 1e-5"
echo "  - WM Batch Size: 1024"
echo "  - Policy Learning Rate: 0.001"
echo "  - Imagination: DISABLED (pretrain phase)"
echo ""
echo "Logging:"
echo "  - Logger: $LOGGER"
echo "  - Run Name: $RUN_NAME"
echo ""
echo "Expected training time: ~10-14 hours"
echo "Checkpoints will be saved to:"
echo "  logs/rsl_rl/anymal_d_flat/<timestamp>/"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
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
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --logger $LOGGER \
    --log_project_name $WANDB_PROJECT \
    --run_name $RUN_NAME \
    agent.system_dynamics.ensemble_size=$ENSEMBLE_SIZE \
    agent.system_dynamics.uncertainty_metric=$UNCERTAINTY_METRIC \
    agent.system_dynamics.prior_scale=$PRIOR_SCALE \
    agent.system_dynamics.prior_hidden_div=$PRIOR_HIDDEN_DIV \
    agent.algorithm.system_dynamics_learning_rate=1e-4 \
    agent.algorithm.system_dynamics_weight_decay=1e-5 \
    agent.algorithm.system_dynamics_mini_batch_size=1024 \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo ""
echo "========================================"
echo "Random Priors Pretraining Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Random priors ensemble pretraining completed!"
    echo ""
    echo "Checkpoint location:"
    LATEST_LOG=$(ls -td logs/rsl_rl/anymal_d_flat/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  $LATEST_LOG"
        echo ""
        echo "Files saved:"
        ls -lh "$LATEST_LOG"model_*.pt 2>/dev/null | tail -3
    fi
    echo ""
    echo "The ensemble checkpoint contains:"
    echo "  - ${ENSEMBLE_SIZE}-member ensemble with randomized priors"
    echo "  - Prior scale: $PRIOR_SCALE"
    echo "  - Uncertainty metric: $UNCERTAINTY_METRIC"
    echo ""
    echo "Next steps:"
    echo "  1. Update PRETRAIN_DIR in slurm_finetune_rwm_ensemble.sh"
    echo "  2. Set PRIOR_SCALE=$PRIOR_SCALE in finetune script"
    echo "  3. sbatch slurm_finetune_rwm_ensemble.sh"
else
    echo ""
    echo "FAILED: Random priors pretraining failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/rwm-prior-pretrain-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - Wrong branch: Make sure rsl_rl_rwm is on random_priors branch"
    echo "  - Hydra override: Check parameter names match config"
fi

exit $EXITCODE
