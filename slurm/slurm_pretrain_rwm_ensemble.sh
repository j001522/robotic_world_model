#!/bin/bash
#SBATCH --job-name=rwm-ensemble-pretrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=20:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-ensemble-pretrain-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-ensemble-pretrain-%j.err

# =============================================================================
# U-RWM: Ensemble World Model Pretraining
# =============================================================================
# This script trains a 5-member ensemble world model for uncertainty estimation.
# Uses the same task as classic RWM but with ensemble_size=5 via Hydra override.
#
# The ensemble enables:
#   - Epistemic uncertainty quantification
#   - More robust dynamics predictions
#   - Required for offline U-RWM training (Phase 2)
#
# Usage:
#   sbatch slurm_pretrain_rwm_ensemble.sh
# =============================================================================

# ======================= CONFIGURATION =======================
# Training parameters (from Table S10)
MAX_ITERATIONS=2500
NUM_ENVS=4096
ENSEMBLE_SIZE=5

# Uncertainty penalty (0.0 for pretrain, -0.1 for conservative exploration)
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
# Weights & Biases logging (set to "tensorboard" to disable wandb)
LOGGER="tensorboard"
WANDB_PROJECT="rwm-anymal"

# Run name for easy identification (will appear in wandb dashboard)
RUN_NAME="pretrain-ensemble"
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
echo "U-RWM: Ensemble World Model Pretraining"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0"
echo ""
echo "Configuration (via Hydra overrides, from Table S10/S11):"
echo "  - Ensemble Size: $ENSEMBLE_SIZE (U-RWM ensemble)"
echo "  - Real Envs: $NUM_ENVS"
echo "  - Max Iterations: $MAX_ITERATIONS"
echo "  - WM Learning Rate: 1e-4"
echo "  - WM Weight Decay: 1e-5"
echo "  - WM Batch Size: 1024"
echo "  - Policy Learning Rate: 0.001"
echo "  - Gamma: 0.99 (default)"
echo "  - Entropy Coef: 0.005 (default)"
echo "  - Uncertainty Penalty: $UNCERTAINTY_PENALTY"
echo "  - Imagination: DISABLED (pretrain phase)"
echo ""
echo "Logging:"
echo "  - Logger: $LOGGER"
echo "  - W&B Project: $WANDB_PROJECT"
echo "  - Run Name: $RUN_NAME"
echo ""
echo "Computational overhead vs single model:"
echo "  - ${ENSEMBLE_SIZE}x world model parameters"
echo "  - ~50-100% longer training time"
echo ""
echo "Expected training time: ~8-12 hours"
echo "Checkpoints will be saved to:"
echo "  logs/rsl_rl/anymal_d_flat/<timestamp>/"
echo "========================================"
echo ""

# Create logs directory if it doesn't exist
mkdir -p /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm

# Run training inside container
# Using Hydra overrides to change ensemble_size without modifying config files
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
    agent.algorithm.system_dynamics_learning_rate=1e-4 \
    agent.algorithm.system_dynamics_weight_decay=1e-5 \
    agent.algorithm.system_dynamics_mini_batch_size=1024 \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.imagination.uncertainty_penalty_weight=$UNCERTAINTY_PENALTY \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo ""
echo "========================================"
echo "Ensemble Pretraining Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: U-RWM ensemble pretraining completed!"
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
    echo "The world model checkpoint contains a ${ENSEMBLE_SIZE}-member ensemble with:"
    echo "  - Epistemic uncertainty quantification"
    echo "  - Bootstrap-trained diverse models"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  Option A: Online finetuning with imagination"
    echo "    sbatch slurm_finetune_rwm.sh"
    echo "    (Update PRETRAIN_DIR to point to this run)"
    echo ""
    echo "  Option B: Offline policy training (U-RWM)"
    echo "    python scripts/reinforcement_learning/model_based/train.py \\"
    echo "      --task anymal_d_flat"
    echo "    (Update resume_path in configs/anymal_d_flat_cfg.py)"
    echo ""
    echo "  Option C: Evaluate pretrained policy"
    echo "    python scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "      --task Isaac-Velocity-Flat-Anymal-D-Play-v0 \\"
    echo "      --checkpoint ${LATEST_LOG}model_${MAX_ITERATIONS}.pt"
else
    echo ""
    echo "FAILED: Ensemble pretraining failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/rwm-ensemble-pretrain-$SLURM_JOB_ID.err"
    echo ""
    echo "Common issues:"
    echo "  - OOM: Reduce NUM_ENVS (ensemble uses ${ENSEMBLE_SIZE}x memory)"
    echo "  - Hydra override syntax error: Check parameter names"
fi

exit $EXITCODE
