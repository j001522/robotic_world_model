#!/bin/bash
#SBATCH --job-name=rwm-pretrain
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-pretrain-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/rwm-pretrain-%j.err

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

# ======================= CONFIGURATION =======================
# Weights & Biases logging (set to "tensorboard" to disable wandb)
LOGGER="tensorboard"
WANDB_PROJECT="rwm-anymal"

# Run name for easy identification (will appear in wandb dashboard)
RUN_NAME="pretrain-single"
# =============================================================

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
echo "Classic RWM Training (Paper Parameters)"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0"
echo "Configuration (from Table S10/S11):"
echo "  - Ensemble Size: 1 (classic RWM, no ensemble)"
echo "  - Real Envs: 4096"
echo "  - Max Iterations: 2500 (paper)"
echo "  - WM Learning Rate: 1e-4"
echo "  - WM Weight Decay: 1e-5"
echo "  - WM Batch Size: 1024"
echo "  - Policy Learning Rate: 0.001"
echo "  - Gamma: 0.99 (default)"
echo "  - Entropy Coef: 0.005 (default)"
echo "  - Imagination: DISABLED (pretrain phase)"
echo ""
echo "Logging:"
echo "  - Logger: $LOGGER"
echo "  - W&B Project: $WANDB_PROJECT"
echo "  - Run Name: $RUN_NAME"
echo ""
echo "Expected training time: ~5-6 hours"
echo "Checkpoints will be saved to:"
echo "  logs/rsl_rl/Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0/<timestamp>/"
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
    --num_envs 4096 \
    --max_iterations 2500 \
    --logger $LOGGER \
    --log_project_name $WANDB_PROJECT \
    --run_name $RUN_NAME \
    agent.algorithm.system_dynamics_learning_rate=1e-4 \
    agent.algorithm.system_dynamics_weight_decay=1e-5 \
    agent.algorithm.system_dynamics_mini_batch_size=1024 \
    agent.algorithm.policy_learning_rate=0.001 \
    agent.system_dynamics_num_visualizations=0

EXITCODE=$?

echo ""
echo "========================================"
echo "Training Completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: RWM training completed!"
    echo ""
    echo "Checkpoint location:"
    LATEST_LOG=$(ls -td logs/rsl_rl/Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  $LATEST_LOG"
        echo ""
        echo "Files saved:"
        ls -lh "$LATEST_LOG"*.pt 2>/dev/null | tail -5
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate policy:"
    echo "     python scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "       --task Isaac-Velocity-Flat-Anymal-D-Play-v0 \\"
    echo "       --checkpoint $LATEST_LOG/model_2500.pt \\"
    echo "       --video --video_length 400 --headless"
    echo ""
    echo "  2. Visualize world model (real vs imagination):"
    echo "     python scripts/reinforcement_learning/rsl_rl/visualize.py \\"
    echo "       --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \\"
    echo "       --checkpoint $LATEST_LOG/model_2500.pt \\"
    echo "       --system_dynamics_load_path $LATEST_LOG/system_dynamics_2500.pt \\"
    echo "       --video --headless"
else
    echo ""
    echo "❌ FAILED: Training failed with exit code $EXITCODE"
    echo ""
echo "Check the error log:"
echo "  logs/slurm/rwm-pretrain-$SLURM_JOB_ID.err"
fi

exit $EXITCODE
