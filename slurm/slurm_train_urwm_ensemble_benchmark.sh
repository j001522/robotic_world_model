#!/bin/bash
#SBATCH --job-name=urwm-ensemble-benchmark
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/urwm-ensemble-benchmark-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/urwm-ensemble-benchmark-%j.err

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
echo "U-RWM Ensemble Training (Benchmark Run)"
echo "========================================"
echo "Task: Template-Isaac-Velocity-Flat-Anymal-D-Ensemble-v0"
echo "Configuration:"
echo "  - Ensemble Size: 5 (U-RWM ensemble)"
echo "  - Imagination Envs: 4096"
echo "  - Imagination Steps: 24"
echo "  - Real Envs: 4096"
echo "  - Max Iterations: 5000"
echo "  - Uncertainty Penalty: -0.1 (enabled)"
echo ""
echo "Expected training time: ~15-20 hours (50-100% longer than single model)"
echo "Computational overhead from ensemble:"
echo "  - 5x world model parameters"
echo "  - Ensemble forward passes"
echo "  - Uncertainty computation"
echo ""
echo "Checkpoints will be saved to:"
echo "  logs/rsl_rl/Template-Isaac-Velocity-Flat-Anymal-D-Ensemble-v0/<timestamp>/"
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
    --task Template-Isaac-Velocity-Flat-Anymal-D-Ensemble-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 5000 \
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
    echo "✅ SUCCESS: U-RWM ensemble training completed!"
    echo ""
    echo "Checkpoint location:"
    LATEST_LOG=$(ls -td logs/rsl_rl/Template-Isaac-Velocity-Flat-Anymal-D-Ensemble-v0/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  $LATEST_LOG"
        echo ""
        echo "Files saved:"
        ls -lh "$LATEST_LOG"*.pt 2>/dev/null | tail -5
    fi
    echo ""
    echo "The world model checkpoint contains a 5-member ensemble with:"
    echo "  - Epistemic uncertainty quantification"
    echo "  - Improved dynamics prediction"
    echo "  - Uncertainty-penalized policy training"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate ensemble policy:"
    echo "     python scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "       --task Isaac-Velocity-Flat-Anymal-D-Play-v0 \\"
    echo "       --checkpoint $LATEST_LOG/model_5000.pt \\"
    echo "       --video --video_length 400 --headless"
    echo ""
    echo "  2. Compare uncertainty with single-model RWM"
    echo ""
    echo "  3. Use this ensemble for offline policy training (U-RWM Stage 2)"
else
    echo ""
    echo "❌ FAILED: Training failed with exit code $EXITCODE"
    echo ""
    echo "Check the error log:"
    echo "  logs/slurm/urwm-ensemble-benchmark-$SLURM_JOB_ID.err"
fi

exit $EXITCODE
