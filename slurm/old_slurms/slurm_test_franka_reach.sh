#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/%x-%j.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/%x-%j.err

# =============================================================================
# Franka Reach Setup Test
# =============================================================================
# Verifies that the Franka Reach MBRL task is correctly registered and can
# be instantiated. Tests all 4 gym environments:
#   1. Template-Isaac-Reach-Franka-Init-v0
#   2. Template-Isaac-Reach-Franka-Pretrain-v0
#   3. Template-Isaac-Reach-Franka-Finetune-v0
#   4. Template-Isaac-Reach-Franka-Visualize-v0
#
# Usage:
#   sbatch --job-name=fr-test slurm/slurm_test_franka_reach.sh
# =============================================================================

set -e

echo "========================================"
echo "Franka Reach Setup Test"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"
echo ""

cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

mkdir -p logs/slurm

# ---- Test 1: Gym Registration ----
echo "==== Test 1: Gym Registration ===="
echo "Checking that all Franka Reach envs are registered..."
echo ""

# Create temporary test script
cat > /tmp/test_gym_registration.py << 'EOF'
"""Test script to verify Franka Reach env registration."""

import argparse
import sys

# Import AppLauncher BEFORE any isaaclab imports
from isaaclab.app import AppLauncher

# Parse minimal args needed for AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", default=True)
args_cli, _ = parser.parse_known_args()

# Launch omniverse app - THIS MUST HAPPEN FIRST
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW we can import gymnasium and mbrl.tasks (which trigger isaaclab imports)
import gymnasium as gym
import mbrl.tasks  # triggers gym.register() via import_packages()

expected_envs = [
    'Template-Isaac-Reach-Franka-Init-v0',
    'Template-Isaac-Reach-Franka-Pretrain-v0',
    'Template-Isaac-Reach-Franka-Finetune-v0',
    'Template-Isaac-Reach-Franka-Visualize-v0',
]

print('Registered Franka Reach environments:')
all_envs = [spec.id for spec in gym.registry.values()]
found = 0
for env_id in expected_envs:
    if env_id in all_envs:
        print(f'  [OK] {env_id}')
        found += 1
    else:
        print(f'  [MISSING] {env_id}')

print()
if found == len(expected_envs):
    print(f'SUCCESS: All {found}/{len(expected_envs)} environments registered.')
    # Close simulation app cleanly
    simulation_app.close()
else:
    print(f'FAILURE: Only {found}/{len(expected_envs)} environments registered.')
    simulation_app.close()
    exit(1)
EOF

apptainer exec --nv \
    --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
    --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
    --bind $HOME:$HOME \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env GIT_PYTHON_REFRESH=quiet \
    /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
    /isaac-sim/python.sh /tmp/test_gym_registration.py

EXITCODE_REG=$?
echo ""

if [ $EXITCODE_REG -ne 0 ]; then
    echo "FAILED: Gym registration test failed."
    exit 1
fi

# ---- Test 2: Short Pretrain Run (2 iterations) ----
echo "==== Test 2: Short Pretrain Run (2 iterations) ===="
echo "Running pretrain for 2 iterations with 16 envs to verify end-to-end..."
echo ""

apptainer exec --nv \
    --bind /projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo \
    --bind /gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo \
    --bind $HOME:$HOME \
    --env OMNI_KIT_ACCEPT_EULA=YES \
    --env GIT_PYTHON_REFRESH=quiet \
    /gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif \
    /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
    --task Template-Isaac-Reach-Franka-Pretrain-v0 \
    --headless \
    --num_envs 16 \
    --max_iterations 2 \
    --logger tensorboard \
    --log_project_name rwm-franka-reach-test \
    --run_name fr-test-pretrain \
    agent.system_dynamics_num_visualizations=0

EXITCODE_PRETRAIN=$?
echo ""

if [ $EXITCODE_PRETRAIN -ne 0 ]; then
    echo "FAILED: Short pretrain run failed with exit code $EXITCODE_PRETRAIN"
    exit 1
fi

echo "========================================"
echo "All Tests Passed!"
echo "End Time: $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run pretrain:  sbatch --job-name=fr-pretrain-latent slurm/slurm_train.sh slurm/configs/franka_reach_pretrain_latent.yaml"
echo "  2. Run finetune:  sbatch --job-name=fr-finetune-latent slurm/slurm_train.sh slurm/configs/franka_reach_finetune_latent.yaml"

exit 0
