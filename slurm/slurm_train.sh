#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/%x-%A-%a.out
#SBATCH --error=/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/slurm/%x-%A-%a.err

# =============================================================================
# Unified Training Launcher for RWM (Job Array Version)
# =============================================================================
# Usage:
#   sbatch --job-name=pretrain-latent --partition=gpu_a100 --time=24:00:00 \
#          --array=0-2 slurm_train.sh configs/pretrain_latent.yaml
#
# The array range (0-2) should match the number of seeds in your config.
# For seeds: [42, 123, 1001], use --array=0-2
# =============================================================================

set -e  # Exit on error

# Parse arguments
CONFIG_FILE="$1"

if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: No config file specified"
    echo "Usage: sbatch slurm_train.sh <config.yaml>"
    echo ""
    echo "Available configs:"
    ls -1 "$(dirname "$0")/configs/" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Convert to absolute path before cd
# NOTE: When running via sbatch, $0 is in /var/spool/slurm, so we hardcode the path
if [[ "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE_ABS="$CONFIG_FILE"
else
    # Relative path - resolve relative to the slurm directory
    # The script is in /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/slurm/
    SCRIPT_DIR="/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/slurm"
    CONFIG_FILE_ABS="$SCRIPT_DIR/$CONFIG_FILE"
fi

if [ ! -f "$CONFIG_FILE_ABS" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE_ABS"
    exit 1
fi

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"
echo ""
echo "Config: $CONFIG_FILE_ABS"
echo ""

# Change to robotic_world_model directory
cd /gpfs/work4/0/prjs0951/Giacomo/robotic_world_model

# Environment setup
export ISAAC_SIM_CACHE_DIR=$HOME/isaac-sim/cache
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_PYTHON_REFRESH=quiet

# Create logs directory
mkdir -p logs/slurm

# Source the config to get variables
# The config YAML will be parsed below using simple shell commands
TASK=$(grep "^task:" "$CONFIG_FILE_ABS" | sed 's/task: *//' | sed 's/ *#.*$//' | xargs)
MODE=$(grep "^mode:" "$CONFIG_FILE_ABS" | sed 's/mode: *//' | sed 's/ *#.*$//' | xargs)
MAX_ITERATIONS=$(grep "^max_iterations:" "$CONFIG_FILE_ABS" | sed 's/max_iterations: *//' | sed 's/ *#.*$//' | xargs)
NUM_ENVS=$(grep "num_envs:" "$CONFIG_FILE_ABS" | grep -v "imagination:" | head -1 | sed 's/.*num_envs: *//' | sed 's/ *#.*$//' | xargs)
LOGGER=$(grep "logger:" "$CONFIG_FILE_ABS" | head -1 | sed 's/.*logger: *//' | sed 's/ *#.*$//' | xargs)
WANDB_PROJECT=$(grep "wandb_project:" "$CONFIG_FILE_ABS" | sed 's/.*wandb_project: *//' | sed 's/ *#.*$//' | xargs)
RUN_NAME=$(grep "^run_name:" "$CONFIG_FILE_ABS" | sed 's/run_name: *//' | sed 's/ *#.*$//' | xargs)
SEEDS_LINE=$(grep "^seeds:" "$CONFIG_FILE_ABS" | sed 's/seeds: *//' | sed 's/ *#.*$//' | sed 's/[][]//g' | sed 's/,/ /g' | xargs)

# Default values
[ -z "$TASK" ] && TASK="Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0"
[ -z "$MODE" ] && MODE="pretrain"
[ -z "$MAX_ITERATIONS" ] && MAX_ITERATIONS=2500
[ -z "$NUM_ENVS" ] && NUM_ENVS=4096
[ -z "$LOGGER" ] && LOGGER="tensorboard"
[ -z "$WANDB_PROJECT" ] && WANDB_PROJECT="rwm-anymal"
[ -z "$RUN_NAME" ] && RUN_NAME="${MODE}-${SLURM_JOB_ID}"

# Parse seeds (default to [42] if not specified)
if [ -n "$SEEDS_LINE" ]; then
    SEEDS=($SEEDS_LINE)
else
    SEEDS=(42)
fi

# Check if running as job array
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: This script must be submitted as a job array"
    echo "Usage: sbatch --array=0-$((${#SEEDS[@]}-1)) slurm_train.sh <config.yaml>"
    echo "  where the array range is 0 to (num_seeds - 1)"
    exit 1
fi

# Select seed based on array task ID
ARRAY_ID=$SLURM_ARRAY_TASK_ID
if [ $ARRAY_ID -lt 0 ] || [ $ARRAY_ID -ge ${#SEEDS[@]} ]; then
    echo "ERROR: Array task ID $ARRAY_ID is out of range"
    echo "Seeds: ${SEEDS[@]}"
    echo "Valid range: 0 to $((${#SEEDS[@]}-1))"
    exit 1
fi

SEED=${SEEDS[$ARRAY_ID]}

echo "Parsed config:"
echo "  Mode: $MODE"
echo "  Task: $TASK"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Num Envs: $NUM_ENVS"
echo "  Logger: $LOGGER"
echo "  Run Name: $RUN_NAME"
echo "  Seeds from config: ${SEEDS[@]}"
echo "  Array Task ID: $ARRAY_ID"
echo "  Selected Seed: $SEED"
echo ""

# Build hydra overrides from config using array
declare -a OVERRIDES=()

# Helper function to add override
add_override() {
    local key="$1"
    # Get value and strip inline comments (everything after #)
    local value=$(grep -E "^  $key:" "$CONFIG_FILE_ABS" 2>/dev/null | head -1 | sed "s/  $key: *//" | sed 's/ *#.*$//' | xargs)
    if [ -n "$value" ]; then
        OVERRIDES+=("$2=$value")
    fi
}

# Model overrides
add_override "ensemble_size" "agent.system_dynamics.ensemble_size"
add_override "latent_mode" "agent.system_dynamics.latent_mode"
add_override "latent_dim" "agent.system_dynamics.latent_dim"
add_override "simnorm_dim" "agent.system_dynamics.simnorm_dim"
add_override "prior_scale" "agent.system_dynamics.prior_scale"
add_override "prior_hidden_div" "agent.system_dynamics.prior_hidden_div"
add_override "bootstrap" "agent.system_dynamics.bootstrap"
add_override "uncertainty_metric" "agent.system_dynamics.uncertainty_metric"
add_override "consistency_coef" "agent.system_dynamics.consistency_coef"
add_override "reconstruction_coef" "agent.system_dynamics.reconstruction_coef"
add_override "target_encoder_momentum" "agent.system_dynamics.target_encoder_momentum"
add_override "encoder_consistency_coef" "agent.system_dynamics.encoder_consistency_coef"
add_override "encoder_dropout" "agent.system_dynamics.encoder_dropout"

# Phase 3: Reward and Value head overrides
add_override "reward_head_enabled" "agent.system_dynamics.reward_head_enabled"
add_override "value_head_enabled" "agent.system_dynamics.value_head_enabled"
add_override "value_ensemble_size" "agent.system_dynamics.value_ensemble_size"
add_override "reward_prior_scale" "agent.system_dynamics.reward_prior_scale"
add_override "value_prior_scale" "agent.system_dynamics.value_prior_scale"
add_override "value_gamma" "agent.system_dynamics.value_gamma"

# Architecture config overrides (nested under architecture: in YAML)
# All architecture keys exist in the base config dict (rsl_rl_ppo_cfg.py)
add_arch_override() {
    local key="$1"
    local value=$(grep -E "^  $key:" "$CONFIG_FILE_ABS" 2>/dev/null | head -1 | sed "s/  $key: *//" | sed 's/ *#.*$//' | xargs)
    if [ -n "$value" ]; then
        OVERRIDES+=("agent.system_dynamics.architecture_config.$key=$value")
    fi
}

# Architecture type (backbone selection: mlp, rnn, xlstm) - exists in base config
add_arch_override "type"

# xLSTM-specific architecture overrides (keys exist in base config with defaults)
add_arch_override "xlstm_embedding_dim"
add_arch_override "xlstm_num_blocks"
add_arch_override "xlstm_num_heads"
add_arch_override "xlstm_context_length"
add_arch_override "xlstm_conv1d_kernel_size"
add_arch_override "xlstm_slstm_backend"
add_arch_override "xlstm_proj_factor_mlstm"
add_arch_override "xlstm_proj_factor_slstm_ff"
add_arch_override "xlstm_dropout"

# xLSTM slstm_at is a list, needs special handling
XLSTM_SLSTM_AT=$(grep "xlstm_slstm_at:" "$CONFIG_FILE_ABS" | sed 's/.*xlstm_slstm_at: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$XLSTM_SLSTM_AT" ]; then
    OVERRIDES+=("agent.system_dynamics.architecture_config.xlstm_slstm_at=$XLSTM_SLSTM_AT")
fi

# Head shape overrides (shared across all backbone types) - exist in base config
add_arch_override "state_mean_shape"
add_arch_override "state_logstd_shape"
add_arch_override "extension_shape"
add_arch_override "contact_shape"
add_arch_override "termination_shape"

# Training overrides
add_override "wm_learning_rate" "agent.algorithm.system_dynamics_learning_rate"
add_override "wm_weight_decay" "agent.algorithm.system_dynamics_weight_decay"
add_override "wm_batch_size" "agent.algorithm.system_dynamics_mini_batch_size"
add_override "policy_learning_rate" "agent.algorithm.policy_learning_rate"

# Imagination overrides (for finetune)
if [ "$MODE" = "finetune" ]; then
    add_override "uncertainty_penalty_weight" "agent.imagination.uncertainty_penalty_weight"
fi

# List overrides (need special handling for brackets)
ENCODER_HIDDEN=$(grep "encoder_hidden_dims:" "$CONFIG_FILE_ABS" | sed 's/.*encoder_hidden_dims: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$ENCODER_HIDDEN" ]; then
    OVERRIDES+=("agent.system_dynamics.encoder_hidden_dims=$ENCODER_HIDDEN")
fi

DECODER_HIDDEN=$(grep "decoder_hidden_dims:" "$CONFIG_FILE_ABS" | sed 's/.*decoder_hidden_dims: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$DECODER_HIDDEN" ]; then
    OVERRIDES+=("agent.system_dynamics.decoder_hidden_dims=$DECODER_HIDDEN")
fi

LATENT_HEAD_HIDDEN=$(grep "latent_head_hidden_dims:" "$CONFIG_FILE_ABS" | sed 's/.*latent_head_hidden_dims: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$LATENT_HEAD_HIDDEN" ]; then
    OVERRIDES+=("agent.system_dynamics.latent_head_hidden_dims=$LATENT_HEAD_HIDDEN")
fi

# Phase 3: Reward/Value head hidden dims (list overrides)
REWARD_HIDDEN=$(grep "reward_hidden_dims:" "$CONFIG_FILE_ABS" | sed 's/.*reward_hidden_dims: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$REWARD_HIDDEN" ]; then
    OVERRIDES+=("agent.system_dynamics.reward_hidden_dims=$REWARD_HIDDEN")
fi

VALUE_HIDDEN=$(grep "value_hidden_dims:" "$CONFIG_FILE_ABS" | sed 's/.*value_hidden_dims: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$VALUE_HIDDEN" ]; then
    OVERRIDES+=("agent.system_dynamics.value_hidden_dims=$VALUE_HIDDEN")
fi

# Phase 3: Reward/Value loss weights (override individual keys in the dict)
REWARD_LOSS_WEIGHT=$(grep "reward_loss_weight:" "$CONFIG_FILE_ABS" | sed 's/.*reward_loss_weight: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$REWARD_LOSS_WEIGHT" ]; then
    OVERRIDES+=("agent.algorithm.system_dynamics_loss_weights.reward=$REWARD_LOSS_WEIGHT")
fi

VALUE_LOSS_WEIGHT=$(grep "value_loss_weight:" "$CONFIG_FILE_ABS" | sed 's/.*value_loss_weight: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$VALUE_LOSS_WEIGHT" ]; then
    OVERRIDES+=("agent.algorithm.system_dynamics_loss_weights.value=$VALUE_LOSS_WEIGHT")
fi

# Phase 3: Reward/Value warmup delay (number of optimizer steps before RV losses activate)
RV_WARMUP_STEPS=$(grep "rv_warmup_steps:" "$CONFIG_FILE_ABS" | sed 's/.*rv_warmup_steps: *//' | sed 's/ *#.*$//' | xargs)
if [ -n "$RV_WARMUP_STEPS" ]; then
    OVERRIDES+=("agent.algorithm.system_dynamics_rv_warmup_steps=$RV_WARMUP_STEPS")
fi

    # Handle checkpoint loading for finetune
    if [ "$MODE" = "finetune" ]; then
        RESUME=$(grep "resume:" "$CONFIG_FILE_ABS" | sed 's/resume: *//' | sed 's/ *#.*$//' | xargs)
        CHECKPOINT=$(grep "^  checkpoint:" "$CONFIG_FILE_ABS" | sed 's/.*checkpoint: *//' | sed 's/ *#.*$//' | tr -d '"' | xargs)

        # Check if using seed-aware loading
        PRETRAIN_RUN_NAME=$(grep "pretrain_run_name:" "$CONFIG_FILE_ABS" | sed 's/pretrain_run_name: *//' | sed 's/ *#.*$//' | tr -d '"' | xargs)

    if [ -n "$PRETRAIN_RUN_NAME" ]; then
        # Seed-aware mode: will find pretrain for this seed
        USE_SEED_AWARE_LOADING=true
        
        # Try to auto-detect the log directory
        # Extract task name and map to log directory
        if [[ "$TASK" == *"-Anymal-D-"* ]]; then
            PRETRAIN_LOGS_DIR="logs/rsl_rl/anymal_d_flat/"
        elif [[ "$TASK" == *"-Franka-"* ]]; then
            PRETRAIN_LOGS_DIR="logs/rsl_rl/franka_reach/"
        else
            # Fallback: try to find directory by searching for pretrain_run_name
            PRETRAIN_LOGS_DIR=$(find logs/rsl_rl -type d -name "*$PRETRAIN_RUN_NAME*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null | xargs dirname 2>/dev/null)/
        fi
        
        # Check if directory exists
        if [ ! -d "$PRETRAIN_LOGS_DIR" ]; then
            echo ""
            echo "WARNING: Could not auto-detect log directory for task: $TASK"
            echo "Please specify 'pretrain_logs_dir' in the checkpoint section of your config"
            echo ""
            PRETRAIN_LOGS_DIR=""
        fi
    else
        # Manual mode: use the old approach with explicit load_run and system_dynamics_load_path
        LOAD_RUN=$(grep "load_run:" "$CONFIG_FILE_ABS" | sed 's/load_run: *//' | sed 's/ *#.*$//' | tr -d '"' | xargs)
        SD_LOAD_PATH=$(grep "system_dynamics_load_path:" "$CONFIG_FILE_ABS" | sed 's/system_dynamics_load_path: *//' | sed 's/ *#.*$//' | tr -d '"' | xargs)
        USE_SEED_AWARE_LOADING=false
    fi
fi

echo "Overrides: ${OVERRIDES[*]}"
echo ""

# Track exit code
EXITCODE=0

echo "========================================"
echo "Launching training for SEED=$SEED..."
echo "========================================"
echo ""

# Update RUN_NAME to include seed
CURRENT_RUN_NAME="${RUN_NAME}-seed${SEED}"

# Handle seed-aware checkpoint loading for finetune
if [ "$MODE" = "finetune" ] && [ "$USE_SEED_AWARE_LOADING" = true ]; then
    # Check if pretrain_logs_dir is explicitly specified in config
    PRETRAIN_LOGS_DIR_EXPlicit=$(grep "pretrain_logs_dir:" "$CONFIG_FILE_ABS" | sed 's/pretrain_logs_dir: *//' | sed 's/ *#.*$//' | tr -d '"' | xargs)
    if [ -n "$PRETRAIN_LOGS_DIR_EXPlicit" ]; then
        PRETRAIN_LOGS_DIR="$PRETRAIN_LOGS_DIR_EXPlicit"
    fi

    # Validate log directory exists
    if [ -z "$PRETRAIN_LOGS_DIR" ]; then
        echo ""
        echo "ERROR: Pretrain log directory not specified or could not be auto-detected"
        echo "Please add 'pretrain_logs_dir: logs/rsl_rl/<your_task>' to checkpoint section"
        exit 1
    fi

    # Find pretrain directory for this seed
    SEARCH_PATTERN="${PRETRAIN_RUN_NAME}-seed${SEED}"
    PRETRAIN_DIR=$(ls -td "$PRETRAIN_LOGS_DIR"*$SEARCH_PATTERN 2>/dev/null | head -1)

    if [ -z "$PRETRAIN_DIR" ]; then
        echo ""
        echo "ERROR: Pretrain directory not found for seed=$SEED"
        echo "Searched for pattern: $SEARCH_PATTERN"
        echo "In directory: $PRETRAIN_LOGS_DIR"
        echo ""
        echo "Available pretrain directories:"
        ls -td "$PRETRAIN_LOGS_DIR"* 2>/dev/null | grep "$PRETRAIN_RUN_NAME" || echo "  (none found)"
        exit 1
    fi

    LOAD_RUN=$(basename "$PRETRAIN_DIR")
    SD_LOAD_PATH="${PRETRAIN_DIR}/${CHECKPOINT}"

    # Validate checkpoint exists
    if [ ! -f "$SD_LOAD_PATH" ]; then
        echo ""
        echo "ERROR: Checkpoint not found at: $SD_LOAD_PATH"
        echo "Pretrain dir: $PRETRAIN_DIR"
        exit 1
    fi

    echo "Seed-aware checkpoint loading enabled:"
    echo "  Pretrain run name: $PRETRAIN_RUN_NAME"
    echo "  Finetune seed: $SEED"
    echo "  Matched pretrain dir: $PRETRAIN_DIR"
    echo "  Checkpoint: $SD_LOAD_PATH"
    echo ""
fi

# Build command using array to avoid newline issues
declare -a CMD_ARGS=(
        "apptainer" "exec" "--nv"
        "--bind" "/projects/0/prjs0951/Giacomo:/projects/0/prjs0951/Giacomo"
        "--bind" "/gpfs/work4/0/prjs0951/Giacomo:/gpfs/work4/0/prjs0951/Giacomo"
        "--bind" "$HOME:$HOME"
        "--env" "OMNI_KIT_ACCEPT_EULA=YES"
        "--env" "GIT_PYTHON_REFRESH=quiet"
        "/gpfs/work4/0/prjs0951/Giacomo/isaac-sim/containers/isaac-sim_5.1.0.sif"
        "/isaac-sim/python.sh" "scripts/reinforcement_learning/rsl_rl/train.py"
        "--task" "$TASK"
        "--headless"
        "--num_envs" "$NUM_ENVS"
        "--max_iterations" "$MAX_ITERATIONS"
        "--logger" "$LOGGER"
        "--log_project_name" "$WANDB_PROJECT"
        "--run_name" "$CURRENT_RUN_NAME"
        "--seed" "$SEED"
    )

    # Add finetune-specific args
    if [ "$MODE" = "finetune" ] && [ -n "$LOAD_RUN" ]; then
        CMD_ARGS+=("--resume" "True")
        CMD_ARGS+=("--load_run" "$LOAD_RUN")
        CMD_ARGS+=("--checkpoint" "$CHECKPOINT")
        CMD_ARGS+=("--system_dynamics_load_path" "$SD_LOAD_PATH")
    fi

    # Add all overrides
    for override in "${OVERRIDES[@]}"; do
        CMD_ARGS+=("$override")
    done

    # Always disable visualizations
    CMD_ARGS+=("agent.system_dynamics_num_visualizations=0")

# Execute the command
echo "Running command:"
printf '%q ' "${CMD_ARGS[@]}"
echo ""
echo ""
"${CMD_ARGS[@]}"

EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "FAILED for SEED=$SEED"
    echo "Exit Code: $EXITCODE"
    echo "========================================"
    echo ""
else
    echo ""
    echo "========================================"
    echo "SUCCESS for SEED=$SEED"
    echo "End Time: $(date)"
    echo "========================================"
    echo ""
fi

echo ""
echo "========================================"
echo "Training completed"
echo "Exit Code: $EXITCODE"
echo "End Time: $(date)"
echo "========================================"

if [ $EXITCODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: Training run completed!"
    LATEST_LOG="logs/rsl_rl/*/$CURRENT_RUN_NAME"
    echo "Run directory: $LATEST_LOG"
else
    echo ""
    echo "FAILED: Training run failed"
    echo "Check error log: logs/slurm/$SLURM_JOB_NAME-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID.err"
fi

exit $EXITCODE
