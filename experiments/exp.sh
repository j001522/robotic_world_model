#!/bin/bash
# Helper script for experiment tracking
# Usage: ./exp.sh <command> [args]

EXPERIMENTS_DIR="/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/experiments"
EXPERIMENTS_CSV="$EXPERIMENTS_DIR/experiments.csv"
LOGS_DIR="/gpfs/work4/0/prjs0951/Giacomo/robotic_world_model/logs/rsl_rl/anymal_d_flat"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

case "$1" in
    list|ls)
        echo -e "${BLUE}=== Experiment Log ===${NC}"
        if [ -f "$EXPERIMENTS_CSV" ]; then
            column -t -s',' "$EXPERIMENTS_CSV" | head -1
            echo "---"
            column -t -s',' "$EXPERIMENTS_CSV" | tail -n +2
        else
            echo "No experiments logged yet."
        fi
        ;;
    
    running|r)
        echo -e "${BLUE}=== Running Jobs ===${NC}"
        squeue -u $USER -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null
        echo ""
        echo -e "${BLUE}=== Recent Log Directories ===${NC}"
        ls -lt "$LOGS_DIR" 2>/dev/null | head -6
        ;;
    
    progress|p)
        # Show progress of most recent run
        LATEST=$(ls -td "$LOGS_DIR"/*/ 2>/dev/null | head -1)
        if [ -n "$LATEST" ]; then
            echo -e "${BLUE}=== Latest Run: $(basename $LATEST) ===${NC}"
            echo "Checkpoints:"
            ls -lh "$LATEST"model_*.pt 2>/dev/null | tail -5
            echo ""
            LATEST_CKPT=$(ls -t "$LATEST"model_*.pt 2>/dev/null | head -1)
            if [ -n "$LATEST_CKPT" ]; then
                ITER=$(basename "$LATEST_CKPT" | sed 's/model_\([0-9]*\).pt/\1/')
                echo -e "Current iteration: ${GREEN}$ITER${NC} / 2500"
            fi
        else
            echo "No runs found."
        fi
        ;;
    
    add|a)
        # Add new experiment: ./exp.sh add <slurm_job_id> <type> <description>
        if [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then
            echo "Usage: ./exp.sh add <slurm_job_id> <type> <description>"
            echo "  type: pretrain-single, pretrain-ensemble, finetune-single, finetune-ensemble"
            exit 1
        fi
        
        # Find next experiment ID
        LAST_ID=$(tail -1 "$EXPERIMENTS_CSV" 2>/dev/null | cut -d',' -f1 | grep -oE '[0-9]+' || echo "0")
        NEXT_ID=$(printf "exp%03d" $((10#$LAST_ID + 1)))
        
        # Find log directory for this job
        SLURM_JOB=$2
        TYPE=$3
        DESC=$4
        DATE=$(date +%Y-%m-%d)
        
        # Try to find log dir (might not exist yet)
        LOG_DIR=$(ls -td "$LOGS_DIR"/*/ 2>/dev/null | head -1 | sed "s|$LOGS_DIR/||" | tr -d '/')
        
        echo "$NEXT_ID,$DATE,$SLURM_JOB,$TYPE,running,$LOG_DIR,$DESC,,," >> "$EXPERIMENTS_CSV"
        echo -e "${GREEN}Added experiment $NEXT_ID${NC}"
        ;;
    
    update|u)
        # Update experiment status: ./exp.sh update <exp_id> <status> [final_reward] [notes]
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Usage: ./exp.sh update <exp_id> <status> [final_reward] [notes]"
            echo "  status: completed, failed, cancelled"
            exit 1
        fi
        echo -e "${YELLOW}Manual edit required. Open:${NC}"
        echo "  $EXPERIMENTS_CSV"
        echo "Find experiment $2 and update status to $3"
        ;;
    
    tensorboard|tb)
        echo -e "${BLUE}=== TensorBoard ===${NC}"
        echo "To view training curves, run on your local machine:"
        echo ""
        echo "  # Option 1: SSH tunnel"
        echo "  ssh -L 6006:localhost:6006 cmeo@snellius.surf.nl"
        echo "  # Then on Snellius:"
        echo "  tensorboard --logdir $LOGS_DIR --port 6006"
        echo ""
        echo "  # Option 2: Download and view locally"
        echo "  scp -r cmeo@snellius.surf.nl:$LOGS_DIR/<run_name>/events* ."
        echo "  tensorboard --logdir ."
        ;;
    
    summary|s)
        # Show summary of a specific run
        if [ -z "$2" ]; then
            echo "Usage: ./exp.sh summary <log_dir_name>"
            echo "Available runs:"
            ls "$LOGS_DIR" 2>/dev/null
            exit 1
        fi
        RUN_DIR="$LOGS_DIR/$2"
        if [ -d "$RUN_DIR" ]; then
            echo -e "${BLUE}=== Run Summary: $2 ===${NC}"
            echo ""
            echo "Checkpoints:"
            ls -lh "$RUN_DIR"model_*.pt 2>/dev/null
            echo ""
            echo "Config (agent.yaml):"
            head -50 "$RUN_DIR/params/agent.yaml" 2>/dev/null
        else
            echo "Run not found: $2"
        fi
        ;;
    
    *)
        echo "Experiment Tracking Helper"
        echo ""
        echo "Usage: ./exp.sh <command>"
        echo ""
        echo "Commands:"
        echo "  list, ls        Show all experiments"
        echo "  running, r      Show running SLURM jobs"
        echo "  progress, p     Show progress of latest run"
        echo "  add, a          Add new experiment to log"
        echo "  update, u       Update experiment status"
        echo "  tensorboard, tb Show TensorBoard instructions"
        echo "  summary, s      Show summary of a run"
        ;;
esac
