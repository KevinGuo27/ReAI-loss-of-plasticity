#!/bin/bash
#SBATCH --partition=3090-gcondo
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --job-name=run
#SBATCH --output=run_bp-%A_%a.out   # %A = master job ID, %a = array index
#SBATCH --error=run_bp-%A_%a.err
#SBATCH --array=0-2              # launches 9 tasks with IDs 0,1,...,8

# activate your environment
source ~/envs/lop/bin/activate
export PYTHONPATH="$HOME/ReAI-loss-of-plasticity:$PYTHONPATH"

# go to the folder containing online_expr.py and temp_cfg/*
cd $HOME/ReAI-loss-of-plasticity/lop/permuted_mnist

# pick config based on the Slurm array index
CONFIG="temp_cfg/${SLURM_ARRAY_TASK_ID}.json"

# run
python online_expr.py -c "$CONFIG"
