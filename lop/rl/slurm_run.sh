#!/bin/bash
#SBATCH --partition=3090-gcondo
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --job-name=ppo_sbp
#SBATCH --output=ppo_cbp-%A_%a.out
#SBATCH --error=ppo_cbp-%A_%a.err
#SBATCH --array=0-9          # 5 configs × 10 seeds = 50 array tasks

module load cuda cudnn
module load mesa
module load python/3.9.16s-x3wdtvt

source ~/envs/lop/bin/activate
export PYTHONPATH="$HOME/ReAI-loss-of-plasticity:$PYTHONPATH"
cd "$HOME/ReAI-loss-of-plasticity/lop/rl"

configs=(cbp)            # 5 variants
variant_index=$(( SLURM_ARRAY_TASK_ID / 10 ))   # 0–4
seed=$(( SLURM_ARRAY_TASK_ID % 10 ))            # 0–9

cfg_file="cfg/ant/${configs[$variant_index]}.yml"

echo "Running ${cfg_file} with seed ${seed} on GPU ${CUDA_VISIBLE_DEVICES}"
python run_ppo.py -c "${cfg_file}" -s "${seed}"
