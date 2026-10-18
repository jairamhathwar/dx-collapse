#!/bin/bash
# GRPO training job. All args after the script go to train_grpo.py, e.g.
#   sbatch --job-name=grpo-acc slurm/train.sbatch --model Qwen/Qwen3-1.7B --rewards acc \
#       --train $THESIS/data/ddxplus/cases_train.jsonl --conditions $THESIS/data/ddxplus/conditions.json \
#       --out $THESIS/runs/train/q3-1.7b-acc
# Preemptible checkpoint pool; on requeue, train_grpo.py resumes from the last checkpoint in --out.
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=1
#SBATCH --constraint=l40s|l40|a40|a100|h200
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/scrubbed/jairamh/thesis/logs/%x-%j.out
echo "== $(date) host=$(hostname) restart=${SLURM_RESTART_COUNT:-0}"
source ~/.thesis_env || echo "warn: thesis_env returned $?"
module load conda/Miniforge3-25.9.1-0 || { echo "module load conda failed"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis" || { echo "conda activate failed"; exit 1; }
export CUDA_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$THESIS/code/src"
python train_grpo.py "$@"
echo "== done $(date)"
