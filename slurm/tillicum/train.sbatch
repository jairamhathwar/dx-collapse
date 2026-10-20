#!/bin/bash
# GRPO training on one full H200 (Tillicum, billed ~$0.90/GPU-hour). Args go to train_grpo.py.
#   sbatch --job-name=t-gm slurm/tillicum/train.sbatch --rewards acc,group_match --seed 0 --out $THESIS/runs/train/...
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%j.out
echo "== $(date) host=$(hostname)"
source ~/.thesis_env || echo "warn: thesis_env returned $?"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis" || { echo "conda activate failed"; exit 1; }
export CUDA_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # vLLM 0.3x forks after CUDA init otherwise
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$THESIS/code/src"
python train_grpo.py --model Qwen/Qwen3-1.7B --train $THESIS/data/ddxplus/cases_train.jsonl \
  --conditions $THESIS/data/ddxplus/conditions.json "$@"
echo "== done $(date)"
