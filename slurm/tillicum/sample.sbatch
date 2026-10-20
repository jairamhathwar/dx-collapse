#!/bin/bash
# Sampling/eval on a MIG slice (1/7 H200, 18 GB; ~1/7 the cost). Fits <=4B models.
# Args go to sample_diagnoses.py; supports job arrays with --shard -1 and {shard} in --out.
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%A_%a.out
echo "== $(date) host=$(hostname)"
source ~/.thesis_env || echo "warn: thesis_env returned $?"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis" || { echo "conda activate failed"; exit 1; }
export CUDA_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # vLLM 0.3x forks after CUDA init otherwise
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$THESIS"
python $THESIS/code/src/sample_diagnoses.py "$@"
echo "== done $(date)"
