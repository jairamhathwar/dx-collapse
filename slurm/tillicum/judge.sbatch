#!/bin/bash
# Judge open-ended diagnoses (judge_medcase.py) on one H200.
# Args go to judge_medcase.py.
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
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
cd $THESIS/code/src && python judge_medcase.py "$@"
echo "== done $(date)"
