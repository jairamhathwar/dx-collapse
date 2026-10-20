#!/bin/bash
# Build the thesis Python env on Tillicum (vLLM pinned to Klone's 0.31.0) and smoke-test vLLM.
# Runs under the free debug QOS (1 h max) on a full H200.
#SBATCH --account=stf
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=build-env
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
ENV=$THESIS/envs/thesis
[ -d "$ENV" ] || conda create -y -q -p $ENV python=3.11
conda activate $ENV || exit 1
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m pip install -q -U pip
python -m pip install -q vllm==0.31.0
python -m pip install -q trl peft datasets accelerate pandas pyarrow rapidfuzz pronto pyyaml tqdm matplotlib tensorboard
python - <<PY
import torch, vllm, trl, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__, "trl", trl.__version__)
PY
export CUDA_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python $THESIS/code/src/vllm_smoke.py
echo BUILD_OK
