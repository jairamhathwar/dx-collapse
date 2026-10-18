#!/bin/bash
#SBATCH --job-name=thesis-env
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/gscratch/scrubbed/jairamh/thesis/logs/%x-%j.out
set -euo pipefail
source ~/.thesis_env
ENV=$THESIS/envs/thesis
if [ ! -d "$ENV" ]; then conda create -y -p $ENV python=3.11 >/dev/null; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV
python -m pip install -U pip
python -m pip install vllm
python -m pip install trl peft datasets accelerate pandas pyarrow rapidfuzz pronto pyyaml tqdm
python - <<PY
import torch, vllm, trl, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "trl", trl.__version__, "datasets", datasets.__version__)
PY
echo BUILD_OK
