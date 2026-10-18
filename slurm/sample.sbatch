#!/bin/bash
# Usage (all args after the script go to sample_diagnoses.py):
#   sbatch --job-name=smp-qwen3-8b slurm/sample.sbatch --model Qwen/Qwen3-8B \
#       --cases $THESIS/data/phenopackets/cases.jsonl --out $THESIS/runs/qwen3-8b/samples.jsonl --k 32
# Default: preemptible checkpoint pool (requeued on preemption; resumes from --out).
# Klone allows one partition per job; pass -p gpu-l40s or -p gpu-l40 to use STF priority GPUs.
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=1
#SBATCH --constraint=l40s|l40|a40|a100|h200
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/scrubbed/jairamh/thesis/logs/%x-%j.out
echo "== $(date) host=$(hostname) restart=${SLURM_RESTART_COUNT:-0}"
# env setup without -e/-u: module/conda scripts are not strict-mode safe; check results explicitly
source ~/.thesis_env || echo "warn: thesis_env returned $?"
module load conda/Miniforge3-25.9.1-0 || { echo "module load conda failed"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis" || { echo "conda activate failed"; exit 1; }
# torch.compile/inductor shells out to nvcc; use the CUDA toolkit bundled in the env
export CUDA_HOME=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
echo "python=$(which python) nvcc=$(which nvcc)"
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python $THESIS/code/src/sample_diagnoses.py "$@"
echo "== done $(date)"
