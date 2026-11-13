#!/bin/bash
# Score the self-target seeds (MIG slice, CPU work) with minp_an.py.
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
cd $THESIS/code/src; mkdir -p $THESIS/analysis/pilot
python minp_an.py $THESIS/runs $THESIS/data/ddxplus $THESIS/analysis/pilot/seeds.json \
  untrained:eval_ddx/qwen3-1.7b-untrained acc/s0:eval_ddx/acc-s0 acc/s1:eval_ddx/acc-s1 gm3t12/s0:eval_ddx/gm3t12-s0 gm3t12/s1:eval_ddx/gm3t12-s1 \
  acc+self/s0:selftarget_ddx/st-accsw-s0 acc+self/s1:selftarget_ddx/st-accsw-s1 acc+self/s2:selftarget_ddx/st-accsw-s2 \
  self/s0:selftarget_ddx/st-sw-s0 self/s1:selftarget_ddx/st-sw-s1 self/s2:selftarget_ddx/st-sw-s2 acc+ref/s0:selftarget_ddx/st-accrw-s0
