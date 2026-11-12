#!/bin/bash
# Score the ICML pilot on a MIG slice (CPU work): size ladder and self-targeted GRPO, with minp_an.py.
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus; A=$THESIS/analysis/pilot
mkdir -p $A
cd $THESIS/code/src
python minp_an.py $R $D $A/ladder.json \
  q0.5/base:ladder/q25-05b-base-t07 q0.5/it:ladder/q25-05b-it-t07 q0.5/it-minp:ladder/q25-05b-it-minp15 \
  q1.5/base:ladder/q25-15b-base-t07 q1.5/it:ladder/q25-15b-it-t07 q1.5/it-minp:ladder/q25-15b-it-minp15 \
  q3/base:ladder/q25-3b-base-t07 q3/it:ladder/q25-3b-it-t07 q3/it-minp:ladder/q25-3b-it-minp15 \
  q7/base:meas_ddx/qwen25-7b-base q7/it-minp:minp10_t15/qwen25-7b-it \
  q14/base:ladder/q25-14b-base-t07 q14/it:ladder/q25-14b-it-t07 q14/it-minp:ladder/q25-14b-it-minp15 \
  q32/base:big32_ddx/qwen25-32b-base q32/it:big32_ddx/qwen25-32b-it q32/it-minp:minp10_t15/qwen25-32b-it \
  q72/base:big72_ddx/qwen25-72b-base q72/it:big72_ddx/qwen25-72b-it q72/it-minp:minp10_t15/qwen25-72b-it
python minp_an.py $R $D $A/selftarget.json \
  untrained:eval_ddx/qwen3-1.7b-untrained acc:eval_ddx/acc-s0 gm:eval_ddx/gm-s0 gm3t12:eval_ddx/gm3t12-s0 \
  acc+self:selftarget_ddx/st-accsw-s0 self-only:selftarget_ddx/st-sw-s0 acc+ref:selftarget_ddx/st-accrw-s0
