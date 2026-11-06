#!/bin/bash
# Score the min-p decoding baseline on a MIG slice (CPU work): merge shards, then minp_an.py.
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
R=$THESIS/runs; D=$THESIS/data/ddxplus; A=$THESIS/analysis/minp
mkdir -p $A
for d in $R/minp10_t*/* $R/matchT_t*/*; do
  [ -f $d/samples.jsonl ] || cat $d/samples.s*.jsonl > $d/samples.jsonl
done
cd $THESIS/code/src
python minp_an.py $R $D $A/minp.json \
  q3/T1.3:matchT_t13/qwen3-8b q3/minp-T1.0:minp10_t10/qwen3-8b q3/minp-T1.5:minp10_t15/qwen3-8b \
  q7/base-T0.7:meas_ddx/qwen25-7b-base q7/T1.3:matchT_t13/qwen25-7b-it q7/minp-T1.0:minp10_t10/qwen25-7b-it q7/minp-T1.5:minp10_t15/qwen25-7b-it \
  o7/base-T0.7:meas_ddx/olmo3-base o7/T0.7:meas_ddx/olmo3-instruct o7/minp-T1.0:minp10_t10/olmo3-instruct o7/minp-T1.5:minp10_t15/olmo3-instruct \
  m7/base-fewshot:fewshot_t07/mistral7b-base m7/T0.7:meas_ddx/mistral7b-it m7/minp-T1.0:minp10_t10/mistral7b-it m7/minp-T1.5:minp10_t15/mistral7b-it \
  q32/base-T0.7:big32_ddx/qwen25-32b-base q32/T0.7:big32_ddx/qwen25-32b-it q32/minp-T1.0:minp10_t10/qwen25-32b-it q32/minp-T1.5:minp10_t15/qwen25-32b-it \
  o32/base-T0.7:big32_ddx/olmo31-32b-base o32/T0.7:big32_ddx/olmo31-32b-instruct o32/minp-T1.0:minp10_t10/olmo31-32b-instruct o32/minp-T1.5:minp10_t15/olmo31-32b-instruct
