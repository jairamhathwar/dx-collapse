#!/bin/bash
# Stated-list vs sampled-top-1 differentials for every DDXPlus comparison (CPU only).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus
cd $THESIS/code/src
run() { python stated_vs_sampled.py --runs-dir $R/$1 --tags $2 --cases $D/cases_test.jsonl \
          --conditions-json $D/release_conditions.json --out $R/analysis/svs/$1.json > $THESIS/logs/svs-$1.log 2>&1 & }
run pair_q3 qwen3-8b-base,qwen3-8b
run ddx_qwen3_t1 qwen3-8b-base,qwen3-8b
run pair25_t07 qwen25-7b-base,qwen25-7b-it
run pair25_t1 qwen25-7b-base,qwen25-7b-it
run pv_cot_t07 qwen3-8b-base,qwen3-8b
run ddx_base_rl qwen25-7b-base,qwen25-7b-it,alphamed-7b,alphamed-7b-base-rl
run ddx qwen25-7b-it,alphamed-7b,huatuo-o1-7b,m1-7b
run ddx_olmo olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general
run ddx_olmo_t1 olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general
wait
for f in $THESIS/logs/svs-*.log; do cat $f; echo; done
echo SVS_DONE
