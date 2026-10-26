#!/bin/bash
# Stated-list aggregation across calls (CPU) on the zero-shot and few-shot pairs. Writes to ~/thesis_overflow.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; F=$H/runs/fewshot_t07
cd $THESIS/code/src
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
FS="$F:qwen3-8b-base>qwen3-8b $F:qwen25-7b-base>qwen25-7b-it $F:qwen25-32b-base>qwen25-32b-it $F:olmo3-base>olmo3-instruct $F:olmo31-32b-base>olmo31-32b-instruct"
python aggregate_lists.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --pairs "$ZS" --out $H/analysis/aggregate_zs.json > $H/analysis/aggregate_zs.log 2>&1 &
python aggregate_lists.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --pairs "$FS" --out $H/analysis/aggregate_fs.json > $H/analysis/aggregate_fs.log 2>&1 &
wait
cat $H/analysis/aggregate_zs.log $H/analysis/aggregate_fs.log
echo AGG_DONE
