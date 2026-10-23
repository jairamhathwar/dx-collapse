#!/bin/bash
# Aggregation and uncertainty consequences of the stated/sampled gap, every DDXPlus comparison (CPU only).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus
cd $THESIS/code/src
python -c "
import numpy as np
from consequences import auroc, spearman, rank, case_row
assert abs(auroc(np.array([0.1,0.4,0.35,0.8]), np.array([0,0,1,1])) - 0.75) < 1e-9
assert abs(spearman(np.array([1,2,3,4.]), np.array([1,3,2,4.])) - 0.8) < 1e-9
assert list(rank(np.array([3,1,1,2.]))) == [3, 0.5, 0.5, 2]
r = case_row(['a','b','a'], [{'a':.6,'b':.4},{'b':1.0},{'a':.5,'c':.5}], 'a', 1.0)
assert r['vote@32'] == 1 and r['consist@32'] == 0.5 and r['stated_p'] == 0.6, r
print('unit checks ok')
" || exit 1
run() { python consequences.py --runs-dir $R/$1 --tags $2 --cases $D/cases_test.jsonl \
          --conditions-json $D/release_conditions.json --out $R/analysis/cons/$1.json > $THESIS/logs/cons-$1.log 2>&1 & }
run pair_q3 qwen3-8b-base,qwen3-8b
run ddx_qwen3_t1 qwen3-8b-base,qwen3-8b
run pair25_t07 qwen25-7b-base,qwen25-7b-it
run pair25_t1 qwen25-7b-base,qwen25-7b-it
run pv_cot_t07 qwen3-8b-base,qwen3-8b
run ddx qwen25-7b-it,alphamed-7b,huatuo-o1-7b,m1-7b
run ddx_olmo olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general
run ddx_olmo_t1 olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general
wait
for f in $THESIS/logs/cons-*.log; do cat $f; echo; done
echo CONS_DONE
