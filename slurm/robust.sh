#!/bin/bash
# Review-driven robustness analyses and audit rerun (CPU). Writes to ~/thesis_overflow (scratch over quota).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow
cd $THESIS/code/src
RSPEC="pair_q3:qwen3-8b-base,qwen3-8b ddx_qwen3_t1:qwen3-8b-base,qwen3-8b pair25_t07:qwen25-7b-base,qwen25-7b-it pair25_t1:qwen25-7b-base,qwen25-7b-it \
big32_ddx:qwen25-32b-base,qwen25-32b-it,olmo31-32b-base,olmo31-32b-sft,olmo31-32b-dpo,olmo31-32b-instruct \
ddx_olmo:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general ddx:alphamed-7b,huatuo-o1-7b,m1-7b e9_ddx:b8-acc-s0,b8-gm-s0"
python robustness.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --spec "$RSPEC" --out $H/analysis/robustness.json > $H/analysis/robustness.log 2>&1 &
ASPEC="pair_q3:qwen3-8b-base,qwen3-8b ddx_qwen3_t1:qwen3-8b-base,qwen3-8b pair25_t07:qwen25-7b-base,qwen25-7b-it pair25_t1:qwen25-7b-base,qwen25-7b-it \
pv_plain_t07:qwen3-8b-base,qwen3-8b pv_plain_t1:qwen3-8b-base,qwen3-8b pv_cot_t07:qwen3-8b-base,qwen3-8b pv_cot_t1:qwen3-8b-base,qwen3-8b \
ddx:alphamed-7b,huatuo-o1-7b,m1-7b ddx_base_rl:alphamed-7b-base-rl ddx_llama:huatuo-o1-8b,alphamed-8b,ultramedical-8b \
ddx_olmo:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general ddx_olmo_t1:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general \
e9_ddx:b8-acc-s0,b8-acc-s1,b8-gm-s0,b8-gm-s1 e7v2_ddx:qwen3-1.7b,acc-s0,acc-s1,gm3t12-s0,gm3t12-s1,puri-rlvr-s0,puri-rlcr-s0 \
big32_ddx:qwen25-32b-base,qwen25-32b-it,olmo31-32b-base,olmo31-32b-sft,olmo31-32b-dpo,olmo31-32b-instruct \
pv_full_t07@prob:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it $H/runs/pv_full_t07@prob:olmo3-base,olmo3-instruct"
python audit.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --spec "$ASPEC" --out $H/analysis/audit_all_v2.json > $H/analysis/audit_v2.log 2>&1 &
wait
cat $H/analysis/robustness.log; tail -60 $H/analysis/audit_v2.log
echo ROBUST_DONE
