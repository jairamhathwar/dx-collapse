#!/bin/bash
# OLMo full-distribution runs + audit over all models. Klone scrubbed is over quota (2026-10-24), so this
# reads runs from scrubbed and writes only under ~/thesis_overflow.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow
cd $THESIS/code/src
for t in olmo3-base olmo3-instruct; do
  d=$H/runs/pv_full_t07/$t; cat $d/samples.s*.jsonl > $d/samples.jsonl
  python score_ddxplus.py --top-by-prob --samples $d/samples.jsonl --cases $D/cases_test.jsonl --conditions $D/conditions.json \
    --out $d/scores.json --graded-out $d/graded.jsonl | tail -1 &
done
wait
python stated_vs_sampled.py --top-by-prob --runs-dir $H/runs/pv_full_t07 --tags olmo3-base,olmo3-instruct --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $H/analysis/svs_full_olmo.json
python consequences.py --top-by-prob --runs-dir $H/runs/pv_full_t07 --tags olmo3-base,olmo3-instruct --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $H/analysis/cons_full_olmo.json
SPEC="pair_q3:qwen3-8b-base,qwen3-8b ddx_qwen3_t1:qwen3-8b-base,qwen3-8b pair25_t07:qwen25-7b-base,qwen25-7b-it pair25_t1:qwen25-7b-base,qwen25-7b-it \
pv_plain_t07:qwen3-8b-base,qwen3-8b pv_plain_t1:qwen3-8b-base,qwen3-8b pv_cot_t07:qwen3-8b-base,qwen3-8b pv_cot_t1:qwen3-8b-base,qwen3-8b \
ddx:alphamed-7b,huatuo-o1-7b,m1-7b ddx_base_rl:alphamed-7b-base-rl ddx_llama:huatuo-o1-8b,alphamed-8b,ultramedical-8b \
ddx_olmo:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general ddx_olmo_t1:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general \
e9_ddx:b8-acc-s0,b8-acc-s1,b8-gm-s0,b8-gm-s1 e7v2_ddx:qwen3-1.7b,acc-s0,acc-s1,gm3t12-s0,gm3t12-s1,puri-rlvr-s0,puri-rlcr-s0 \
big32_ddx:qwen25-32b-base,qwen25-32b-it,olmo31-32b-base,olmo31-32b-sft,olmo31-32b-dpo,olmo31-32b-instruct \
pv_full_t07@prob:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it $H/runs/pv_full_t07@prob:olmo3-base,olmo3-instruct"
python audit.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --spec "$SPEC" --out $H/analysis/audit_all.json
echo OLMO_FULL_AUDIT_DONE
