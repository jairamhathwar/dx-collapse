#!/bin/bash
# Crossover analysis and audit simulation over every scored DDXPlus model (CPU only).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus
cd $THESIS/code/src
SPEC="pair_q3:qwen3-8b-base,qwen3-8b ddx_qwen3_t1:qwen3-8b-base,qwen3-8b pair25_t07:qwen25-7b-base,qwen25-7b-it pair25_t1:qwen25-7b-base,qwen25-7b-it \
pv_plain_t07:qwen3-8b-base,qwen3-8b pv_plain_t1:qwen3-8b-base,qwen3-8b pv_cot_t07:qwen3-8b-base,qwen3-8b pv_cot_t1:qwen3-8b-base,qwen3-8b \
ddx:alphamed-7b,huatuo-o1-7b,m1-7b ddx_base_rl:alphamed-7b-base-rl ddx_llama:huatuo-o1-8b,alphamed-8b,ultramedical-8b \
ddx_olmo:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general ddx_olmo_t1:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general \
e9_ddx:b8-acc-s0,b8-acc-s1,b8-gm-s0,b8-gm-s1 e7v2_ddx:qwen3-1.7b,acc-s0,acc-s1,gm3t12-s0,gm3t12-s1,puri-rlvr-s0,puri-rlcr-s0"
python audit.py --runs $THESIS/runs --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json \
  --spec "$SPEC" --out $THESIS/runs/analysis/audit.json
echo AUDIT_DONE
