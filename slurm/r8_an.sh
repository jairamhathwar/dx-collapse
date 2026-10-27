#!/bin/bash
# Sixth review round (CPU): canonical bootstrap for the headline pairs (50k resamples), controls, audit at 5 and 32 samples.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$THESIS/runs; A=$H/analysis/r8
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
ASPEC="pair_q3:qwen3-8b-base,qwen3-8b ddx_qwen3_t1:qwen3-8b-base,qwen3-8b pair25_t07:qwen25-7b-base,qwen25-7b-it pair25_t1:qwen25-7b-base,qwen25-7b-it \
pv_plain_t07:qwen3-8b-base,qwen3-8b pv_plain_t1:qwen3-8b-base,qwen3-8b pv_cot_t07:qwen3-8b-base,qwen3-8b pv_cot_t1:qwen3-8b-base,qwen3-8b \
ddx:alphamed-7b,huatuo-o1-7b,m1-7b ddx_base_rl:alphamed-7b-base-rl ddx_llama:huatuo-o1-8b,alphamed-8b,ultramedical-8b \
ddx_olmo:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general ddx_olmo_t1:olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general \
e9_ddx:b8-acc-s0,b8-acc-s1,b8-gm-s0,b8-gm-s1 e7v2_ddx:qwen3-1.7b,acc-s0,acc-s1,gm3t12-s0,gm3t12-s1,puri-rlvr-s0,puri-rlcr-s0 \
big32_ddx:qwen25-32b-base,qwen25-32b-it,olmo31-32b-base,olmo31-32b-sft,olmo31-32b-dpo,olmo31-32b-instruct \
pv_full_t07@prob:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it $H/runs/pv_full_t07@prob:olmo3-base,olmo3-instruct"
python review2.py --runs $R $C --pairs "$ZS" --n-boot 50000 --out $A/review2_zs_50k.json > $A/review2_zs_50k.log 2>&1 &
python r8_controls.py --spec $HOME/thesis_repo/slurm/r8_spec.json $C --out $A/controls.json > $A/controls.log 2>&1 &
python audit.py --runs $R $C --spec "$ASPEC" --agree 32 --out $A/audit_agree32.json > $A/audit_agree32.log 2>&1 &
python audit.py --runs $R $C --spec "$ASPEC" --agree 5 --out $A/audit_agree5.json > $A/audit_agree5.log 2>&1 &
wait
tail -n 60 $A/*.log | cut -c1-1500
echo R8_DONE
