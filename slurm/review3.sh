#!/bin/bash
# Third review round (CPU): add-one bootstrap p-values with 10k cluster resamples, TV decomposition,
# per-condition severe test, stage/medical pairs, and clustered within-model intervals. Writes to ~/thesis_overflow.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; A=$H/analysis/r3; F=$H/runs/fewshot_t07
mkdir -p $A
cd $THESIS/code/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
FS="$F:qwen3-8b-base>qwen3-8b $F:qwen25-7b-base>qwen25-7b-it $F:qwen25-32b-base>qwen25-32b-it $F:olmo3-base>olmo3-instruct $F:olmo31-32b-base>olmo31-32b-instruct"
TH="$F:mistral7b-base>mistral7b-it $H/runs/think_t07:qwen3-8b-base>qwen3-8b-think $H/runs/think_t07:qwen3-8b>qwen3-8b-think"
ST="ddx_olmo:olmo3-base>olmo3-sft ddx_olmo:olmo3-sft>olmo3-dpo ddx_olmo:olmo3-dpo>olmo3-instruct ddx_olmo:olmo3-base>olmo3-rlzero-general \
ddx_olmo_t1:olmo3-base>olmo3-rlzero-general big32_ddx:olmo31-32b-base>olmo31-32b-sft big32_ddx:olmo31-32b-sft>olmo31-32b-dpo \
big32_ddx:olmo31-32b-dpo>olmo31-32b-instruct ddx:qwen25-7b-it>alphamed-7b ddx:qwen25-7b-it>huatuo-o1-7b ddx:qwen25-7b-it>m1-7b"
python review2.py --runs $R $C --pairs "$ZS" --out $A/review2_zs.json > $A/review2_zs.log 2>&1 &
python review2.py --runs $R $C --pairs "$FS" --out $A/review2_fs.json > $A/review2_fs.log 2>&1 &
python review2.py --runs $R $C --pairs "$TH" --out $A/review2_third.json > $A/review2_third.log 2>&1 &
python review2.py --runs $R $C --pairs "$ST" --out $A/review2_stages.json > $A/review2_stages.log 2>&1 &
python cluster_within.py --runs $R $C --items "pair_q3:qwen3-8b-base,qwen3-8b pair25_t07:qwen25-7b-base,qwen25-7b-it big32_ddx:qwen25-32b-base,qwen25-32b-it,olmo31-32b-base,olmo31-32b-instruct ddx_olmo:olmo3-base,olmo3-instruct ddx:alphamed-7b,huatuo-o1-7b,m1-7b pv_cot_t07:qwen3-8b" --out $A/within.json > $A/within.log 2>&1 &
wait
cat $A/*.log
echo R3_DONE
