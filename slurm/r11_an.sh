#!/bin/bash
# Seventh review round (CPU): stricter packing test, patient-shuffled pass@32, severe thresholds 0.01/0.05,
# clustered stated-vs-agreement AUROC intervals for the zero-shot post-trained models. Code from the home clone.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; R=$THESIS/runs; A=$HOME/thesis_overflow/analysis/r11
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
python r11_an.py --runs $R $C --out $A/r11.json > $A/r11.log 2>&1 &
python review2.py --runs $R $C --pairs "$ZS" --tau 0.01 --out $A/review2_zs_tau001.json > $A/review2_zs_tau001.log 2>&1 &
python review2.py --runs $R $C --pairs "$ZS" --tau 0.05 --out $A/review2_zs_tau005.json > $A/review2_zs_tau005.log 2>&1 &
python cluster_within.py --runs $R $C --items "pair_q3:qwen3-8b pair25_t07:qwen25-7b-it big32_ddx:qwen25-32b-it,olmo31-32b-instruct ddx_olmo:olmo3-instruct" --out $A/within_zs.json > $A/within_zs.log 2>&1 &
wait
tail -n 30 $A/*.log | cut -c1-1500
echo R11_DONE
