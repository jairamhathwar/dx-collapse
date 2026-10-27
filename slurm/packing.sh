#!/bin/bash
# Packing test (CPU): does one post-trained list reproduce its base model's averaged list? Code from the home clone.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; A=$H/analysis/r5; F=$H/runs/fewshot_t07
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
FS="$F:qwen3-8b-base>qwen3-8b $F:qwen25-7b-base>qwen25-7b-it $F:qwen25-32b-base>qwen25-32b-it $F:olmo3-base>olmo3-instruct $F:olmo31-32b-base>olmo31-32b-instruct \
$F:mistral7b-base>mistral7b-it $H/runs/think_t07:qwen3-8b-base>qwen3-8b-think"
python packing_test.py --runs $THESIS/runs $C --pairs "$ZS" --out $A/packing_zs.json > $A/packing_zs.log 2>&1 &
python packing_test.py --runs $THESIS/runs $C --pairs "$FS" --out $A/packing_fs.json > $A/packing_fs.log 2>&1 &
wait
cat $A/*.log
echo PACK_DONE
