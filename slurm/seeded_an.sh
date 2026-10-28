#!/bin/bash
# Recovery experiment scoring (CPU): seeded picks vs ordinary zero-shot samples and the base models' samples.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; R=$THESIS/runs; S=$HOME/thesis_overflow/runs/seeded_t07; A=$HOME/thesis_overflow/analysis/seeded
mkdir -p $A
cd $HOME/thesis_repo/src
python seeded_an.py --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --n-boot 10000 --spec \
  qwen3-8b=$S/qwen3-8b/samples.jsonl:$R/pair_q3/qwen3-8b/samples.jsonl:$R/pair_q3/qwen3-8b-base/samples.jsonl \
  qwen25-7b-it=$S/qwen25-7b-it/samples.jsonl:$R/pair25_t07/qwen25-7b-it/samples.jsonl:$R/pair25_t07/qwen25-7b-base/samples.jsonl \
  qwen25-32b-it=$S/qwen25-32b-it/samples.jsonl:$R/big32_ddx/qwen25-32b-it/samples.jsonl:$R/big32_ddx/qwen25-32b-base/samples.jsonl \
  olmo3-instruct=$S/olmo3-instruct/samples.jsonl:$R/ddx_olmo/olmo3-instruct/samples.jsonl:$R/ddx_olmo/olmo3-base/samples.jsonl \
  olmo31-32b-instruct=$S/olmo31-32b-instruct/samples.jsonl:$R/big32_ddx/olmo31-32b-instruct/samples.jsonl:$R/big32_ddx/olmo31-32b-base/samples.jsonl \
  --out $A/seeded.json > $A/seeded.log 2>&1
tail -n 15 $A/seeded.log | cut -c1-1500
echo SEEDED_DONE
