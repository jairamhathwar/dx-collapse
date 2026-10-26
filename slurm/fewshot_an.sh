#!/bin/bash
# Analyses of the few-shot pairs (CPU). Reads and writes under ~/thesis_overflow (scratch over quota).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$H/runs
cd $THESIS/code/src
PAIRS="fewshot_t07:qwen3-8b-base>qwen3-8b fewshot_t07:qwen25-7b-base>qwen25-7b-it fewshot_t07:qwen25-32b-base>qwen25-32b-it fewshot_t07:olmo3-base>olmo3-instruct fewshot_t07:olmo31-32b-base>olmo31-32b-instruct"
python review2.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --pairs "$PAIRS" --out $H/analysis/review2_fewshot.json > $H/analysis/review2_fewshot.log 2>&1 &
python consequences.py --runs-dir $R/fewshot_t07 --tags qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,qwen25-32b-base,qwen25-32b-it,olmo3-base,olmo3-instruct,olmo31-32b-base,olmo31-32b-instruct --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $H/analysis/cons_fewshot.json > $H/analysis/cons_fewshot.log 2>&1 &
python robustness.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --spec "fewshot_t07:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,qwen25-32b-base,qwen25-32b-it,olmo3-base,olmo3-instruct,olmo31-32b-base,olmo31-32b-instruct" --out $H/analysis/robust_fewshot.json > $H/analysis/robust_fewshot.log 2>&1 &
wait
cat $H/analysis/review2_fewshot.log $H/analysis/cons_fewshot.log $H/analysis/robust_fewshot.log
echo FEWSHOT_AN_DONE
