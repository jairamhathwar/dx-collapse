#!/bin/bash
# Matched-prompt greedy (CPU): post-trained models with the base models' few-shot examples vs the base few-shot greedy runs,
# base models with reference-free examples vs the post-trained zero-shot greedy runs.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; O=$H/runs; A=$H/analysis/r9
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
GP=""
for p in qwen3-8b-base:qwen3-8b qwen25-7b-base:qwen25-7b-it qwen25-32b-base:qwen25-32b-it olmo3-base:olmo3-instruct olmo31-32b-base:olmo31-32b-instruct mistral7b-base:mistral7b-it; do
  b=${p%%:*}; q=${p#*:}
  GP="$GP greedy/$b>greedy_fs/$q greedy_rf/$b>greedy/$q"
done
python r7_analysis.py --mode greedy --runs $O $C --pairs "$GP" --out $A/greedy_matched.json > $A/greedy_matched.log 2>&1
tail -n 40 $A/greedy_matched.log | cut -c1-1500
echo R9_DONE
