#!/bin/bash
# Score and analyze the Mistral-7B v0.3 pair and Qwen3-8B thinking-mode runs (CPU). Reads and writes under ~/thesis_overflow.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$H/runs; A=$H/analysis/third
mkdir -p $A
cd $THESIS/code/src
# thinking-mode comparisons need the non-thinking pair in the same runs dir
ln -sfn $THESIS/runs/pair_q3/qwen3-8b-base $R/think_t07/qwen3-8b-base
ln -sfn $THESIS/runs/pair_q3/qwen3-8b $R/think_t07/qwen3-8b
echo "########## score"
for d in $R/mistral_t07/mistral7b-base $R/mistral_t07/mistral7b-it $R/mistral_t1/mistral7b-base $R/mistral_t1/mistral7b-it \
         $R/fewshot_t07/mistral7b-base $R/fewshot_t07/mistral7b-it $R/think_t07/qwen3-8b-think; do
  python score_ddxplus.py --samples $d/samples.jsonl --cases $D/cases_test.jsonl --conditions $D/conditions.json \
    --out $d/scores.json --graded-out $d/graded.jsonl > /dev/null &
done
wait
for t in mistral_t07 mistral_t1; do
  python compare_runs.py --runs-dir $R/$t --base mistral7b-base --out-dir $R/$t/report \
    --title "DDXPlus differential diagnosis" --related-label "Gold or in ground-truth differential" > $A/compare_$t.log 2>&1
done
PAIRS="mistral_t07:mistral7b-base>mistral7b-it mistral_t1:mistral7b-base>mistral7b-it fewshot_t07:mistral7b-base>mistral7b-it \
think_t07:qwen3-8b-base>qwen3-8b-think think_t07:qwen3-8b>qwen3-8b-think"
echo "########## analyses"
python review2.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --pairs "$PAIRS" --out $A/review2.json > $A/review2.log 2>&1 &
python aggregate_lists.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --pairs "$PAIRS" --out $A/aggregate.json > $A/aggregate.log 2>&1 &
python robustness.py --runs $R --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json \
  --spec "mistral_t07:mistral7b-base,mistral7b-it mistral_t1:mistral7b-base,mistral7b-it fewshot_t07:mistral7b-base,mistral7b-it think_t07:qwen3-8b-think" \
  --out $A/robust.json > $A/robust.log 2>&1 &
for spec in "mistral_t07 mistral7b-base,mistral7b-it" "mistral_t1 mistral7b-base,mistral7b-it" "fewshot_t07 mistral7b-base,mistral7b-it" "think_t07 qwen3-8b-base,qwen3-8b,qwen3-8b-think"; do
  set -- $spec
  python consequences.py --runs-dir $R/$1 --tags $2 --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $A/cons_$1.json > $A/cons_$1.log 2>&1 &
done
wait
for f in compare_mistral_t07 compare_mistral_t1; do echo "== $f"; awk '/pass@k \(exact\)/{p=1} p' $A/$f.log | head -20; done
cat $A/review2.log $A/robust.log $A/cons_*.log $A/aggregate.log
echo NEW_AN_DONE
