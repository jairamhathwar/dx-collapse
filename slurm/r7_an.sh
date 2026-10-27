#!/bin/bash
# Greedy single-call, few-shot temperature frontier and must-not-miss prompt (CPU). Code from the home clone.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$THESIS/runs; O=$H/runs; F=$O/fewshot_t07; A=$H/analysis/r7
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
GP="greedy:qwen3-8b-base>qwen3-8b greedy:qwen25-7b-base>qwen25-7b-it greedy:qwen25-32b-base>qwen25-32b-it greedy:olmo3-base>olmo3-instruct greedy:olmo31-32b-base>olmo31-32b-instruct greedy:mistral7b-base>mistral7b-it"
FP=""; for t in 04 07 10 13; do for p in qwen3-8b-base:qwen3-8b qwen25-7b-base:qwen25-7b-it olmo3-base:olmo3-instruct; do FP="$FP frontier_t$t:${p%%:*}>${p#*:}"; done; done
MP="$O/mnm_t07/qwen3-8b>$R/pair_q3/qwen3-8b $O/mnm_t07/qwen25-7b-it>$R/pair25_t07/qwen25-7b-it $O/mnm_t07/qwen25-32b-it>$R/big32_ddx/qwen25-32b-it \
$O/mnm_t07/olmo3-instruct>$R/ddx_olmo/olmo3-instruct $O/mnm_t07/olmo31-32b-instruct>$R/big32_ddx/olmo31-32b-instruct $O/mnm_t07/mistral7b-it>$O/mistral_t07/mistral7b-it \
$O/mnm_t07/qwen3-8b>$F/qwen3-8b-base $O/mnm_t07/qwen25-7b-it>$F/qwen25-7b-base $O/mnm_t07/qwen25-32b-it>$F/qwen25-32b-base \
$O/mnm_t07/olmo3-instruct>$F/olmo3-base $O/mnm_t07/olmo31-32b-instruct>$F/olmo31-32b-base $O/mnm_t07/mistral7b-it>$F/mistral7b-base"
python r7_analysis.py --mode greedy --runs $O $C --pairs "$GP" --out $A/greedy.json > $A/greedy.log 2>&1 &
python r7_analysis.py --mode mnm --runs / $C --pairs "$MP" --out $A/mnm.json > $A/mnm.log 2>&1 &
python review2.py --runs $O $C --pairs "$FP" --out $A/frontier_review2.json > $A/frontier_review2.log 2>&1 &
for t in 04 07 10 13; do
  python consequences.py --runs-dir $O/frontier_t$t --tags qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct $C --out $A/frontier_cons_t$t.json > $A/frontier_cons_t$t.log 2>&1 &
done
python robustness.py --runs $O $C --spec "greedy:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,qwen25-32b-base,qwen25-32b-it,olmo3-base,olmo3-instruct,olmo31-32b-base,olmo31-32b-instruct,mistral7b-base,mistral7b-it frontier_t04:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct frontier_t07:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct frontier_t10:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct frontier_t13:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct" --out $A/robust.json > $A/robust.log 2>&1 &
wait
tail -n 30 $A/*.log | cut -c1-1500
echo R7_DONE
