#!/bin/bash
# Round-7 analyses on Tillicum (CPU work on a MIG slice): merge shards, then greedy, must-not-miss and frontier.
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus; A=$THESIS/analysis/r7
mkdir -p $A
for d in $R/greedy/* $R/frontier_t*/* $R/mnm_t07/* $R/fewshot_t07/* $R/big32_ddx/qwen25-32b-it $R/big32_ddx/olmo31-32b-instruct $R/meas_ddx/olmo3-instruct $R/meas_ddx/mistral7b-it; do
  [ -d "$d" ] && ls $d/samples.s*.jsonl >/dev/null 2>&1 && cat $d/samples.s*.jsonl > $d/samples.jsonl
done
cd $THESIS/code/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
GP="greedy:qwen3-8b-base>qwen3-8b greedy:qwen25-7b-base>qwen25-7b-it greedy:qwen25-32b-base>qwen25-32b-it greedy:olmo3-base>olmo3-instruct greedy:olmo31-32b-base>olmo31-32b-instruct greedy:mistral7b-base>mistral7b-it"
FP=""; for t in 04 07 10 13; do for p in qwen3-8b-base:qwen3-8b qwen25-7b-base:qwen25-7b-it olmo3-base:olmo3-instruct; do FP="$FP frontier_t$t:${p%%:*}>${p#*:}"; done; done
MP="mnm_t07/qwen3-8b>zs_ref/qwen3-8b mnm_t07/qwen25-7b-it>zs_ref/qwen25-7b-it mnm_t07/qwen25-32b-it>big32_ddx/qwen25-32b-it mnm_t07/olmo3-instruct>meas_ddx/olmo3-instruct \
mnm_t07/olmo31-32b-instruct>big32_ddx/olmo31-32b-instruct mnm_t07/mistral7b-it>meas_ddx/mistral7b-it \
mnm_t07/qwen3-8b>fewshot_t07/qwen3-8b-base mnm_t07/qwen25-7b-it>fewshot_t07/qwen25-7b-base mnm_t07/qwen25-32b-it>fewshot_t07/qwen25-32b-base \
mnm_t07/olmo3-instruct>fewshot_t07/olmo3-base mnm_t07/olmo31-32b-instruct>fewshot_t07/olmo31-32b-base mnm_t07/mistral7b-it>fewshot_t07/mistral7b-base"
python r7_analysis.py --mode greedy --runs $R $C --pairs "$GP" --out $A/greedy.json > $A/greedy.log 2>&1 &
python r7_analysis.py --mode mnm --runs $R $C --pairs "$MP" --out $A/mnm.json > $A/mnm.log 2>&1 &
python review2.py --runs $R $C --pairs "$FP" --out $A/frontier_review2.json > $A/frontier_review2.log 2>&1 &
wait
for t in 04 07 10 13; do
  python consequences.py --runs-dir $R/frontier_t$t --tags qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct $C --out $A/frontier_cons_t$t.json > $A/frontier_cons_t$t.log 2>&1 &
done
python robustness.py --runs $R $C --spec "greedy:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,qwen25-32b-base,qwen25-32b-it,olmo3-base,olmo3-instruct,olmo31-32b-base,olmo31-32b-instruct,mistral7b-base,mistral7b-it frontier_t04:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct frontier_t13:qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,olmo3-base,olmo3-instruct" --out $A/robust.json > $A/robust.log 2>&1 &
wait
tail -n 40 $A/*.log | cut -c1-1500
echo R7_DONE
