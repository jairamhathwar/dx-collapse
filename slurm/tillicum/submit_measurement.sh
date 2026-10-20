#!/bin/bash
# Tillicum: finish the measurement runs that were stuck on Klone (7-8B models -> full H200).
# Same prompts, sampling and token budgets as the Klone runs. Outputs under runs/meas_*/<tag>/.
#   bash slurm/tillicum/submit_measurement.sh
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --shard -1 --num-shards 2"
STD="--max-tokens 4096 --max-model-len 6144"
LONG="--max-tokens 8192 --max-model-len 10240"   # m1 reasons at length
T1="--temperature 1.0 --top-p 1.0"

run() {  # dir tag model args...
  local dir=$1 tag=$2 model=$3; shift 3
  j=$(sbatch --parsable -p gpu-h200 --time=04:00:00 --array=0-1 --job-name=m-$dir-$tag $S \
      --model $model --out runs/$dir/$tag/samples.s{shard}.jsonl "$@")
  echo "$dir/$tag  $model  job=$j"
}
# DDXPlus, T=0.7
run meas_ddx huatuo-o1-7b FreedomIntelligence/HuatuoGPT-o1-7B $DDX $STD
run meas_ddx m1-7b UCSC-VLAA/m1-7B-23K $DDX $LONG
run meas_ddx qwen25-7b-base Qwen/Qwen2.5-7B $DDX $STD
# DDXPlus, T=1.0
run meas_ddx_t1 qwen25-7b-base Qwen/Qwen2.5-7B $DDX $STD $T1
run meas_ddx_t1 alphamed-7b che111/AlphaMed-7B-instruct-rl $DDX $STD $T1
# Phenopackets go set, T=0.7
run meas_go m1-7b UCSC-VLAA/m1-7B-23K $PHENO $LONG
run meas_go qwen25-7b-base Qwen/Qwen2.5-7B $PHENO $STD
run meas_go qwen3-8b-base Qwen/Qwen3-8B-Base $PHENO $STD
run meas_go qwen3-8b Qwen/Qwen3-8B $PHENO $STD
