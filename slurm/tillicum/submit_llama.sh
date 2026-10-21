#!/bin/bash
# Tillicum: second model family (Llama-3.1-8B) plus AlphaMed RL-from-base (Qwen2.5-7B).
# Same prompts, sampling and token budgets as submit_measurement.sh. Outputs under runs/meas_*/<tag>/.
#   bash slurm/tillicum/submit_llama.sh descendants   # ungated medical descendants (run now)
#   bash slurm/tillicum/submit_llama.sh pair          # Llama-3.1-8B base + Instruct (needs Meta gate approval)
#   bash slurm/tillicum/submit_llama.sh prompts       # prompt-robustness variants on the Qwen3-8B pair
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --shard -1 --num-shards 2"
STD="--max-tokens 4096 --max-model-len 6144"
T1="--temperature 1.0 --top-p 1.0"

run() {  # dir tag model args...
  local dir=$1 tag=$2 model=$3; shift 3
  j=$(sbatch --parsable -p gpu-h200 --time=04:00:00 --array=0-1 --job-name=m-$dir-$tag $S \
      --model $model --out runs/$dir/$tag/samples.s{shard}.jsonl "$@")
  echo "$dir/$tag  $model  job=$j"
}
case "${1:-}" in
  descendants)
    for spec in alphamed-7b-base-rl:che111/AlphaMed-7B-base-rl \
                alphamed-8b:che111/AlphaMed-8B-instruct-rl \
                huatuo-o1-8b:FreedomIntelligence/HuatuoGPT-o1-8B \
                ultramedical-8b:TsinghuaC3I/Llama-3.1-8B-UltraMedical; do
      tag=${spec%%:*}; model=${spec#*:}
      run meas_ddx $tag $model $DDX $STD
      run meas_go $tag $model $PHENO $STD
    done ;;
  pair)
    for spec in llama31-8b-base:meta-llama/Llama-3.1-8B llama31-8b-it:meta-llama/Llama-3.1-8B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      run meas_ddx $tag $model $DDX $STD
      run meas_ddx_t1 $tag $model $DDX $STD $T1
      run meas_go $tag $model $PHENO $STD
    done ;;
  prompts)
    for v in plain cot; do
      for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B; do
        tag=${spec%%:*}; model=${spec#*:}
        run pv_${v}_t07 $tag $model $DDX $STD --prompt-variant $v
        run pv_${v}_t1 $tag $model $DDX $STD $T1 --prompt-variant $v
      done
    done ;;
  *) echo "usage: $0 descendants|pair|prompts"; exit 1 ;;
esac
