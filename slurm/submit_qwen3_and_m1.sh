#!/bin/bash
# Qwen3 base-vs-post-trained pair on both datasets, plus the m1 Phenopacket rerun with an
# 8k token budget. Each model is a 4-shard job array (at most 2 shards at once, to share
# the STF GPUs); scoring jobs run automatically once their arrays finish.
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/sample.sbatch
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt"
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl"
SHARDS="--shard -1 --num-shards 4 --k 32"

submit() {  # name runs_dir tag model extra-args...
  local name=$1 runs=$2 tag=$3 model=$4; shift 4
  sbatch --parsable --array=0-3%2 --job-name="$name" --time=06:00:00 $S \
    --model "$model" --out "$runs/$tag/samples.s{shard}.jsonl" $SHARDS "$@"
}

std="--max-tokens 4096 --max-model-len 6144"
think="--thinking --max-tokens 8192 --max-model-len 10240"

d1=$(submit q3b-ddx  runs/ddx_qwen3 qwen3-8b-base  Qwen/Qwen3-8B-Base $DDX $std)
d2=$(submit q3-ddx   runs/ddx_qwen3 qwen3-8b       Qwen/Qwen3-8B      $DDX $std)
d3=$(submit q3t-ddx  runs/ddx_qwen3 qwen3-8b-think Qwen/Qwen3-8B      $DDX $think)
p1=$(submit q3b-go   runs/go_qwen3  qwen3-8b-base  Qwen/Qwen3-8B-Base $PHENO $std)
p2=$(submit q3-go    runs/go_qwen3  qwen3-8b       Qwen/Qwen3-8B      $PHENO $std)
p3=$(submit q3t-go   runs/go_qwen3  qwen3-8b-think Qwen/Qwen3-8B      $PHENO $think)
m1=$(submit m1-go8k  runs/go        m1-7b          UCSC-VLAA/m1-7B-23K $PHENO --max-tokens 8192 --max-model-len 10240)

sbatch --parsable --dependency=afterany:$d1:$d2:$d3 \
  --export=ALL,RUNS=$THESIS/runs/ddx_qwen3,BASE=qwen3-8b-base code/slurm/score_ddx.sbatch
for spec in "$p1:$p2:$p3 $THESIS/runs/go_qwen3 qwen3-8b-base score-go-q3" "$m1 $THESIS/runs/go qwen25-7b-it score-go"; do
  set -- $spec
  sbatch --parsable --dependency=afterany:$1 -A stf -p ckpt-all -c 4 --mem=24G --time=01:30:00 \
    -J "$4" -o "logs/$4-%j.out" --export=ALL,RUNS=$2,BASE=$3 --wrap "bash $THESIS/code/slurm/score_go.sh"
done
echo "arrays: ddx=$d1,$d2,$d3 pheno=$p1,$p2,$p3 m1=$m1"
