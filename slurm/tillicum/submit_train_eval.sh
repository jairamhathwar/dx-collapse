#!/bin/bash
# Tillicum: GRPO train (full H200) -> sample merged model on DDXPlus test + Phenopackets go (MIG slices).
# Same stabilized defaults as Klone v2 (train_grpo.py); per-config reward weights and train temperature.
#   CONFIGS="tag|rewards|seed|weights|temperature ..." bash slurm/tillicum/submit_train_eval.sh
# Example: CONFIGS="gm3t12-s0|acc,group_match|0|1,1,3|1.2" bash slurm/tillicum/submit_train_eval.sh
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
TR=code/slurm/tillicum/train.sbatch
S=code/slurm/tillicum/sample.sbatch
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"

for spec in $CONFIGS; do
  IFS="|" read -r tag rewards seed weights temp <<< "$spec"
  out=$THESIS/runs/train/$tag
  extra=""
  [ -n "${weights:-}" ] && extra="$extra --weights $weights"
  [ -n "${temp:-}" ] && extra="$extra --temperature $temp"
  t=$(sbatch --parsable --job-name=t-$tag $TR --rewards $rewards --seed $seed --out $out $extra)
  d=$(sbatch --parsable --array=0-1 --dependency=afterok:$t --job-name=ev-ddx-$tag \
      $S --model $out/merged --out runs/eval_ddx/$tag/samples.s{shard}.jsonl $DDX)
  g=$(sbatch --parsable --array=0-1 --dependency=afterok:$t --job-name=ev-go-$tag \
      $S --model $out/merged --out runs/eval_go/$tag/samples.s{shard}.jsonl $PHENO)
  echo "$tag rewards=$rewards seed=$seed weights=${weights:-default} T=${temp:-default}  train=$t ddx=$d go=$g"
done
