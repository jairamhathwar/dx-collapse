#!/bin/bash
# Tillicum: GRPO train (full H200) -> sample merged model on DDXPlus test + Phenopackets go (MIG slices).
# Same stabilized defaults as Klone v2 (train_grpo.py); per-config reward weights and train temperature.
#   CONFIGS="tag|rewards|seed|weights|temperature|model ..." [EVAL_PART=gpu-h200] [TRAIN_TIME=20:00:00] bash slurm/tillicum/submit_train_eval.sh
# Empty fields keep defaults (model Qwen/Qwen3-1.7B). Use EVAL_PART=gpu-h200 for models too big for an 18 GB MIG slice.
# Example: CONFIGS="gm3t12-s0|acc,group_match|0|1,1,3|1.2" bash slurm/tillicum/submit_train_eval.sh
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
TR=code/slurm/tillicum/train.sbatch
S=code/slurm/tillicum/sample.sbatch
EVAL_PART=${EVAL_PART:-gpu-h200-mig}
TRAIN_TIME=${TRAIN_TIME:-10:00:00}   # 8B runs need ~20h (about 320 s/step)
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"

for spec in $CONFIGS; do
  IFS="|" read -r tag rewards seed weights temp model <<< "$spec"
  out=$THESIS/runs/train/$tag
  extra=""
  [ -n "${weights:-}" ] && extra="$extra --weights $weights"
  [ -n "${temp:-}" ] && extra="$extra --temperature $temp"
  [ -n "${model:-}" ] && extra="$extra --model $model"
  t=$(sbatch --parsable --time=$TRAIN_TIME --job-name=t-$tag $TR --rewards $rewards --seed $seed --out $out $extra)
  d=$(sbatch --parsable -p $EVAL_PART --array=0-1 --dependency=afterok:$t --job-name=ev-ddx-$tag \
      $S --model $out/merged --out runs/eval_ddx/$tag/samples.s{shard}.jsonl $DDX)
  g=$(sbatch --parsable -p $EVAL_PART --array=0-1 --dependency=afterok:$t --job-name=ev-go-$tag \
      $S --model $out/merged --out runs/eval_go/$tag/samples.s{shard}.jsonl $PHENO)
  echo "$tag rewards=$rewards seed=$seed weights=${weights:-default} T=${temp:-default} model=${model:-Qwen3-1.7B}  train=$t ddx=$d go=$g"
done
