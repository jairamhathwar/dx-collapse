#!/bin/bash
# E7/E8: GRPO reward-recipe comparison on Qwen3-1.7B, with automatic evaluation.
# For every config: train (unless an existing training job id is given) -> sample the merged
# model on DDXPlus test (980 x 32) and Phenopackets go set (500 x 32) -> score everything
# against the untrained Qwen3-1.7B. All steps are chained with Slurm dependencies.
#   bash slurm/submit_e7.sh
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/sample.sbatch
TR=code/slurm/train.sbatch
TRAIN="--model Qwen/Qwen3-1.7B --train $THESIS/data/ddxplus/cases_train.jsonl --conditions $THESIS/data/ddxplus/conditions.json --max-steps 300"
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
mkdir -p runs/e7_ddx runs/e7_go

# tag | rewards | seed | existing training job id (empty = submit new)
CONFIGS=(
  "acc-s0|acc|0|41982845"
  "dist-s0|acc,brier,overlap|0|41982846"
  "brier-s0|acc,brier|0|"
  "overlap-s0|acc,overlap|0|"
  "puri-rlvr-s0|multi_rlvr|0|"
  "puri-rlcr-s0|multi_rlvr,set_brier|0|"
  "acc-s1|acc|1|"
  "dist-s1|acc,brier,overlap|1|"
)

# the first dist run was launched before this script, under its full reward name
ln -sfn q3-1.7b-acc-brier-overlap-s0 "$THESIS/runs/train/q3-1.7b-dist-s0"

# untrained reference: DDXPlus samples already exist; Phenopackets needs sampling
mkdir -p runs/e7_ddx/qwen3-1.7b
ln -sf "$THESIS"/runs/train_base/ddx/qwen3-1.7b/samples.s*.jsonl runs/e7_ddx/qwen3-1.7b/
pb=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --job-name=e7-go-base $S \
  --model Qwen/Qwen3-1.7B --out runs/e7_go/qwen3-1.7b/samples.s{shard}.jsonl $PHENO)
ddx_deps="" ; go_deps=":$pb"

for spec in "${CONFIGS[@]}"; do
  IFS="|" read -r tag rewards seed tid <<< "$spec"
  out=$THESIS/runs/train/q3-1.7b-$tag
  if [ -z "$tid" ]; then
    tid=$(sbatch --parsable -p ckpt-all --job-name=grpo-$tag $TR $TRAIN --rewards $rewards --seed $seed --out $out)
  fi
  d=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --dependency=afterok:$tid --job-name=e7-ddx-$tag \
      $S --model $out/merged --out runs/e7_ddx/$tag/samples.s{shard}.jsonl $DDX)
  g=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --dependency=afterok:$tid --job-name=e7-go-$tag \
      $S --model $out/merged --out runs/e7_go/$tag/samples.s{shard}.jsonl $PHENO)
  ddx_deps="$ddx_deps:$d"; go_deps="$go_deps:$g"
  echo "$tag  rewards=$rewards seed=$seed  train=$tid  eval ddx=$d go=$g"
done

sd=$(sbatch --parsable -p compute --dependency=afterany$ddx_deps --job-name=score-e7-ddx \
  --export=ALL,RUNS=$THESIS/runs/e7_ddx,BASE=qwen3-1.7b code/slurm/score_ddx.sbatch)
sg=$(sbatch --parsable -A stf -p compute -c 4 --mem=24G --time=01:30:00 --dependency=afterany$go_deps \
  -J score-e7-go -o logs/score-e7-go-%j.out --export=ALL,RUNS=$THESIS/runs/e7_go,BASE=qwen3-1.7b \
  --wrap "bash $THESIS/code/slurm/score_go.sh")
echo "scoring: ddx=$sd go=$sg"
