#!/bin/bash
# E7 v2: GRPO with stabilized defaults (lr 5e-6, KL beta 0.04, no reward std normalization,
# 16 generations/prompt, 200 steps) after the v1 sweep collapsed to constant answers.
# Same chain as submit_e7.sh: train -> sample merged model on DDXPlus test + Phenopackets go -> score.
#   CONFIGS="acc-s0|acc|0 dist-s0|acc,brier,overlap|0" bash slurm/submit_e7v2.sh
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/sample.sbatch
TR=code/slurm/train.sbatch
TRAIN="--model Qwen/Qwen3-1.7B --train $THESIS/data/ddxplus/cases_train.jsonl --conditions $THESIS/data/ddxplus/conditions.json"
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
CONFIGS=${CONFIGS:-"acc-s0|acc|0 acc-s1|acc|1 dist-s0|acc,brier,overlap|0 dist-s1|acc,brier,overlap|1"}

# untrained reference (already sampled for v1)
for d in ddx go; do
  mkdir -p runs/e7v2_$d/qwen3-1.7b
  ln -sf "$THESIS"/runs/e7_$d/qwen3-1.7b/samples.s*.jsonl runs/e7v2_$d/qwen3-1.7b/
done

ddx_deps="" ; go_deps=""
for spec in $CONFIGS; do
  IFS="|" read -r tag rewards seed <<< "$spec"
  out=$THESIS/runs/train/v2-q3-1.7b-$tag
  tid=$(sbatch --parsable -p ckpt-all --time=10:00:00 --job-name=g2-$tag $TR $TRAIN --rewards $rewards --seed $seed --out $out)
  d=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --dependency=afterok:$tid --job-name=e7v2-ddx-$tag \
      $S --model $out/merged --out runs/e7v2_ddx/$tag/samples.s{shard}.jsonl $DDX)
  g=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --dependency=afterok:$tid --job-name=e7v2-go-$tag \
      $S --model $out/merged --out runs/e7v2_go/$tag/samples.s{shard}.jsonl $PHENO)
  ddx_deps="$ddx_deps:$d"; go_deps="$go_deps:$g"
  echo "$tag  rewards=$rewards seed=$seed  train=$tid  eval ddx=$d go=$g"
done

sd=$(sbatch --parsable -p compute --dependency=afterany$ddx_deps --job-name=score-e7v2-ddx \
  --export=ALL,RUNS=$THESIS/runs/e7v2_ddx,BASE=qwen3-1.7b code/slurm/score_ddx.sbatch)
sg=$(sbatch --parsable -A stf -p compute -c 4 --mem=24G --time=01:30:00 --dependency=afterany$go_deps \
  -J score-e7v2-go -o logs/score-e7v2-go-%j.out --export=ALL,RUNS=$THESIS/runs/e7v2_go,BASE=qwen3-1.7b \
  --wrap "bash $THESIS/code/slurm/score_go.sh")
echo "scoring: ddx=$sd go=$sg"
