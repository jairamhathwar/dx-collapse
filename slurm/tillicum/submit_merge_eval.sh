#!/bin/bash
# Tillicum: merge a copied LoRA adapter (runs/train/<dir>) on a MIG slice, then evaluate it on
# DDXPlus test + Phenopackets go (MIG slices). Used for runs trained on Klone.
#   RUNS="tag:train_dir ..." bash slurm/tillicum/submit_merge_eval.sh
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
for spec in $RUNS; do
  tag=${spec%%:*}; dir=$THESIS/runs/train/${spec#*:}
  m=$(sbatch --parsable -A stf -p gpu-h200-mig --qos=normal --gpus=1 -c 2 --mem=32G --time=00:45:00 -J merge-$tag \
      -o $THESIS/logs/merge-$tag-%j.out --wrap "source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; \
      conda activate \$THESIS/envs/thesis; python \$THESIS/code/src/merge_lora.py --adapter $dir --out $dir/merged")
  d=$(sbatch --parsable --array=0-1 --dependency=afterok:$m --job-name=ev-ddx-$tag $S --model $dir/merged \
      --out runs/eval_ddx/$tag/samples.s{shard}.jsonl $DDX)
  g=$(sbatch --parsable --array=0-1 --dependency=afterok:$m --job-name=ev-go-$tag $S --model $dir/merged \
      --out runs/eval_go/$tag/samples.s{shard}.jsonl $PHENO)
  echo "$tag  merge=$m ddx=$d go=$g"
done
