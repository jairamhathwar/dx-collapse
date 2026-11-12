#!/bin/bash
# ICML pilot: self-targeted GRPO (Qwen3-1.7B, DDXPlus). Same stabilized settings as E7 v2.
#   1. sample the starting model 4x on every training case; 2. build its mean stated list per case (self-targets);
#   3. train arms (seed 0, 200 steps); 4. sample each merged model on the DDXPlus test set (980 x 32).
# Baselines already exist: runs/e7v2_ddx/{qwen3-1.7b, acc-s0, gm-s0, gm3t12-s0}.
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/sample.sbatch
TR=code/slurm/train.sbatch
ST=$THESIS/data/ddxplus/self_target_q3-1.7b.json
TRAIN="--model Qwen/Qwen3-1.7B --train $THESIS/data/ddxplus/cases_train.jsonl --conditions $THESIS/data/ddxplus/conditions.json --self-targets $ST"
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
s=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --job-name=st-sample $S --model Qwen/Qwen3-1.7B \
    --task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_train.jsonl --k 4 \
    --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2 --out runs/selftarget/q3-1.7b-train/samples.s{shard}.jsonl)
m=$(sbatch --parsable -A stf -p compute -c 2 --mem=16G --time=00:30:00 --dependency=afterok:$s -J st-build \
    -o logs/st-build-%j.out --wrap "source ~/.thesis_env; module load conda/Miniforge3-25.9.1-0; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src; cat $THESIS/runs/selftarget/q3-1.7b-train/samples.s*.jsonl > $THESIS/runs/selftarget/q3-1.7b-train/samples.jsonl; python make_self_targets.py $THESIS/runs/selftarget/q3-1.7b-train/samples.jsonl $THESIS/data/ddxplus/conditions.json $ST")
echo "self-target sampling=$s build=$m"
for spec in "accsw-s0|acc,self_witness|0" "sw-s0|self_witness|0" "accrw-s0|acc,ref_witness|0"; do
  IFS="|" read -r tag rewards seed <<< "$spec"
  out=$THESIS/runs/train/st-q3-1.7b-$tag
  tid=$(sbatch --parsable -p ckpt-all --time=10:00:00 --dependency=afterok:$m --job-name=st-$tag $TR $TRAIN --rewards $rewards --seed $seed --out $out)
  d=$(sbatch --parsable -p ckpt-all --time=04:00:00 --array=0-1 --dependency=afterok:$tid --job-name=st-ddx-$tag \
      $S --model $out/merged --out runs/selftarget_ddx/$tag/samples.s{shard}.jsonl $DDX)
  echo "$tag rewards=$rewards train=$tid eval=$d"
done
