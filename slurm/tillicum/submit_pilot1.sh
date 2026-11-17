#!/bin/bash
# ICML idea pilot 1 (LoRA rank as an implicit trust region in RLVR). Qwen3-1.7B, GRPO on DDXPlus (accuracy reward),
# rank 4 / 64 / 256 x 2 seeds and full fine-tuning x 1 seed; then drift per checkpoint (kl_drift.py) and held-out
# evaluations of the final models (DDXPlus test, AmbigQA). Tight limits; worst case about $18.
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch; TR=code/slurm/tillicum/train.sbatch
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard 0 --num-shards 1"
AQ="--task ambigqa --cases data/ambigqa/cases_test.jsonl --k 32 --max-tokens 512 --max-model-len 2048 --shard 0 --num-shards 1"
deps=""; runs=""
for spec in "r4-s0|--lora-r 4 --seed 0" "r4-s1|--lora-r 4 --seed 1" "r64-s0|--lora-r 64 --seed 0" "r64-s1|--lora-r 64 --seed 1" \
            "r256-s0|--lora-r 256 --seed 0" "r256-s1|--lora-r 256 --seed 1" "full-s0|--full-ft --lr 1e-6 --save-steps 50 --seed 0"; do
  IFS="|" read -r tag args <<< "$spec"; out=$THESIS/runs/train/p1-$tag
  t=$(sbatch --parsable --time=02:15:00 --job-name=p1-$tag $TR --rewards acc $args --out $out 2>/dev/null)
  sbatch -p gpu-h200 --time=00:15:00 --dependency=afterok:$t --job-name=p1-ed-$tag $S --model $out/merged --out runs/p1/ddx/$tag/samples.jsonl $DDX > /dev/null 2>&1
  sbatch -p gpu-h200 --time=00:15:00 --dependency=afterok:$t --job-name=p1-ea-$tag $S --model $out/merged --out runs/p1/aq/$tag/samples.jsonl $AQ > /dev/null 2>&1
  deps="$deps:$t"; runs="$runs $out"; echo "p1-$tag train=$t"
done
sbatch --parsable -p gpu-h200 --gpus=1 -c 4 --mem=64G --time=00:45:00 --dependency=afterany$deps -J p1-drift -o logs/p1-drift-%j.out --wrap "source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src; python kl_drift.py --base Qwen/Qwen3-1.7B --runs $runs --off $THESIS/data/ambigqa/cases_test.jsonl --on $THESIS/data/ddxplus/cases_test.jsonl --conditions $THESIS/data/ddxplus/conditions.json --out $THESIS/analysis/pilot/p1_drift.json" 2>/dev/null
