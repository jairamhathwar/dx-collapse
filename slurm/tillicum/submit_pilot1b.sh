#!/bin/bash
# Pilot 1b: LoRA rank vs drift without the update-scale confound. Qwen3-1.7B, GRPO on DDXPlus (accuracy reward).
# Rank-stabilized LoRA (alpha 16, scale 16/sqrt(r)) at rank 4 / 64 / 256, each at four learning rates; full
# fine-tuning at three learning rates; seed 0. Drift is then compared at matched training reward across all runs
# (kl_drift.py in float32, with the base-vs-base floor reported). Worst case about $34.
source ~/.thesis_env
cd "$THESIS"
TR=code/slurm/tillicum/train.sbatch
deps=""; runs=""
sub() { tag=$1; shift; out=$THESIS/runs/train/p1b-$tag
  t=$(sbatch --parsable --time=02:10:00 --job-name=p1b-$tag $TR --rewards acc --seed 0 "$@" --out $out 2>/dev/null)
  deps="$deps:$t"; runs="$runs $out"; echo "p1b-$tag $t"; }
for r in 4 64 256; do for lr in 2.5e-6 5e-6 1e-5 2e-5; do sub r$r-lr$lr --lora-r $r --lora-alpha 16 --rslora --lr $lr; done; done
for lr in 5e-7 1e-6 2e-6; do sub full-lr$lr --full-ft --lr $lr --save-steps 50; done
sbatch --parsable -p gpu-h200 --gpus=1 -c 4 --mem=64G --time=01:30:00 --dependency=afterany$deps -J p1b-drift -o logs/p1b-drift-%j.out --wrap "source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src; python kl_drift.py --base Qwen/Qwen3-1.7B --runs $runs --off $THESIS/data/ambigqa/cases_test.jsonl --on $THESIS/data/ddxplus/cases_test.jsonl --conditions $THESIS/data/ddxplus/conditions.json --out $THESIS/analysis/pilot/p1b_drift.json" 2>/dev/null
