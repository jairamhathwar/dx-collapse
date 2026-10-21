#!/bin/bash
# Score only the listed tags in DDXPlus and Phenopacket runs dirs, then rebuild each report.
#   sbatch --export=ALL,SPEC="e9_ddx:qwen3-8b-base:b8-acc-s1+b8-gm-s1 e9_go:qwen3-8b-base:b8-acc-s1" (tags joined with +; sbatch --export splits on commas) slurm/score_new.sbatch
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=score-new
#SBATCH --output=/gscratch/scrubbed/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
cd "$THESIS/code/src" || exit 1
for item in $SPEC; do
  dir=${item%%:*}; rest=${item#*:}; base=${rest%%:*}; tags=${rest#*:}
  RUNS=$THESIS/runs/$dir
  for t in ${tags//+/ }; do
    d=$RUNS/$t
    cat "$d"/samples.s*.jsonl > "$d/samples.jsonl"
    echo "=== $dir/$t ($(wc -l < "$d/samples.jsonl") cases)"
    if [[ $dir == *ddx* ]]; then
      python score_ddxplus.py --samples "$d/samples.jsonl" --cases "$THESIS/data/ddxplus/cases_test.jsonl" \
        --conditions "$THESIS/data/ddxplus/conditions.json" --out "$d/scores.json" --graded-out "$d/graded.jsonl" &
    else
      python score_samples.py --samples "$d/samples.jsonl" --cases "$THESIS/data/phenopackets/cases.jsonl" \
        --normalizer "$THESIS/data/ontology/normalizer.pkl" --out "$d/scores.json" --graded-out "$d/graded.jsonl" | tail -1 &
    fi
  done
  wait
  if [[ $dir == *ddx* ]]; then
    python compare_runs.py --runs-dir "$RUNS" --base "$base" --out-dir "$RUNS/report" \
      --title "DDXPlus differential diagnosis" --related-label "Gold or in ground-truth differential"
  else
    python compare_runs.py --runs-dir "$RUNS" --base "$base" --out-dir "$RUNS/report"
  fi
done
echo SCORE_NEW_DONE
