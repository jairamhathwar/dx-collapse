#!/bin/bash
# Score every model under a DDXPlus runs dir and build the comparison report.
# Submitted with --dependency=afterany:<sampling job ids> so it runs unattended.
#   sbatch --export=ALL,RUNS=$THESIS/runs/ddx_qwen3,BASE=qwen3-8b-base slurm/score_ddx.sbatch
#SBATCH --account=stf
#SBATCH --partition=ckpt-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=score-ddx
#SBATCH --output=/gscratch/scrubbed/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
cd "$THESIS/code/src" || exit 1
RUNS=${RUNS:-$THESIS/runs/ddx}
BASE=${BASE:-qwen25-7b-it}
D=$THESIS/data/ddxplus
for d in "$RUNS"/*/; do
  tag=$(basename "$d")
  [ "$tag" = report ] && continue
  cat "$d"/samples.s*.jsonl > "$d/samples.jsonl"
  echo "=== $tag ($(wc -l < "$d/samples.jsonl") cases)"
  python score_ddxplus.py --samples "$d/samples.jsonl" --cases "$D/cases_test.jsonl" \
    --conditions "$D/conditions.json" --out "$d/scores.json" --graded-out "$d/graded.jsonl" &
done
wait
python compare_runs.py --runs-dir "$RUNS" --base "$BASE" --out-dir "$RUNS/report" \
  --title "DDXPlus differential diagnosis" --related-label "Gold or in ground-truth differential"
