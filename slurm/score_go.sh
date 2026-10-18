#!/bin/bash
# Score every model under a Phenopacket runs dir: merge shards, grade, write scores.json + graded.jsonl,
# then build the comparison report.   RUNS=... BASE=... bash score_go.sh
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
cd "$THESIS/code/src" || exit 1
RUNS=${RUNS:-$THESIS/runs/go}
BASE=${BASE:-qwen25-7b-it}
for d in "$RUNS"/*/; do
  tag=$(basename "$d")
  [ "$tag" = report ] && continue
  cat "$d"/samples.s*.jsonl > "$d/samples.jsonl"
  echo "=== $tag"
  python score_samples.py --samples "$d/samples.jsonl" \
    --cases "$THESIS/data/phenopackets/cases.jsonl" \
    --normalizer "$THESIS/data/ontology/normalizer.pkl" \
    --out "$d/scores.json" --graded-out "$d/graded.jsonl" | tail -1 &
done
wait
python compare_runs.py --runs-dir "$RUNS" --base "$BASE" --out-dir "$RUNS/report"
