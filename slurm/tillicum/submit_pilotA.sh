#!/bin/bash
# Pilot A (listing answer options): 300 AmbigQA questions, 32 samples, six models. New arms, both with a fresh
# candidate order per sample: "clean" = other models' frequent errors with this model's own top-3 errors kept off the
# list; "random" = distractors are answers to other questions. Existing arms: open (no list) and "own" (the first list,
# which can contain the model's own errors, fixed order). Same lenient scorer for every arm. Worst case about $5.
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch; D=data/ambigqa; R=$THESIS/runs
E="source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src"
python3 - <<PY
import json, random
ids = open("$D/candidates_ids.txt").read().split()
random.Random(1).shuffle(ids)
keep = set(ids[:300])
open("$D/p300_ids.txt", "w").write("\n".join(sorted(keep)) + "\n")
with open("$D/cases_p300.jsonl", "w") as fh:
    for l in open("$D/cases_test.jsonl"):
        if json.loads(l)["case_id"] in keep:
            fh.write(l)
PY
ALL="$R/ambigqa/q3-8b/samples.jsonl $R/ambigqa/q3-8b-base/samples.jsonl $R/ambigqa/q3-1.7b/samples.jsonl $R/p3/aq_open/olmo3-sft/samples.jsonl $R/p3/aq_open/olmo3-dpo/samples.jsonl $R/p3/aq_open/olmo3-instruct/samples.jsonl $R/p3/aq_open/olmo3-base/samples.jsonl"
AQ="--task ambigqa --cases $D/cases_test.jsonl --case-ids $D/p300_ids.txt --k 32 --max-tokens 512 --max-model-len 3072 --shard 0 --num-shards 1 --shuffle-candidates"
deps=""; A=""
for spec in "qwen3-8b-base|Qwen/Qwen3-8B-Base|$R/ambigqa/q3-8b-base/samples.jsonl|" "qwen3-8b|Qwen/Qwen3-8B|$R/ambigqa/q3-8b/samples.jsonl|" \
            "olmo3-base|allenai/Olmo-3-1025-7B|$R/p3/aq_open/olmo3-base/samples.jsonl|--chat-template code/templates/olmo3_sft.jinja" \
            "olmo3-sft|allenai/Olmo-3-7B-Instruct-SFT|$R/p3/aq_open/olmo3-sft/samples.jsonl|" \
            "olmo3-dpo|allenai/Olmo-3-7B-Instruct-DPO|$R/p3/aq_open/olmo3-dpo/samples.jsonl|" \
            "olmo3-instruct|allenai/Olmo-3-7B-Instruct|$R/p3/aq_open/olmo3-instruct/samples.jsonl|"; do
  IFS="|" read -r tag model own extra <<< "$spec"
  c=$(sbatch --parsable -p gpu-h200-mig --gpus=1 -c 2 --mem=16G --time=00:15:00 -J pA-c-$tag -o logs/pA-cands-%j.out --wrap "$E; python make_ambigqa_candidates.py $THESIS/$D/cases_test.jsonl $THESIS/$D/cand_clean_$tag.jsonl $ALL --case-ids $THESIS/$D/p300_ids.txt --exclude-from $own; python make_ambigqa_candidates.py $THESIS/$D/cases_test.jsonl $THESIS/$D/cand_random.$tag.jsonl $ALL --case-ids $THESIS/$D/p300_ids.txt --random-distractors" 2>/dev/null)
  j1=$(sbatch --parsable -p gpu-h200 --time=00:25:00 --dependency=afterok:$c --job-name=pA-clean-$tag $S --model $model --out runs/pA/clean/$tag/samples.jsonl $AQ --candidates $D/cand_clean_$tag.jsonl $extra 2>/dev/null)
  j2=$(sbatch --parsable -p gpu-h200 --time=00:25:00 --dependency=afterok:$c --job-name=pA-rand-$tag $S --model $model --out runs/pA/random/$tag/samples.jsonl $AQ --candidates $D/cand_random.$tag.jsonl $extra 2>/dev/null)
  deps="$deps:$j1:$j2"
  A="$A open/$tag:$own own/$tag:$R/p3/aq_closed/$tag/samples.jsonl clean/$tag:$R/pA/clean/$tag/samples.jsonl random/$tag:$R/pA/random/$tag/samples.jsonl"
  echo "$tag cands=$c clean=$j1 random=$j2"
done
sbatch --parsable -p gpu-h200-mig --gpus=1 -c 2 --mem=32G --time=00:30:00 --dependency=afterany$deps -J pA-score -o logs/pA-score-%j.out --wrap "$E; python qa_an.py $THESIS/$D/cases_p300.jsonl $THESIS/analysis/pilot/pA.json $A" 2>/dev/null
