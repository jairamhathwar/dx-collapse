#!/bin/bash
# ICML idea pilot 3 (stage crossing; listed vs unlisted answers). Tight limits; worst case about $9.
#   OLMo 3 7B base / SFT / DPO / Instruct / RL-Zero and the Qwen3-8B pair, each on:
#   DDXPlus closed (existing or rerun), DDXPlus open (no condition list), AmbigQA open, AmbigQA closed (candidates).
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch
G="-p gpu-h200 --time=00:25:00"
ENV="source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src"
c=$(sbatch --parsable -p gpu-h200-mig --gpus=1 -c 2 --mem=16G --time=00:15:00 -J p3-cands -o logs/p3-cands-%j.out --wrap "$ENV; python make_ambigqa_candidates.py $THESIS/data/ambigqa/cases_test.jsonl $THESIS/data/ambigqa/candidates_test.jsonl $THESIS/runs/ambigqa/q3-8b/samples.jsonl $THESIS/runs/ambigqa/q3-8b-base/samples.jsonl $THESIS/runs/ambigqa/q3-1.7b/samples.jsonl")
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 6144 --shard 0 --num-shards 1"
AQ="--task ambigqa --cases data/ambigqa/cases_test.jsonl --k 32 --max-tokens 512 --max-model-len 3072 --shard 0 --num-shards 1"
for spec in "olmo3-base|allenai/Olmo-3-1025-7B|--chat-template code/templates/olmo3_sft.jinja" "olmo3-sft|allenai/Olmo-3-7B-Instruct-SFT|" \
            "olmo3-dpo|allenai/Olmo-3-7B-Instruct-DPO|" "olmo3-instruct|allenai/Olmo-3-7B-Instruct|" \
            "olmo3-rlzero|allenai/Olmo-3-7B-RL-Zero-General|" "qwen3-8b-base|Qwen/Qwen3-8B-Base|" "qwen3-8b|Qwen/Qwen3-8B|"; do
  IFS="|" read -r tag model extra <<< "$spec"
  sbatch $G --job-name=p3-do-$tag $S --model $model --out runs/p3/ddx_open/$tag/samples.jsonl $DDX --prompt-variant open $extra > /dev/null
  sbatch $G --job-name=p3-ac-$tag --dependency=afterok:$c $S --model $model --out runs/p3/aq_closed/$tag/samples.jsonl $AQ --candidates data/ambigqa/candidates_test.jsonl $extra > /dev/null
  case $tag in qwen3*) sbatch $G --job-name=p3-dc-$tag $S --model $model --out runs/p3/ddx_closed/$tag/samples.jsonl $DDX $extra > /dev/null ;;
               *) sbatch $G --job-name=p3-ao-$tag $S --model $model --out runs/p3/aq_open/$tag/samples.jsonl $AQ $extra > /dev/null ;; esac
  echo "submitted $tag"
done
