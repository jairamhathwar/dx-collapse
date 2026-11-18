#!/bin/bash
# Stage arithmetic: wider measurements on the existing 1B models (no training).
#   KL on three prompt sets (AmbigQA questions, held-out preference prompts, held-out SFT-mixture prompts) for every
#   arithmetic-vs-true pair, every stage effect and ALL seed pairs; greedy first-answer agreement on AmbigQA.
source ~/.thesis_env
cd "$THESIS"
O=/gpfs/scrubbed/jairamh/thesis_runs/stage; BASE=allenai/OLMo-2-0425-1B; TOK=allenai/OLMo-2-0425-1B-SFT
S=code/slurm/tillicum/sample.sbatch; A=$THESIS/analysis/stage; mkdir -p $A runs/stage_aq
E="source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src"
M="base=$BASE"; P=""
for s in 0 1 2 3; do
  M="$M S$s=$O/B-S-s$s D$s=$O/B2-D-s$s SD$s=$O/B2-SD-s$s aD$s=$O/B2-arithD-s$s aSD$s=$O/B2-arithSD-s$s T$s=$O/B4-T-s$s ST$s=$O/B4-ST-s$s aT$s=$O/B4-aT-s$s aST$s=$O/B4-aST-s$s"
  P="$P D$s:aD$s SD$s:aSD$s T$s:aT$s ST$s:aST$s D$s:base SD$s:S$s T$s:base ST$s:S$s S$s:base"
done
for s in 0 2; do
  M="$M xD$s=$O/B5-D-s$s xSD$s=$O/B5-SD-s$s xaD$s=$O/B5-aD-s$s xaSD$s=$O/B5-aSD-s$s"
  P="$P xD$s:xaD$s xSD$s:xaSD$s xD$s:base xSD$s:S$s"
done
P="$P xD0:xD2 xSD0:xSD2"
for a in 0 1 2; do for b in 1 2 3; do [ $a -lt $b ] && for k in S D SD T ST; do P="$P $k$a:$k$b"; done; done; done
pj=$(sbatch --parsable -p gpu-h200-mig --gpus=1 -c 2 --mem=32G --time=00:30:00 -J se-prompts -o logs/se-prompts-%j.out --wrap "$E; python stage_train.py prompts --kind pref --out $A/prompts_pref.jsonl; python stage_train.py prompts --kind sft --out $A/prompts_sft.jsonl" 2>/dev/null)
for kind in ambigqa pref sft; do
  f=$A/prompts_$kind.jsonl; [ $kind = ambigqa ] && f=$THESIS/data/ambigqa/cases_test.jsonl
  sbatch -p gpu-h200 --gpus=1 -c 4 --mem=64G --time=02:00:00 --dependency=afterok:$pj -J se-kl-$kind -o logs/se-kl-$kind-%j.out --wrap "$E; python kl_pair.py --tokenizer $TOK --prompts $f --out $A/kl_$kind.json $M --pairs $P" > /dev/null 2>&1
done
AQ="--task ambigqa --cases data/ambigqa/cases_test.jsonl --k 1 --temperature 0 --max-tokens 256 --max-model-len 2048 --shard 0 --num-shards 1"
deps=""; F=""
for m in $M; do n=${m%%=*}; p=${m#*=}
  j=$(sbatch --parsable -p gpu-h200 --time=00:12:00 --job-name=se-aq-$n $S --model $p --out runs/stage_aq/$n/samples.jsonl $AQ --chat-template code/templates/olmo2_chat.jinja 2>/dev/null)
  deps="$deps:$j"; F="$F $n=$THESIS/runs/stage_aq/$n/samples.jsonl"
done
sbatch --parsable -p gpu-h200-mig --gpus=1 -c 2 --mem=32G --time=00:30:00 --dependency=afterany$deps -J se-agree -o logs/se-agree-%j.out --wrap "$E; python agree_an.py $THESIS/data/ambigqa/cases_test.jsonl $A/agree_aq.json $F --pairs $P" 2>/dev/null
