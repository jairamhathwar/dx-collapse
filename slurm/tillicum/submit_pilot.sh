#!/bin/bash
# ICML pilot on Tillicum (H200, ~$0.90/GPU-h). Tight time limits cap the cost of a hung job (worst case ~$16).
#   bash slurm/tillicum/submit_pilot.sh
# 1. Size ladder on DDXPlus: Qwen2.5 0.5/1.5/3/14B base T0.7, Instruct T0.7, Instruct min-p 0.1 T1.5 (runs/ladder/).
# 2. Self-targeted GRPO (Qwen3-1.7B): sample the starting model 4x per training case, build its mean stated list per
#    case (frozen target), train acc+self_witness, self_witness only, acc+ref_witness (seed 0, 200 steps), and sample
#    each merged model on the test set (runs/selftarget_ddx/).
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch
TR=code/slurm/tillicum/train.sbatch
DDX1="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 4096 --max-model-len 6144 --shard 0 --num-shards 1"
DDX2="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 4096 --max-model-len 6144 --shard -1 --num-shards 2"
MP="--temperature 1.5 --top-p 1.0 --min-p 0.1"
G="-p gpu-h200"
for sz in 0.5 1.5 3; do
  t=$(echo $sz | tr -d .)
  sbatch $G --time=00:20:00 --job-name=lad-q$t-base $S --model Qwen/Qwen2.5-${sz}B --out runs/ladder/q25-${t}b-base-t07/samples.jsonl $DDX1
  sbatch $G --time=00:20:00 --job-name=lad-q$t-it $S --model Qwen/Qwen2.5-${sz}B-Instruct --out runs/ladder/q25-${t}b-it-t07/samples.jsonl $DDX1
  sbatch $G --time=00:20:00 --job-name=lad-q$t-itmp $S --model Qwen/Qwen2.5-${sz}B-Instruct --out runs/ladder/q25-${t}b-it-minp15/samples.jsonl $DDX1 $MP
done
sbatch $G --time=00:30:00 --array=0-1 --job-name=lad-q14-base $S --model Qwen/Qwen2.5-14B --out runs/ladder/q25-14b-base-t07/samples.s{shard}.jsonl $DDX2
sbatch $G --time=00:30:00 --array=0-1 --job-name=lad-q14-it $S --model Qwen/Qwen2.5-14B-Instruct --out runs/ladder/q25-14b-it-t07/samples.s{shard}.jsonl $DDX2
sbatch $G --time=00:30:00 --array=0-1 --job-name=lad-q14-itmp $S --model Qwen/Qwen2.5-14B-Instruct --out runs/ladder/q25-14b-it-minp15/samples.s{shard}.jsonl $DDX2 $MP

ST=$THESIS/data/ddxplus/self_target_q3-1.7b.json
SD=runs/selftarget/q3-1.7b-train
s=$(sbatch --parsable $G --time=00:45:00 --array=0-1 --job-name=st-sample $S --model Qwen/Qwen3-1.7B \
    --task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_train.jsonl --k 4 \
    --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2 --out $SD/samples.s{shard}.jsonl)
m=$(sbatch --parsable -p gpu-h200-mig --gpus=1 -c 2 --mem=16G --time=00:20:00 --dependency=afterok:$s -J st-build \
    -o logs/st-build-%j.out --wrap "source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src; cat $THESIS/$SD/samples.s*.jsonl > $THESIS/$SD/samples.jsonl; python make_self_targets.py $THESIS/$SD/samples.jsonl $THESIS/data/ddxplus/conditions.json $ST")
echo "self-target sampling=$s build=$m"
EV="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --max-tokens 2048 --max-model-len 4096 --shard -1 --num-shards 2"
for spec in "st-accsw-s0|acc,self_witness" "st-sw-s0|self_witness" "st-accrw-s0|acc,ref_witness"; do
  IFS="|" read -r tag rewards <<< "$spec"
  out=$THESIS/runs/train/$tag
  t=$(sbatch --parsable --time=02:30:00 --dependency=afterok:$m --job-name=t-$tag $TR --rewards $rewards --seed 0 --self-targets $ST --out $out)
  d=$(sbatch --parsable $G --time=00:30:00 --array=0-1 --dependency=afterok:$t --job-name=ev-$tag $S --model $out/merged --out runs/selftarget_ddx/$tag/samples.s{shard}.jsonl $EV)
  echo "$tag rewards=$rewards train=$t eval=$d"
done
