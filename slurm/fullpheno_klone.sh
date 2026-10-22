#!/bin/bash
# Assemble runs/go_full/<tag> = existing 500-case samples + 901 new cases, score, then build reports.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs
declare -A OLD=([qwen25-7b-base]=pheno25 [qwen25-7b-it]=go [alphamed-7b]=go [huatuo-o1-7b]=go [m1-7b]=go
                [alphamed-7b-base-rl]=go_base_rl [qwen3-8b-base]=go_qwen3 [qwen3-8b]=go_qwen3)
mkdir -p $R/go_full
for t in "${!OLD[@]}"; do
  d=$R/go_full/$t; mkdir -p $d
  cp $R/${OLD[$t]}/$t/samples.jsonl $d/samples.s_go.jsonl
  cp $R/go_full_rest/$t/samples.s0.jsonl $d/samples.s0.jsonl
  cp $R/go_full_rest/$t/samples.s1.jsonl $d/samples.s1.jsonl
  echo "$t $(cat $d/samples.s*.jsonl | wc -l) lines, $(cat $d/samples.s*.jsonl | python -c 'import sys,json; print(len({json.loads(l)["case_id"] for l in sys.stdin}))') unique"
done
RUNS=$R/go_full BASE=qwen25-7b-it bash $THESIS/code/slurm/score_go.sh
cd $THESIS/code/src
for spec in "go_full_pair25:qwen25-7b-base:qwen25-7b-base qwen25-7b-it" "go_full_q3:qwen3-8b-base:qwen3-8b-base qwen3-8b" "go_full_base_rl:qwen25-7b-base:qwen25-7b-base qwen25-7b-it alphamed-7b alphamed-7b-base-rl"; do
  name=${spec%%:*}; rest=${spec#*:}; base=${rest%%:*}; tags=${rest#*:}
  mkdir -p $R/$name; for t in $tags; do ln -sfn ../go_full/$t $R/$name/$t; done
  echo "########## $name"; python compare_runs.py --runs-dir $R/$name --base $base --out-dir $R/$name/report | grep -v "^figure"
done
echo "########## era"; python era_table.py --runs-dir $R/go_full --out $R/go_full/report/era.json
echo FULLPHENO_DONE
