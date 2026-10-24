#!/bin/bash
# Score and analyze the 32B replication (runs/big32_ddx) and the full 49-condition prompt runs (runs/pv_full_t07).
# Usage: bash big32_full49.sh [full49-tags]   (default: the four Qwen tags; the OLMo tags are added after their resume)
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; D=$THESIS/data/ddxplus
FULLTAGS=${1:-"qwen3-8b-base qwen3-8b qwen25-7b-base qwen25-7b-it"}
cd $THESIS/code
echo "########## score big32_ddx"
RUNS=$R/big32_ddx BASE=olmo31-32b-base bash slurm/score_ddx.sbatch > $THESIS/logs/big32-score.log 2>&1
cd src
echo "########## score pv_full_t07 (top-1 = highest stated probability)"
for t in $FULLTAGS; do
  d=$R/pv_full_t07/$t; cat $d/samples.s*.jsonl > $d/samples.jsonl
  python score_ddxplus.py --top-by-prob --samples $d/samples.jsonl --cases $D/cases_test.jsonl --conditions $D/conditions.json \
    --out $d/scores.json --graded-out $d/graded.jsonl > /dev/null &
done
wait
echo "########## 32B stage-to-stage and pair comparisons"
python compare_recipes.py --runs-dir $R/big32_ddx --a qwen25-32b-it --b qwen25-32b-base
python compare_recipes.py --runs-dir $R/big32_ddx --a olmo31-32b-sft --b olmo31-32b-base
python compare_recipes.py --runs-dir $R/big32_ddx --a olmo31-32b-dpo --b olmo31-32b-sft
python compare_recipes.py --runs-dir $R/big32_ddx --a olmo31-32b-instruct --b olmo31-32b-dpo
python compare_recipes.py --runs-dir $R/big32_ddx --a olmo31-32b-instruct --b olmo31-32b-base
echo "########## 32B format robustness"
python format_robust.py --runs-dir $R/big32_ddx --base qwen25-32b-base --post qwen25-32b-it | grep -E "unparsed|parsed-only"
python format_robust.py --runs-dir $R/big32_ddx --base olmo31-32b-base --post olmo31-32b-instruct | grep -E "unparsed|parsed-only"
echo "########## 32B severity"
python per_condition.py --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json \
  --model base=$R/big32_ddx/qwen25-32b-base/samples.jsonl --model instruct=$R/big32_ddx/qwen25-32b-it/samples.jsonl \
  --ref base --out-dir $R/analysis/pc_qwen25_32b > /dev/null
python per_condition.py --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json \
  $(for t in base sft dpo instruct; do echo --model $t=$R/big32_ddx/olmo31-32b-$t/samples.jsonl; done) \
  --ref base --out-dir $R/analysis/pc_olmo31_32b > /dev/null
for p in pc_qwen25_32b pc_olmo31_32b; do python -c "
import json; d=json.load(open('$R/analysis/$p/per_condition.json'))
for tier,v in d['severity'].items():
    print('$p', tier, {m:(round(x['considered_top1'],3), [round(100*y,1) for y in x.get('delta_vs_ref',[])]) for m,x in v.items()})
"; done
echo "########## stated vs sampled and consequences"
for spec in "big32_ddx qwen25-32b-base,qwen25-32b-it big32_q" "big32_ddx olmo31-32b-base,olmo31-32b-sft,olmo31-32b-dpo,olmo31-32b-instruct big32_o"; do
  set -- $spec
  python stated_vs_sampled.py --runs-dir $R/$1 --tags $2 --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $R/analysis/svs/$3.json
  python consequences.py --runs-dir $R/$1 --tags $2 --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $R/analysis/cons/$3.json
done
for pair in "qwen3-8b-base,qwen3-8b full_q3" "qwen25-7b-base,qwen25-7b-it full_q25" "olmo3-base,olmo3-instruct full_olmo"; do
  set -- $pair
  case " $FULLTAGS " in *" ${1%%,*} "*) ;; *) continue ;; esac
  python stated_vs_sampled.py --top-by-prob --runs-dir $R/pv_full_t07 --tags $1 --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $R/analysis/svs/$2.json
  python consequences.py --top-by-prob --runs-dir $R/pv_full_t07 --tags $1 --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json --out $R/analysis/cons/$2.json
done
echo "########## big32 pass@k report"
awk '/pass@k \(exact\)/{p=1} p' $THESIS/logs/big32-score.log | grep -vE "^figure|^\s+related"
echo BIG32_DONE
