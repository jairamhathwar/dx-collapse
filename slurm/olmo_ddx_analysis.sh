#!/bin/bash
# OLMo 3 DDXPlus stage analyses (needs scored runs in ddx_olmo, ddx_olmo_t1): format robustness,
# stage-to-stage paired comparisons, severity tiers.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs
cd $THESIS/code/src
for d in ddx_olmo ddx_olmo_t1; do
  for t in olmo3-sft olmo3-dpo olmo3-instruct olmo3-rlzero-general; do
    echo "##### $d format $t"; python format_robust.py --runs-dir $R/$d --base olmo3-base --post $t
  done
  echo "##### $d stage-to-stage"
  python compare_recipes.py --runs-dir $R/$d --a olmo3-sft --b olmo3-base
  python compare_recipes.py --runs-dir $R/$d --a olmo3-dpo --b olmo3-sft
  python compare_recipes.py --runs-dir $R/$d --a olmo3-instruct --b olmo3-dpo
  python compare_recipes.py --runs-dir $R/$d --a olmo3-rlzero-general --b olmo3-base
  python per_condition.py --cases $THESIS/data/ddxplus/cases_test.jsonl --conditions-json $THESIS/data/ddxplus/release_conditions.json \
    $(for t in olmo3-base olmo3-sft olmo3-dpo olmo3-instruct olmo3-rlzero-general; do echo --model ${t#olmo3-}=$R/$d/$t/samples.jsonl; done) \
    --ref base --out-dir $R/analysis/pc_$d > /dev/null
  python -c "
import json; d=json.load(open('$R/analysis/pc_$d/per_condition.json'))
for tier,v in d['severity'].items():
    print('$d', tier, {m:(round(x['considered_top1'],3), [round(100*y,1) for y in x.get('delta_vs_ref',[])]) for m,x in v.items()})
"
done
echo ANALYSIS_DONE
