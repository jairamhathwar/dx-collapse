#!/bin/bash
# Score OLMo 3 stage runs (DDXPlus T=0.7/T=1, Phenopackets full) and run stage analyses.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs
cd $THESIS/code
for d in ddx_olmo ddx_olmo_t1; do
  RUNS=$R/$d BASE=olmo3-base bash slurm/score_ddx.sbatch > $THESIS/logs/olmo-score-$d.log 2>&1 &
done
RUNS=$R/go_full_olmo BASE=olmo3-base bash slurm/score_go.sh > $THESIS/logs/olmo-score-go.log 2>&1 &
wait
cd src
for d in ddx_olmo ddx_olmo_t1; do echo "########## $d"; awk '/pass@k \(exact\)/{p=1} p' $THESIS/logs/olmo-score-$d.log | grep -v ^figure; done
echo "########## go_full_olmo"; awk '/pass@k \(exact\)/{p=1} p' $THESIS/logs/olmo-score-go.log | grep -v ^figure
for t in olmo3-sft olmo3-dpo olmo3-instruct olmo3-rlzero-general; do
  echo "##### format $t"; python format_robust.py --runs-dir $R/ddx_olmo --base olmo3-base --post $t | sed -n '1p;7p;8p'
done
echo "##### stage-to-stage (T=0.7)"
python compare_recipes.py --runs-dir $R/ddx_olmo --a olmo3-sft --b olmo3-base
python compare_recipes.py --runs-dir $R/ddx_olmo --a olmo3-dpo --b olmo3-sft
python compare_recipes.py --runs-dir $R/ddx_olmo --a olmo3-instruct --b olmo3-dpo
python per_condition.py --cases $THESIS/data/ddxplus/cases_test.jsonl --conditions-json $THESIS/data/ddxplus/release_conditions.json \
  $(for t in olmo3-base olmo3-sft olmo3-dpo olmo3-instruct olmo3-rlzero-general; do echo --model ${t#olmo3-}=$R/ddx_olmo/$t/samples.jsonl; done) \
  --ref base --out-dir $R/analysis/pc_olmo > /dev/null
python -c "
import json; d=json.load(open('$R/analysis/pc_olmo/per_condition.json'))
for tier,v in d['severity'].items():
    print(tier, {m:(round(x['considered_top1'],3), [round(100*y,1) for y in x.get('delta_vs_ref',[])]) for m,x in v.items()})
"
echo OLMO_DONE
