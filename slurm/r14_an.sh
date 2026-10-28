#!/bin/bash
# Tenth round (CPU): one pipeline for the DDXPlus appendix tables (review2 loader, condition-clustered intervals) and
# disease-clustered Phenopacket intervals. Code from the home clone.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; R=$THESIS/runs; A=$HOME/thesis_overflow/analysis/r14
mkdir -p $A
cd $HOME/thesis_repo/src
C="--runs $R --overflow $HOME/thesis_overflow/runs --cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
python r14_pheno.py --runs $R --out $A/r14_pheno.json > $A/r14_pheno.log 2>&1 &
for g in general,prompts robust,full stages big32 medical,grpo; do
  python r14_an.py $C --groups $g --out $A/r14_${g//,/_}.json > $A/r14_${g//,/_}.log 2>&1 &
done
wait
tail -n 5 $A/*.log | cut -c1-600
echo R14_DONE
