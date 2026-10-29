#!/bin/bash
# MedGemma-4B pt/it pair (CPU): headline tests (review2), appendix readouts (r14), chance-corrected pass@32 (r18).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; R=$THESIS/runs; O=$HOME/thesis_overflow/runs; A=$HOME/thesis_overflow/analysis/r24
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
python review2.py --runs $O $C --pairs "medgemma_ddx:medgemma-4b-pt>medgemma-4b-it" --n-boot 10000 --out $A/review2_mg_zs.json > $A/review2_mg_zs.log 2>&1 &
python review2.py --runs $O $C --pairs "fewshotmg_t07:medgemma-4b-pt>medgemma-4b-it" --n-boot 10000 --out $A/review2_mg_fs.json > $A/review2_mg_fs.log 2>&1 &
python r14_an.py --runs $R --overflow $O $C --groups medgemma --out $A/r14_medgemma.json > $A/r14_medgemma.log 2>&1 &
python r18_chance.py --runs $R --overflow $O $C --out $A/r18_mg.json > $A/r18_mg.log 2>&1 &
wait
echo R24_DONE
