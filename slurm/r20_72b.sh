#!/bin/bash
# Qwen2.5-72B pair (CPU): headline tests (review2), appendix readouts (r14), chance-corrected pass@32 (r18), breadth (r16).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; R=$THESIS/runs; O=$HOME/thesis_overflow/runs; A=$HOME/thesis_overflow/analysis/r20
mkdir -p $A
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
python review2.py --runs $R $C --pairs "big72_ddx:qwen25-72b-base>qwen25-72b-it" --n-boot 10000 --out $A/review2_72b_zs.json > $A/review2_72b_zs.log 2>&1 &
python review2.py --runs $O $C --pairs "fewshot72_t07:qwen25-72b-base>qwen25-72b-it" --n-boot 10000 --out $A/review2_72b_fs.json > $A/review2_72b_fs.log 2>&1 &
python r14_an.py --runs $R --overflow $O $C --groups big72 --out $A/r14_big72.json > $A/r14_big72.log 2>&1 &
python r18_chance.py --runs $R --overflow $O $C --out $A/r18_72b.json > $A/r18_72b.log 2>&1 &
python r16_breadth.py --runs $R $C --out $A/r16_72b.json > $A/r16_72b.log 2>&1 &
wait
tail -n 4 $A/*.log | cut -c1-500
echo R20_DONE
