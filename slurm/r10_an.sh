#!/bin/bash
# Must-not-miss conditions defined outside DDXPlus (src/make_mnm_sets.py): re-run the severe-condition analyses with
# the harm-based, chest-pain-guideline and union sets in place of DDXPlus severity <= 2. Code from the home clone.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$THESIS/runs; O=$H/runs; F=$O/fewshot_t07; RF=$O/fewshot_rf_t07; A=$H/analysis/r10
mkdir -p $A $A/sets
cd $HOME/thesis_repo/src
python make_mnm_sets.py --conditions-json $D/release_conditions.json --out-dir $A/sets
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
FS="$F:qwen3-8b-base>qwen3-8b $F:qwen25-7b-base>qwen25-7b-it $F:qwen25-32b-base>qwen25-32b-it $F:olmo3-base>olmo3-instruct $F:olmo31-32b-base>olmo31-32b-instruct $F:mistral7b-base>mistral7b-it"
RFP="$RF:qwen3-8b-base>qwen3-8b $RF:qwen25-7b-base>qwen25-7b-it $RF:qwen25-32b-base>qwen25-32b-it $RF:olmo3-base>olmo3-instruct $RF:olmo31-32b-base>olmo31-32b-instruct $RF:mistral7b-base>mistral7b-it"
MP="$O/mnm_t07/qwen3-8b>$R/pair_q3/qwen3-8b $O/mnm_t07/qwen25-7b-it>$R/pair25_t07/qwen25-7b-it $O/mnm_t07/qwen25-32b-it>$R/big32_ddx/qwen25-32b-it \
$O/mnm_t07/olmo3-instruct>$R/ddx_olmo/olmo3-instruct $O/mnm_t07/olmo31-32b-instruct>$R/big32_ddx/olmo31-32b-instruct $O/mnm_t07/mistral7b-it>$O/mistral_t07/mistral7b-it \
$O/mnm_t07/qwen3-8b>$F/qwen3-8b-base $O/mnm_t07/qwen25-7b-it>$F/qwen25-7b-base $O/mnm_t07/qwen25-32b-it>$F/qwen25-32b-base \
$O/mnm_t07/olmo3-instruct>$F/olmo3-base $O/mnm_t07/olmo31-32b-instruct>$F/olmo31-32b-base $O/mnm_t07/mistral7b-it>$F/mistral7b-base"
for s in union harm chestpain; do
  C="--cases $D/cases_test.jsonl --conditions-json $A/sets/release_conditions_mnm_$s.json"
  python review2.py --runs $R $C --pairs "$ZS" --out $A/review2_zs_$s.json > $A/review2_zs_$s.log 2>&1 &
  python review2.py --runs $R $C --pairs "$FS" --out $A/review2_fs_$s.json > $A/review2_fs_$s.log 2>&1 &
  python r7_analysis.py --mode mnm --runs / $C --pairs "$MP" --out $A/mnm_$s.json > $A/mnm_$s.log 2>&1 &
done
C="--cases $D/cases_test.jsonl --conditions-json $A/sets/release_conditions_mnm_union.json"
python review2.py --runs $R $C --pairs "$RFP" --out $A/review2_rf_union.json > $A/review2_rf_union.log 2>&1 &
python r8_controls.py --spec $HOME/thesis_repo/slurm/r8_spec.json $C --out $A/controls_union.json > $A/controls_union.log 2>&1 &
wait
tail -n 30 $A/*.log | cut -c1-1200
echo R10_DONE
