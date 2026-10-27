#!/bin/bash
# Matched-temperature and reference-free few-shot controls (CPU). Reads and writes under ~/thesis_overflow.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$H/runs; A=$H/analysis/r4
mkdir -p $A $R/matchT
cd $THESIS/code/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
# one directory with every temperature of the two Qwen pairs
for spec in qwen3-8b-base-t07:$THESIS/runs/pair_q3/qwen3-8b-base qwen3-8b-base-t1:$THESIS/runs/ddx_qwen3_t1/qwen3-8b-base \
            qwen3-8b-t07:$THESIS/runs/pair_q3/qwen3-8b qwen3-8b-t1:$THESIS/runs/ddx_qwen3_t1/qwen3-8b \
            qwen3-8b-t13:$R/matchT_t13/qwen3-8b qwen3-8b-t16:$R/matchT_t16/qwen3-8b \
            qwen25-7b-base-t07:$THESIS/runs/pair25_t07/qwen25-7b-base qwen25-7b-base-t1:$THESIS/runs/pair25_t1/qwen25-7b-base \
            qwen25-7b-it-t07:$THESIS/runs/pair25_t07/qwen25-7b-it qwen25-7b-it-t1:$THESIS/runs/pair25_t1/qwen25-7b-it \
            qwen25-7b-it-t13:$R/matchT_t13/qwen25-7b-it qwen25-7b-it-t16:$R/matchT_t16/qwen25-7b-it; do
  ln -sfn ${spec#*:} $R/matchT/${spec%%:*}
done
MT="matchT:qwen3-8b-base-t1>qwen3-8b-t13 matchT:qwen3-8b-base-t1>qwen3-8b-t16 matchT:qwen25-7b-base-t1>qwen25-7b-it-t13 matchT:qwen25-7b-base-t1>qwen25-7b-it-t16 \
matchT:qwen3-8b-base-t07>qwen3-8b-t16 matchT:qwen25-7b-base-t07>qwen25-7b-it-t16"
RF="fewshot_rf_t07:qwen3-8b-base>qwen3-8b fewshot_rf_t07:qwen25-7b-base>qwen25-7b-it fewshot_rf_t07:qwen25-32b-base>qwen25-32b-it \
fewshot_rf_t07:olmo3-base>olmo3-instruct fewshot_rf_t07:olmo31-32b-base>olmo31-32b-instruct fewshot_rf_t07:mistral7b-base>mistral7b-it"
RFT=qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,qwen25-32b-base,qwen25-32b-it,olmo3-base,olmo3-instruct,olmo31-32b-base,olmo31-32b-instruct,mistral7b-base,mistral7b-it
MTT=qwen3-8b-base-t07,qwen3-8b-base-t1,qwen3-8b-t07,qwen3-8b-t1,qwen3-8b-t13,qwen3-8b-t16,qwen25-7b-base-t07,qwen25-7b-base-t1,qwen25-7b-it-t07,qwen25-7b-it-t1,qwen25-7b-it-t13,qwen25-7b-it-t16
python review2.py --runs $R $C --pairs "$RF" --out $A/review2_rf.json > $A/review2_rf.log 2>&1 &
python review2.py --runs $R $C --pairs "$MT" --out $A/review2_mt.json > $A/review2_mt.log 2>&1 &
python aggregate_lists.py --runs $R $C --pairs "$RF" --out $A/aggregate_rf.json > $A/aggregate_rf.log 2>&1 &
python consequences.py --runs-dir $R/fewshot_rf_t07 --tags $RFT $C --out $A/cons_rf.json > $A/cons_rf.log 2>&1 &
python consequences.py --runs-dir $R/matchT --tags $MTT $C --out $A/cons_mt.json > $A/cons_mt.log 2>&1 &
python robustness.py --runs $R $C --spec "fewshot_rf_t07:$RFT matchT:$MTT" --out $A/robust.json > $A/robust.log 2>&1 &
wait
cat $A/*.log | cut -c1-3000
echo R4_DONE
