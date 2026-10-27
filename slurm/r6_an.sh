#!/bin/bash
# Fifth review round (CPU): scale-free allocation error, recall@k, severe-shift magnitude test, best-elicitation
# comparison, clustered within-model intervals for few-shot and thinking models. Code from the home clone.
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
D=$THESIS/data/ddxplus; H=$HOME/thesis_overflow; R=$THESIS/runs; A=$H/analysis/r6; F=$H/runs/fewshot_t07; RF=$H/runs/fewshot_rf_t07; B=$H/runs/bestelic
mkdir -p $A $B
cd $HOME/thesis_repo/src
C="--cases $D/cases_test.jsonl --conditions-json $D/release_conditions.json"
# best elicitation: base with worked examples; post-trained at whichever prompt gives the higher pass@32
for spec in qwen3-8b-base:$F/qwen3-8b-base qwen3-8b:$F/qwen3-8b qwen25-7b-base:$F/qwen25-7b-base qwen25-7b-it:$R/pair25_t07/qwen25-7b-it \
            qwen25-32b-base:$F/qwen25-32b-base qwen25-32b-it:$R/big32_ddx/qwen25-32b-it olmo3-base:$F/olmo3-base olmo3-instruct:$F/olmo3-instruct \
            olmo31-32b-base:$F/olmo31-32b-base olmo31-32b-instruct:$R/big32_ddx/olmo31-32b-instruct mistral7b-base:$F/mistral7b-base mistral7b-it:$F/mistral7b-it; do
  ln -sfn ${spec#*:} $B/${spec%%:*}
done
ZS="pair_q3:qwen3-8b-base>qwen3-8b ddx_qwen3_t1:qwen3-8b-base>qwen3-8b pair25_t07:qwen25-7b-base>qwen25-7b-it pair25_t1:qwen25-7b-base>qwen25-7b-it \
big32_ddx:qwen25-32b-base>qwen25-32b-it ddx_olmo:olmo3-base>olmo3-instruct ddx_olmo_t1:olmo3-base>olmo3-instruct big32_ddx:olmo31-32b-base>olmo31-32b-instruct"
FS="$F:qwen3-8b-base>qwen3-8b $F:qwen25-7b-base>qwen25-7b-it $F:qwen25-32b-base>qwen25-32b-it $F:olmo3-base>olmo3-instruct $F:olmo31-32b-base>olmo31-32b-instruct"
TH="$F:mistral7b-base>mistral7b-it $H/runs/think_t07:qwen3-8b-base>qwen3-8b-think $H/runs/think_t07:qwen3-8b>qwen3-8b-think"
RFP="$RF:qwen3-8b-base>qwen3-8b $RF:qwen25-7b-base>qwen25-7b-it $RF:qwen25-32b-base>qwen25-32b-it $RF:olmo3-base>olmo3-instruct $RF:olmo31-32b-base>olmo31-32b-instruct $RF:mistral7b-base>mistral7b-it"
BE="$B:qwen3-8b-base>qwen3-8b $B:qwen25-7b-base>qwen25-7b-it $B:qwen25-32b-base>qwen25-32b-it $B:olmo3-base>olmo3-instruct $B:olmo31-32b-base>olmo31-32b-instruct $B:mistral7b-base>mistral7b-it"
python review2.py --runs $R $C --pairs "$ZS" --out $A/review2_zs.json > $A/review2_zs.log 2>&1 &
python review2.py --runs $R $C --pairs "$FS" --out $A/review2_fs.json > $A/review2_fs.log 2>&1 &
python review2.py --runs $R $C --pairs "$TH" --out $A/review2_third.json > $A/review2_third.log 2>&1 &
python review2.py --runs $R $C --pairs "$RFP" --out $A/review2_rf.json > $A/review2_rf.log 2>&1 &
python review2.py --runs $R $C --pairs "$BE" --out $A/review2_best.json > $A/review2_best.log 2>&1 &
python cluster_within.py --runs $R $C --items "$F:qwen3-8b,qwen25-7b-it,qwen25-32b-it,olmo3-instruct,olmo31-32b-instruct,mistral7b-it $H/runs/think_t07:qwen3-8b-think" --out $A/within.json > $A/within.log 2>&1 &
wait
cat $A/*.log | cut -c1-2500
echo R6_DONE
