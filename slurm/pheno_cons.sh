#!/bin/bash
# Phenopacket stated-vs-sampled consequences (CPU). Reads runs from scratch; writes to ~/thesis_overflow (scratch over quota).
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs; P=$THESIS/data/phenopackets/cases.jsonl; N=$THESIS/data/ontology/normalizer.pkl; H=$HOME/thesis_overflow/analysis/pheno
mkdir -p $H
cd $THESIS/code/src
for spec in "go_full_pair25 qwen25-7b-base,qwen25-7b-it" "go_full_q3 qwen3-8b-base,qwen3-8b" \
            "go_full_olmo olmo3-base,olmo3-sft,olmo3-dpo,olmo3-instruct,olmo3-rlzero-general" \
            "go_full alphamed-7b,huatuo-o1-7b,m1-7b,alphamed-7b-base-rl"; do
  set -- $spec
  python $HOME/thesis_overflow/pheno_consequences.py --runs-dir $R/$1 --tags $2 --cases $P --normalizer $N --out $H/$1.json > $H/$1.log 2>&1 &
done
wait
cat $H/*.log
echo PHENO_CONS_DONE
