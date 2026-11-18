#!/bin/bash
# Pilots B and C on the 1B OLMo 2 model (full fine-tuning; outputs on scratch storage, not the 500 GB project quota).
# B (stage arithmetic): S = SFT(base), D = DPO(base), SD = DPO(S), two seeds. Is base + (SD - S) the same model as D,
#    and is S + D - base the same as SD? Distances in units of the seed-to-seed distance (kl_pair.py).
# C (DPO pairing): from the public SFT checkpoint, DPO with real pairs, rejections shuffled across prompts, a fixed
#    pool of 100 rejections, flipped pairs, and SFT on the chosen responses; held-out preference accuracy (prefacc).
# Worst case about $16.
source ~/.thesis_env
cd "$THESIS"
O=/gpfs/scrubbed/jairamh/thesis_runs/stage; mkdir -p $O logs
BASE=allenai/OLMo-2-0425-1B; SFTPUB=allenai/OLMo-2-0425-1B-SFT; DPOPUB=allenai/OLMo-2-0425-1B-DPO
E="source ~/.thesis_env; source \$(conda info --base)/etc/profile.d/conda.sh; conda activate $THESIS/envs/thesis; cd $THESIS/code/src"
run() { name=$1; dep=$2; shift 2
  sbatch --parsable -p gpu-h200 --gpus=1 -c 4 --mem=64G --time=01:00:00 ${dep:+--dependency=afterok:$dep} -J st-$name -o logs/st-$name-%j.out --wrap "$E; python stage_train.py $* --out $O/$name" 2>/dev/null; }
depB=""
for s in 0 1; do
  js=$(run B-S-s$s "" sft --model $BASE --seed $s); jd=$(run B-D-s$s "" dpo --model $BASE --seed $s)
  jsd=$(run B-SD-s$s $js dpo --model $O/B-S-s$s --seed $s); depB="$depB:$js:$jd:$jsd"
done
sbatch --parsable -p gpu-h200 --gpus=1 -c 4 --mem=64G --time=00:45:00 --dependency=afterok$depB -J st-B-eval -o logs/st-B-eval-%j.out --wrap "$E; set -e; for s in 0 1; do python stage_train.py arith --out $O/B-arithD-s\$s --terms \"+$BASE +$O/B-SD-s\$s -$O/B-S-s\$s\"; python stage_train.py arith --out $O/B-arithSD-s\$s --terms \"+$O/B-S-s\$s +$O/B-D-s\$s -$BASE\"; done; python kl_pair.py --tokenizer $SFTPUB --prompts $THESIS/data/ambigqa/cases_test.jsonl --out $THESIS/analysis/pilot/pB_kl.json base=$BASE S0=$O/B-S-s0 S1=$O/B-S-s1 D0=$O/B-D-s0 D1=$O/B-D-s1 SD0=$O/B-SD-s0 SD1=$O/B-SD-s1 aD0=$O/B-arithD-s0 aD1=$O/B-arithD-s1 aSD0=$O/B-arithSD-s0 aSD1=$O/B-arithSD-s1 --pairs D0:D1 D1:D0 D0:aD0 D1:aD1 D0:base D0:SD0 D0:S0 SD0:SD1 SD1:SD0 SD0:aSD0 SD1:aSD1 SD0:S0 SD0:base S0:S1" 2>/dev/null
depC=""; names=""
for spec in "real-s0|--pairing real --seed 0" "real-s1|--pairing real --seed 1" "shuf-s0|--pairing shuffled --seed 0" "shuf-s1|--pairing shuffled --seed 1" \
            "pool-s0|--pairing pool --seed 0" "pool-s1|--pairing pool --seed 1" "flip-s0|--pairing flipped --seed 0"; do
  IFS="|" read -r tag args <<< "$spec"; j=$(run C-$tag "" dpo --model $SFTPUB $args); depC="$depC:$j"; names="$names $O/C-$tag"
done
j=$(run C-sftchosen-s0 "" sft-chosen --model $SFTPUB --seed 0); depC="$depC:$j"; names="$names $O/C-sftchosen-s0"
sbatch --parsable -p gpu-h200 --gpus=1 -c 4 --mem=64G --time=00:45:00 --dependency=afterany$depC -J st-C-eval -o logs/st-C-eval-%j.out --wrap "$E; for m in $SFTPUB $DPOPUB $names; do python stage_train.py prefacc --model \$m --n 1000; done" 2>/dev/null
