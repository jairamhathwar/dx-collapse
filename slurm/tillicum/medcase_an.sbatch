#!/bin/bash
# MedCaseReasoning scoring (CPU work on a MIG slice): both judges, lenient (Mistral-Small judge) and strict (both judges).
#SBATCH --account=stf
#SBATCH --partition=gpu-h200-mig
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/stf/jairamh/thesis/logs/%x-%j.out
source ~/.thesis_env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$THESIS/envs/thesis"
R=$THESIS/runs/medcase_t07; A=$THESIS/analysis/medcase
mkdir -p $A
cd $THESIS/code/src
P="qwen3-8b-base>qwen3-8b qwen25-7b-base>qwen25-7b-it qwen25-32b-base>qwen25-32b-it olmo3-base>olmo3-instruct olmo31-32b-base>olmo31-32b-instruct"
C="--cases $THESIS/data/medcase/cases_test.jsonl --runs $R --judge $R/judge_mistral24b.jsonl --judge2 $R/judge2_qwen32b.jsonl --pairs"
python medcase_analysis.py $C "$P" --out $A/medcase.json > $A/medcase.log 2>&1
python medcase_analysis.py $C "$P" --strict --out $A/medcase_strict.json > $A/medcase_strict.log 2>&1
tail -n 20 $A/*.log | cut -c1-1500
echo MEDCASE_DONE
