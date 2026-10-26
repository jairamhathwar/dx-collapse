#!/bin/bash
# Tillicum: second model family (Llama-3.1-8B) plus AlphaMed RL-from-base (Qwen2.5-7B).
# Same prompts, sampling and token budgets as submit_measurement.sh. Outputs under runs/meas_*/<tag>/.
#   bash slurm/tillicum/submit_llama.sh descendants   # ungated medical descendants (run now)
#   bash slurm/tillicum/submit_llama.sh pair          # Llama-3.1-8B base + Instruct (needs Meta gate approval)
#   bash slurm/tillicum/submit_llama.sh prompts       # prompt-robustness variants on the Qwen3-8B pair
#   bash slurm/tillicum/submit_llama.sh fullpheno     # remaining 901 Phenopacket cases (eval_full minus eval_go)
#   bash slurm/tillicum/submit_llama.sh olmo          # OLMo 3 7B post-training stages (base, SFT, DPO, RLVR, RL-Zero)
#   bash slurm/tillicum/submit_llama.sh full49        # full 49-condition distribution prompt, 7B pairs + OLMo base/Instruct (T=0.7)
#   bash slurm/tillicum/submit_llama.sh big32         # 32B replication: Qwen2.5-32B pair + OLMo 3.1 32B stages (T=0.7)
set -euo pipefail
source ~/.thesis_env
cd "$THESIS"
S=code/slurm/tillicum/sample.sbatch
DDX="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 2"
PHENO="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_go.txt --k 32 --shard -1 --num-shards 2"
STD="--max-tokens 4096 --max-model-len 6144"
T1="--temperature 1.0 --top-p 1.0"

run4() {  # dir tag model time args...   (4 shards, larger memory; for 32B models)
  local dir=$1 tag=$2 model=$3 t=$4; shift 4
  j=$(sbatch --parsable -p gpu-h200 --time=$t --mem=96G --array=0-3 --job-name=m-$dir-$tag $S \
      --model $model --out runs/$dir/$tag/samples.s{shard}.jsonl "$@")
  echo "$dir/$tag  $model  job=$j"
}

run() {  # dir tag model args...
  local dir=$1 tag=$2 model=$3; shift 3
  j=$(sbatch --parsable -p gpu-h200 --time=04:00:00 --array=0-1 --job-name=m-$dir-$tag $S \
      --model $model --out runs/$dir/$tag/samples.s{shard}.jsonl "$@")
  echo "$dir/$tag  $model  job=$j"
}
case "${1:-}" in
  descendants)
    for spec in alphamed-7b-base-rl:che111/AlphaMed-7B-base-rl \
                alphamed-8b:che111/AlphaMed-8B-instruct-rl \
                huatuo-o1-8b:FreedomIntelligence/HuatuoGPT-o1-8B \
                ultramedical-8b:TsinghuaC3I/Llama-3.1-8B-UltraMedical; do
      tag=${spec%%:*}; model=${spec#*:}
      run meas_ddx $tag $model $DDX $STD
      run meas_go $tag $model $PHENO $STD
    done ;;
  pair)
    for spec in llama31-8b-base:meta-llama/Llama-3.1-8B llama31-8b-it:meta-llama/Llama-3.1-8B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      run meas_ddx $tag $model $DDX $STD
      run meas_ddx_t1 $tag $model $DDX $STD $T1
      run meas_go $tag $model $PHENO $STD
    done ;;
  prompts)
    for v in plain cot; do
      for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B; do
        tag=${spec%%:*}; model=${spec#*:}
        run pv_${v}_t07 $tag $model $DDX $STD --prompt-variant $v
        run pv_${v}_t1 $tag $model $DDX $STD $T1 --prompt-variant $v
      done
    done ;;
  fullpheno)
    comm -23 <(sort data/phenopackets/splits/eval_full.txt) <(sort data/phenopackets/splits/eval_go.txt) > data/phenopackets/splits/eval_rest.txt
    echo "eval_rest: $(wc -l < data/phenopackets/splits/eval_rest.txt) cases"
    REST="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_rest.txt --k 32 --shard -1 --num-shards 2"
    for spec in qwen25-7b-base:Qwen/Qwen2.5-7B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct alphamed-7b:che111/AlphaMed-7B-instruct-rl \
                huatuo-o1-7b:FreedomIntelligence/HuatuoGPT-o1-7B alphamed-7b-base-rl:che111/AlphaMed-7B-base-rl \
                qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B; do
      tag=${spec%%:*}; model=${spec#*:}
      run meas_rest $tag $model $REST $STD
    done
    run meas_rest m1-7b UCSC-VLAA/m1-7B-23K $REST --max-tokens 8192 --max-model-len 10240 ;;
  olmo)
    FULL="--cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_full.txt --k 32 --shard -1 --num-shards 2"
    for spec in olmo3-base:allenai/Olmo-3-1025-7B olmo3-sft:allenai/Olmo-3-7B-Instruct-SFT olmo3-dpo:allenai/Olmo-3-7B-Instruct-DPO \
                olmo3-instruct:allenai/Olmo-3-7B-Instruct olmo3-rlzero-general:allenai/Olmo-3-7B-RL-Zero-General \
                olmo3-rlzero-mix:allenai/Olmo-3-7B-RL-Zero-Mix; do
      tag=${spec%%:*}; model=${spec#*:}
      tpl=""; [ "$tag" = olmo3-base ] && tpl="--chat-template code/templates/olmo3_sft.jinja"
      run meas_ddx $tag $model $DDX $STD $tpl
      run meas_ddx_t1 $tag $model $DDX $STD $T1 $tpl
      run meas_full $tag $model $FULL $STD $tpl
    done ;;
  full49)
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B qwen25-7b-base:Qwen/Qwen2.5-7B \
                qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-base:allenai/Olmo-3-1025-7B olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      tpl=""; [ "$tag" = olmo3-base ] && tpl="--chat-template code/templates/olmo3_sft.jinja"
      run pv_full_t07 $tag $model $DDX $STD --prompt-variant full $tpl
    done ;;
  big32)
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    run4 big32_ddx qwen25-32b-base Qwen/Qwen2.5-32B 06:00:00 $DDX4 $STD
    run4 big32_ddx qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 06:00:00 $DDX4 $STD
    run4 big32_ddx olmo31-32b-base allenai/Olmo-3-1125-32B 12:00:00 $DDX4 $STD --chat-template code/templates/olmo31_32b_sft.jinja
    run4 big32_ddx olmo31-32b-sft allenai/Olmo-3.1-32B-Instruct-SFT 08:00:00 $DDX4 $STD --dtype bfloat16
    run4 big32_ddx olmo31-32b-dpo allenai/Olmo-3.1-32B-Instruct-DPO 08:00:00 $DDX4 $STD --dtype bfloat16
    run4 big32_ddx olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 06:00:00 $DDX4 $STD ;;
  big32fix)  # SFT and DPO checkpoints are stored in float32; load as bfloat16 like the other stages
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    run4 big32_ddx olmo31-32b-sft allenai/Olmo-3.1-32B-Instruct-SFT 08:00:00 $DDX4 $STD --dtype bfloat16
    run4 big32_ddx olmo31-32b-dpo allenai/Olmo-3.1-32B-Instruct-DPO 08:00:00 $DDX4 $STD --dtype bfloat16 ;;
  fewshot)  # review control: few-shot base elicitation (plain completion) and the same examples for post-trained models (chat)
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen25-7b-base:Qwen/Qwen2.5-7B olmo3-base:allenai/Olmo-3-1025-7B; do
      tag=${spec%%:*}; model=${spec#*:}
      run fewshot_t07 $tag $model $DDX $FS --completion --max-tokens 512 --max-model-len 6144
    done
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      run fewshot_t07 $tag $model $DDX $FS $STD
    done
    run4 fewshot_t07 qwen25-32b-base Qwen/Qwen2.5-32B 04:00:00 $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144
    run4 fewshot_t07 olmo31-32b-base allenai/Olmo-3-1125-32B 04:00:00 $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144
    run4 fewshot_t07 qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 04:00:00 $DDX4 $FS $STD
    run4 fewshot_t07 olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 06:00:00 $DDX4 $FS $STD ;;
  fewshot-smoke)  # 8 cases x 4 samples per mode, to check the prompts before the full run
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 4 --shard 0 --num-shards 1 --limit 8"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fewshot-base $S --model Qwen/Qwen2.5-7B --out runs/fewshot_smoke/qwen25-7b-base.jsonl $SM $FS --completion --max-tokens 512 --max-model-len 6144
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fewshot-it $S --model Qwen/Qwen2.5-7B-Instruct --out runs/fewshot_smoke/qwen25-7b-it.jsonl $SM $FS $STD ;;
  mistral)  # third family: Mistral-7B v0.3 base + Instruct; zero-shot at both temperatures and few-shot at T=0.7
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    TPL="--chat-template code/templates/mistral_v03_instruct.jinja"  # the base model ships without a chat template
    run meas_ddx mistral7b-base mistralai/Mistral-7B-v0.3 $DDX $STD $TPL
    run meas_ddx_t1 mistral7b-base mistralai/Mistral-7B-v0.3 $DDX $STD $T1 $TPL
    run fewshot_t07 mistral7b-base mistralai/Mistral-7B-v0.3 $DDX $FS --completion --max-tokens 512 --max-model-len 6144
    run meas_ddx mistral7b-it mistralai/Mistral-7B-Instruct-v0.3 $DDX $STD
    run meas_ddx_t1 mistral7b-it mistralai/Mistral-7B-Instruct-v0.3 $DDX $STD $T1
    run fewshot_t07 mistral7b-it mistralai/Mistral-7B-Instruct-v0.3 $DDX $FS $STD ;;
  mistral-smoke)  # 8 cases x 4 samples per mode
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    TPL="--chat-template code/templates/mistral_v03_instruct.jinja"
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 4 --shard 0 --num-shards 1 --limit 8"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mistral-base $S --model mistralai/Mistral-7B-v0.3 --out runs/mistral_smoke/base_zs.jsonl $SM $STD $TPL
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mistral-base-fs $S --model mistralai/Mistral-7B-v0.3 --out runs/mistral_smoke/base_fs.jsonl $SM $FS --completion --max-tokens 512 --max-model-len 6144
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mistral-it $S --model mistralai/Mistral-7B-Instruct-v0.3 --out runs/mistral_smoke/it_zs.jsonl $SM $STD
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mistral-it-fs $S --model mistralai/Mistral-7B-Instruct-v0.3 --out runs/mistral_smoke/it_fs.jsonl $SM $FS $STD ;;
  think)  # Qwen3-8B with thinking enabled, default prompt, T=0.7
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    run4 think_t07 qwen3-8b-think Qwen/Qwen3-8B 05:00:00 $DDX4 --thinking --max-tokens 8192 --max-model-len 10240 ;;
  think-smoke)  # 20 cases x 4 samples, to measure trace length
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 4 --shard 0 --num-shards 1 --limit 20"
    sbatch -p gpu-h200 --time=01:00:00 --job-name=smoke-think $S --model Qwen/Qwen3-8B --out runs/think_smoke/qwen3-8b-think.jsonl $SM --thinking --max-tokens 8192 --max-model-len 10240 ;;
  *) echo "usage: $0 descendants|pair|prompts|fullpheno|olmo|full49|big32|fewshot|fewshot-smoke|mistral|mistral-smoke|think|think-smoke"; exit 1 ;;
esac
