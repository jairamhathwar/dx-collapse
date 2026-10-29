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
  matcht)  # review control: post-trained models at higher temperatures, to compare at matched sharpness
    for t in 1.3 1.6; do
      d=matchT_t${t/./}
      run $d qwen3-8b Qwen/Qwen3-8B $DDX $STD --temperature $t --top-p 1.0
      run $d qwen25-7b-it Qwen/Qwen2.5-7B-Instruct $DDX $STD --temperature $t --top-p 1.0
    done ;;
  matcht-smoke)
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 4 --shard 0 --num-shards 1 --limit 8"
    for t in 1.3 1.6; do
      sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-matcht-q3-$t $S --model Qwen/Qwen3-8B --out runs/matcht_smoke/qwen3-8b-t$t.jsonl $SM $STD --temperature $t --top-p 1.0
      sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-matcht-q25-$t $S --model Qwen/Qwen2.5-7B-Instruct --out runs/matcht_smoke/qwen25-7b-it-t$t.jsonl $SM $STD --temperature $t --top-p 1.0
    done ;;
  fewshot-rf)  # review control: few-shot examples whose lists come from Qwen2.5-32B-Instruct, not from the reference
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl --fewshot-answers data/ddxplus/fewshot_answers_q32.json"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen25-7b-base:Qwen/Qwen2.5-7B olmo3-base:allenai/Olmo-3-1025-7B mistral7b-base:mistralai/Mistral-7B-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run fewshot_rf_t07 $tag $model $DDX $FS --completion --max-tokens 512 --max-model-len 6144
    done
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct mistral7b-it:mistralai/Mistral-7B-Instruct-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run fewshot_rf_t07 $tag $model $DDX $FS $STD
    done
    run4 fewshot_rf_t07 qwen25-32b-base Qwen/Qwen2.5-32B 04:00:00 $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144
    run4 fewshot_rf_t07 olmo31-32b-base allenai/Olmo-3-1125-32B 04:00:00 $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144
    run4 fewshot_rf_t07 qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 04:00:00 $DDX4 $FS $STD
    run4 fewshot_rf_t07 olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 06:00:00 $DDX4 $FS $STD ;;
  fewshot-rf-smoke)
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl --fewshot-answers data/ddxplus/fewshot_answers_q32.json"
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 4 --shard 0 --num-shards 1 --limit 8"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fsrf-base $S --model Qwen/Qwen2.5-7B --out runs/fewshot_rf_smoke/qwen25-7b-base.jsonl $SM $FS --completion --max-tokens 512 --max-model-len 6144
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fsrf-it $S --model Qwen/Qwen2.5-7B-Instruct --out runs/fewshot_rf_smoke/qwen25-7b-it.jsonl $SM $FS $STD ;;
  greedy)  # one greedy answer per case: post-trained models zero-shot, base models with the few-shot prompt
    G="--k 1 --temperature 0.0 --top-p 1.0"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard -1 --num-shards 4"
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct mistral7b-it:mistralai/Mistral-7B-Instruct-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run greedy $tag $model $DDX $STD $G
    done
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen25-7b-base:Qwen/Qwen2.5-7B olmo3-base:allenai/Olmo-3-1025-7B mistral7b-base:mistralai/Mistral-7B-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run greedy $tag $model $DDX $FS --completion --max-tokens 512 --max-model-len 6144 $G
    done
    run4 greedy qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 02:00:00 $DDX4 $STD $G
    run4 greedy olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 02:00:00 $DDX4 $STD $G
    run4 greedy qwen25-32b-base Qwen/Qwen2.5-32B 02:00:00 $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144 $G
    run4 greedy olmo31-32b-base allenai/Olmo-3-1125-32B 02:00:00 $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144 $G ;;
  greedy-matched)  # matched-prompt greedy: post-trained models with the base models' few-shot examples (vs greedy/ base),
                   # base models with the reference-free examples (vs greedy/ post-trained zero-shot)
    G="--k 1 --temperature 0.0 --top-p 1.0"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    RF="$FS --fewshot-answers data/ddxplus/fewshot_answers_q32.json"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard -1 --num-shards 4"
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct mistral7b-it:mistralai/Mistral-7B-Instruct-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run greedy_fs $tag $model $DDX $FS $STD $G
    done
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen25-7b-base:Qwen/Qwen2.5-7B olmo3-base:allenai/Olmo-3-1025-7B mistral7b-base:mistralai/Mistral-7B-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run greedy_rf $tag $model $DDX $RF --completion --max-tokens 512 --max-model-len 6144 $G
    done
    run4 greedy_fs qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 02:00:00 $DDX4 $FS $STD $G
    run4 greedy_fs olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 02:00:00 $DDX4 $FS $STD $G
    run4 greedy_rf qwen25-32b-base Qwen/Qwen2.5-32B 02:00:00 $DDX4 $RF --completion --max-tokens 512 --max-model-len 6144 $G
    run4 greedy_rf olmo31-32b-base allenai/Olmo-3-1125-32B 02:00:00 $DDX4 $RF --completion --max-tokens 512 --max-model-len 6144 $G ;;
  greedy-matched-smoke)  # 8 cases: the most verbose post-trained model with examples, one base model with reference-free examples
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard 0 --num-shards 1 --limit 8 --k 1 --temperature 0.0 --top-p 1.0"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-gfs-olmo $S --model allenai/Olmo-3-7B-Instruct --out runs/gm_smoke/fs_olmo_it.jsonl $SM $FS $STD
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-grf-mistral $S --model mistralai/Mistral-7B-v0.3 --out runs/gm_smoke/rf_mistral_base.jsonl $SM $FS --fewshot-answers data/ddxplus/fewshot_answers_q32.json --completion --max-tokens 512 --max-model-len 6144 ;;
  medcase)  # published case reports (MedCaseReasoning test, 897 cases), zero-shot, T=0.7, five Qwen/OLMo pairs
    MC="--task casereport --cases data/medcase/cases_test.jsonl --k 32 --shard -1 --num-shards 2"
    MC4="--task casereport --cases data/medcase/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B qwen25-7b-base:Qwen/Qwen2.5-7B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct \
                olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      run medcase_t07 $tag $model $MC $STD
    done
    run medcase_t07 olmo3-base allenai/Olmo-3-1025-7B $MC $STD --chat-template code/templates/olmo3_sft.jinja
    run4 medcase_t07 qwen25-32b-base Qwen/Qwen2.5-32B 04:00:00 $MC4 $STD
    run4 medcase_t07 qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 04:00:00 $MC4 $STD
    run4 medcase_t07 olmo31-32b-base allenai/Olmo-3-1125-32B 06:00:00 $MC4 $STD --chat-template code/templates/olmo31_32b_sft.jinja
    run4 medcase_t07 olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 06:00:00 $MC4 $STD ;;
  medcase-smoke)  # 8 cases x 4 samples: a base model and the most verbose post-trained model
    SM="--task casereport --cases data/medcase/cases_test.jsonl --k 4 --shard 0 --num-shards 1 --limit 8"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mc-olmo-base $S --model allenai/Olmo-3-1025-7B --out runs/mc_smoke/olmo3-base/samples.jsonl $SM $STD --chat-template code/templates/olmo3_sft.jinja
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mc-qwen-base $S --model Qwen/Qwen2.5-7B --out runs/mc_smoke/qwen25-7b-base/samples.jsonl $SM $STD
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mc-olmo-it $S --model allenai/Olmo-3-7B-Instruct --out runs/mc_smoke/olmo3-instruct/samples.jsonl $SM $STD ;;
  medcase-judge)  # judge every distinct (case, name) with two judges (agreement, and strict grading = both say yes)
    J=code/slurm/tillicum/judge.sbatch
    TAGS=qwen3-8b-base,qwen3-8b,qwen25-7b-base,qwen25-7b-it,qwen25-32b-base,qwen25-32b-it,olmo3-base,olmo3-instruct  # OLMo-3.1-32B base cancelled (cost)
    JA="--cases $THESIS/data/medcase/cases_test.jsonl --runs $THESIS/runs/medcase_t07 --tags $TAGS"
    # DEP (optional): a Slurm dependency, e.g. afterok:<sampling job ids>
    sbatch ${DEP:+--dependency=$DEP} --time=03:00:00 --job-name=judge-mc $J $JA --model mistralai/Mistral-Small-24B-Instruct-2501 --out $THESIS/runs/medcase_t07/judge_mistral24b.jsonl
    sbatch ${DEP:+--dependency=$DEP} --time=03:00:00 --job-name=judge2-mc $J $JA --model microsoft/phi-4 --out $THESIS/runs/medcase_t07/judge2_phi4.jsonl
    # names against the case authors' own differential (item 1 = final diagnosis, then the authors' alternatives)
    sbatch ${DEP:+--dependency=$DEP} --time=03:00:00 --job-name=judge-ddx-mc $J $JA --ddx $THESIS/runs/medcase_t07/authors_ddx.jsonl --model mistralai/Mistral-Small-24B-Instruct-2501 --out $THESIS/runs/medcase_t07/judge_ddx_mistral24b.jsonl ;;
  medcase-an)  # scoring after both judges (DEP: afterok:<judge job ids>)
    sbatch ${DEP:+--dependency=$DEP} code/slurm/tillicum/medcase_an.sbatch ;;
  medcase-judge-smoke)
    J=code/slurm/tillicum/judge.sbatch
    sbatch --time=00:30:00 --job-name=judge-mc-smoke $J --cases $THESIS/data/medcase/cases_test.jsonl --runs $THESIS/runs/mc_smoke \
      --tags olmo3-base,qwen25-7b-base,olmo3-instruct --model mistralai/Mistral-Small-24B-Instruct-2501 --out $THESIS/runs/mc_smoke/judge.jsonl ;;
  frontier)  # few-shot temperature frontier for the three 7-8B pairs, top-p fixed at 1
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    for t in 0.4 0.7 1.0 1.3; do
      d=frontier_t${t/./}
      for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen25-7b-base:Qwen/Qwen2.5-7B olmo3-base:allenai/Olmo-3-1025-7B; do
        tag=${spec%%:*}; model=${spec#*:}
        run $d $tag $model $DDX $FS --completion --max-tokens 512 --max-model-len 6144 --temperature $t --top-p 1.0
      done
      for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
        tag=${spec%%:*}; model=${spec#*:}
        run $d $tag $model $DDX $FS $STD --temperature $t --top-p 1.0
      done
    done ;;
  mnm)  # must-not-miss prompt, six final post-trained models, T=0.7
    MNM="--prompt-variant mustnotmiss"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct mistral7b-it:mistralai/Mistral-7B-Instruct-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      run mnm_t07 $tag $model $DDX $STD $MNM
    done
    run4 mnm_t07 qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 04:00:00 $DDX4 $STD $MNM
    run4 mnm_t07 olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 06:00:00 $DDX4 $STD $MNM ;;
  r7-smoke)  # greedy (base few-shot + post), frontier extremes, must-not-miss
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard 0 --num-shards 1 --limit 8"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-greedy-it $S --model Qwen/Qwen2.5-7B-Instruct --out runs/r7_smoke/greedy_it.jsonl $SM $STD --k 1 --temperature 0.0 --top-p 1.0
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-greedy-base $S --model allenai/Olmo-3-1025-7B --out runs/r7_smoke/greedy_olmo_base.jsonl $SM $FS --completion --max-tokens 512 --max-model-len 6144 --k 1 --temperature 0.0 --top-p 1.0
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-frontier $S --model allenai/Olmo-3-7B-Instruct --out runs/r7_smoke/frontier_olmo_it_t13.jsonl $SM $FS $STD --k 4 --temperature 1.3 --top-p 1.0
    for spec in qwen3-8b:Qwen/Qwen3-8B olmo3-instruct:allenai/Olmo-3-7B-Instruct mistral7b-it:mistralai/Mistral-7B-Instruct-v0.3; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mnm-$tag $S --model $model --out runs/r7_smoke/mnm_$tag.jsonl $SM $STD --k 4 --prompt-variant mustnotmiss
    done ;;
  seeded-smoke)  # recovery experiment: list then seeded pick, 8 cases x 32 numbers, one 7-8B post-trained model each
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard 0 --num-shards 1 --limit 8"
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-seed-$tag $S --model $model --out runs/seeded_smoke/$tag.jsonl $SM $STD --k 32 --prompt-variant seeded
    done ;;
  seeded)  # recovery experiment, full: five post-trained models, 980 cases x 32 seeded calls at T=0.7
    SEED="--prompt-variant seeded"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    for spec in qwen3-8b:Qwen/Qwen3-8B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      run seeded_t07 $tag $model $DDX $STD $SEED
    done
    run4 seeded_t07 qwen25-32b-it Qwen/Qwen2.5-32B-Instruct 04:00:00 $DDX4 $STD $SEED
    run4 seeded_t07 olmo31-32b-instruct allenai/Olmo-3.1-32B-Instruct 06:00:00 $DDX4 $STD $SEED ;;
  closed-smoke)  # closed-set case reports and free-length DDXPlus lists: 8 cases x 4 samples
    CS="--task casereport --cases data/medcase/cases_test.jsonl --candidates data/medcase/candidates_closed.jsonl --k 4 --shard 0 --num-shards 1 --limit 8"
    FL="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --prompt-variant freelen --k 4 --shard 0 --num-shards 1 --limit 8"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-cs-olmo-base $S --model allenai/Olmo-3-1025-7B --out runs/closed_smoke/olmo3-base.jsonl $CS $STD --chat-template code/templates/olmo3_sft.jinja
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-cs-qwen3 $S --model Qwen/Qwen3-8B --out runs/closed_smoke/qwen3-8b.jsonl $CS $STD
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fl-base $S --model Qwen/Qwen3-8B-Base --out runs/closed_smoke/fl_qwen3-8b-base.jsonl $FL $STD
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fl-post $S --model Qwen/Qwen3-8B --out runs/closed_smoke/fl_qwen3-8b.jsonl $FL $STD ;;
  closed)  # closed-set case reports (897 cases, 10 candidates each), zero-shot, T=0.7, the four case-report pairs
    CS="--task casereport --cases data/medcase/cases_test.jsonl --candidates data/medcase/candidates_closed.jsonl --k 32"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B qwen25-7b-base:Qwen/Qwen2.5-7B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct \
                olmo3-instruct:allenai/Olmo-3-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=${CT:-00:45:00} --array=0-1 --job-name=m-closed-$tag $S --model $model --out runs/medcase_closed/$tag/samples.s{shard}.jsonl $CS --shard -1 --num-shards 2 $STD
    done
    sbatch -p gpu-h200 --time=${CTB:-02:00:00} --array=0-1 --job-name=m-closed-olmo3-base $S --model allenai/Olmo-3-1025-7B --out runs/medcase_closed/olmo3-base/samples.s{shard}.jsonl $CS --shard -1 --num-shards 2 $STD --chat-template code/templates/olmo3_sft.jinja --max-tokens 1024
    for spec in qwen25-32b-base:Qwen/Qwen2.5-32B qwen25-32b-it:Qwen/Qwen2.5-32B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=${C32:-01:15:00} --mem=96G --array=0-3 --job-name=m-closed-$tag $S --model $model --out runs/medcase_closed/$tag/samples.s{shard}.jsonl $CS --shard -1 --num-shards 4 $STD
    done ;;
  single-smoke)  # single-answer prompt (one diagnosis per call), the two 7-8B Qwen pairs: 8 cases x 4 samples
    SG="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --prompt-variant single --k 4 --shard 0 --num-shards 1 --limit 8"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B qwen25-7b-base:Qwen/Qwen2.5-7B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-sg-$tag $S --model $model --out runs/single_smoke/$tag.jsonl $SG $STD
    done ;;
  single)  # single-answer prompt, zero-shot, T=0.7, the two 7-8B Qwen pairs
    SG="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --prompt-variant single --k 32"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B qwen25-7b-base:Qwen/Qwen2.5-7B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=${ST:-00:40:00} --array=0-1 --job-name=m-single-$tag $S --model $model --out runs/single_t07/$tag/samples.s{shard}.jsonl $SG --shard -1 --num-shards 2 $STD
    done ;;
  freelen)  # free-length DDXPlus lists, zero-shot, T=0.7, the three Qwen pairs
    FL="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --prompt-variant freelen --k 32"
    for spec in qwen3-8b-base:Qwen/Qwen3-8B-Base qwen3-8b:Qwen/Qwen3-8B qwen25-7b-base:Qwen/Qwen2.5-7B qwen25-7b-it:Qwen/Qwen2.5-7B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=${FT:-00:40:00} --array=0-1 --job-name=m-freelen-$tag $S --model $model --out runs/freelen_t07/$tag/samples.s{shard}.jsonl $FL --shard -1 --num-shards 2 $STD
    done
    for spec in qwen25-32b-base:Qwen/Qwen2.5-32B qwen25-32b-it:Qwen/Qwen2.5-32B-Instruct; do
      tag=${spec%%:*}; model=${spec#*:}
      sbatch -p gpu-h200 --time=${F32:-01:00:00} --mem=96G --array=0-3 --job-name=m-freelen-$tag $S --model $model --out runs/freelen_t07/$tag/samples.s{shard}.jsonl $FL --shard -1 --num-shards 4 $STD
    done ;;
  freelen-olmo-smoke)  # free-length lists for the OLMo bases (1,024-token cap): 8 cases x 4 samples
    FL="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --prompt-variant freelen --k 4 --shard 0 --num-shards 1 --limit 8"
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-fl-olmo3-base $S --model allenai/Olmo-3-1025-7B --out runs/freelen_smoke/olmo3-base.jsonl $FL $STD --max-tokens 1024 --chat-template code/templates/olmo3_sft.jinja
    sbatch -p gpu-h200 --time=00:30:00 --mem=96G --job-name=smoke-fl-olmo31-base $S --model allenai/Olmo-3-1125-32B --out runs/freelen_smoke/olmo31-32b-base.jsonl $FL $STD --max-tokens 1024 --chat-template code/templates/olmo31_32b_sft.jinja ;;
  freelen-olmo)  # free-length lists, zero-shot, T=0.7, the two OLMo pairs (bases capped at 1,024 tokens)
    FL="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --prompt-variant freelen --k 32"
    sbatch -p gpu-h200 --time=${FB:-01:30:00} --array=0-1 --job-name=m-freelen-olmo3-base $S --model allenai/Olmo-3-1025-7B --out runs/freelen_t07/olmo3-base/samples.s{shard}.jsonl $FL --shard -1 --num-shards 2 $STD --max-tokens 1024 --chat-template code/templates/olmo3_sft.jinja
    sbatch -p gpu-h200 --time=${FI:-02:30:00} --array=0-1 --job-name=m-freelen-olmo3-instruct $S --model allenai/Olmo-3-7B-Instruct --out runs/freelen_t07/olmo3-instruct/samples.s{shard}.jsonl $FL --shard -1 --num-shards 2 $STD
    sbatch -p gpu-h200 --time=${FB32:-01:30:00} --mem=96G --array=0-3 --job-name=m-freelen-olmo31-32b-base $S --model allenai/Olmo-3-1125-32B --out runs/freelen_t07/olmo31-32b-base/samples.s{shard}.jsonl $FL --shard -1 --num-shards 4 $STD --max-tokens 1024 --chat-template code/templates/olmo31_32b_sft.jinja
    sbatch -p gpu-h200 --time=${FI32:-01:00:00} --mem=96G --array=0-3 --job-name=m-freelen-olmo31-32b-instruct $S --model allenai/Olmo-3.1-32B-Instruct --out runs/freelen_t07/olmo31-32b-instruct/samples.s{shard}.jsonl $FL --shard -1 --num-shards 4 $STD ;;
  frontier32)  # few-shot temperature frontier for the two 32B pairs, top-p fixed at 1 (as for the 7-8B frontier)
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    DDX4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4"
    for t in 0.4 0.7 1.0 1.3; do
      d=frontier_t${t/./}
      for spec in qwen25-32b-base:Qwen/Qwen2.5-32B olmo31-32b-base:allenai/Olmo-3-1125-32B; do
        tag=${spec%%:*}; model=${spec#*:}
        sbatch -p gpu-h200 --time=${TB:-00:40:00} --mem=96G --array=0-3 --job-name=m-$d-$tag $S --model $model --out runs/$d/$tag/samples.s{shard}.jsonl $DDX4 $FS --completion --max-tokens 512 --max-model-len 6144 --temperature $t --top-p 1.0
      done
      for spec in qwen25-32b-it:Qwen/Qwen2.5-32B-Instruct olmo31-32b-instruct:allenai/Olmo-3.1-32B-Instruct; do
        tag=${spec%%:*}; model=${spec#*:}
        sbatch -p gpu-h200 --time=${TI:-00:50:00} --mem=96G --array=0-3 --job-name=m-$d-$tag $S --model $model --out runs/$d/$tag/samples.s{shard}.jsonl $DDX4 $FS $STD --temperature $t --top-p 1.0
      done
    done ;;
  big72-smoke)  # Qwen2.5-72B pair on two H200s: zero-shot both, worked examples for the base; 8 cases x 4 samples
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard 0 --num-shards 1 --limit 8 --k 4 --tp 2 --enforce-eager"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    sbatch -p gpu-h200 --nodes=1 --gpus=2 --cpus-per-task=4 --mem=160G --export=ALL,VLLM_ALLREDUCE_USE_FLASHINFER=0,VLLM_USE_FLASHINFER_SAMPLER=0 --time=00:40:00 --job-name=smoke-72b-it $S --model Qwen/Qwen2.5-72B-Instruct --out runs/big72_smoke/it.jsonl $SM $STD
    sbatch -p gpu-h200 --nodes=1 --gpus=2 --cpus-per-task=4 --mem=160G --export=ALL,VLLM_ALLREDUCE_USE_FLASHINFER=0,VLLM_USE_FLASHINFER_SAMPLER=0 --time=00:40:00 --job-name=smoke-72b-base $S --model Qwen/Qwen2.5-72B --out runs/big72_smoke/base.jsonl $SM $STD
    sbatch -p gpu-h200 --nodes=1 --gpus=2 --cpus-per-task=4 --mem=160G --export=ALL,VLLM_ALLREDUCE_USE_FLASHINFER=0,VLLM_USE_FLASHINFER_SAMPLER=0 --time=00:40:00 --job-name=smoke-72b-basefs $S --model Qwen/Qwen2.5-72B --out runs/big72_smoke/base_fs.jsonl $SM $FS --completion --max-tokens 512 --max-model-len 6144 ;;
  medgemma-smoke)  # MedGemma-4B pt/it (medical-domain pair), DDXPlus T=0.7: zero-shot both, worked examples for the base; 8 cases x 4 samples
    SM="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --shard 0 --num-shards 1 --limit 8 --k 4"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    MGT=${MGT:+--chat-template $MGT}  # the pt checkpoint uses the it chat template if it ships none
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mg-it $S --model google/medgemma-4b-it --out runs/medgemma_smoke/it.jsonl $SM $STD
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mg-pt $S --model google/medgemma-4b-pt --out runs/medgemma_smoke/pt.jsonl $SM $STD $MGT
    sbatch -p gpu-h200 --time=00:30:00 --job-name=smoke-mg-ptfs $S --model google/medgemma-4b-pt --out runs/medgemma_smoke/pt_fs.jsonl $SM $FS --completion --max-tokens 512 --max-model-len 6144 ;;
  medgemma)  # MedGemma-4B pt/it, DDXPlus T=0.7, 32 samples: zero-shot (both), worked examples (both)
    D2="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 2"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    MGT=${MGT:+--chat-template $MGT}
    sbatch -p gpu-h200 --time=${MT:-00:40:00} --array=0-1 --job-name=m-mg-it $S --model google/medgemma-4b-it --out runs/medgemma_ddx/medgemma-4b-it/samples.s{shard}.jsonl $D2 $STD
    sbatch -p gpu-h200 --time=${MT:-00:40:00} --array=0-1 --job-name=m-mg-pt $S --model google/medgemma-4b-pt --out runs/medgemma_ddx/medgemma-4b-pt/samples.s{shard}.jsonl $D2 $STD $MGT
    sbatch -p gpu-h200 --time=${MT:-00:40:00} --array=0-1 --job-name=m-mgfs-it $S --model google/medgemma-4b-it --out runs/fewshotmg_t07/medgemma-4b-it/samples.s{shard}.jsonl $D2 $FS $STD
    sbatch -p gpu-h200 --time=${MT:-00:40:00} --array=0-1 --job-name=m-mgfs-pt $S --model google/medgemma-4b-pt --out runs/fewshotmg_t07/medgemma-4b-pt/samples.s{shard}.jsonl $D2 $FS --completion --max-tokens 512 --max-model-len 6144 ;;
  big72)  # Qwen2.5-72B pair, DDXPlus T=0.7, 32 samples: zero-shot (both), worked examples (both), two H200s per shard
    D4="--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl --k 32 --shard -1 --num-shards 4 --tp 2 --enforce-eager"
    FS="--prompt-variant fewshot --fewshot-train data/ddxplus/cases_train.jsonl"
    G="-p gpu-h200 --nodes=1 --gpus=2 --cpus-per-task=4 --mem=160G --export=ALL,VLLM_ALLREDUCE_USE_FLASHINFER=0,VLLM_USE_FLASHINFER_SAMPLER=0 --array=0-3"
    sbatch $G --time=${T72Z:-01:30:00} --job-name=m-big72-qwen25-72b-it $S --model Qwen/Qwen2.5-72B-Instruct --out runs/big72_ddx/qwen25-72b-it/samples.s{shard}.jsonl $D4 $STD
    sbatch $G --time=${T72Z:-01:30:00} --job-name=m-big72-qwen25-72b-base $S --model Qwen/Qwen2.5-72B --out runs/big72_ddx/qwen25-72b-base/samples.s{shard}.jsonl $D4 $STD
    sbatch $G --time=${T72F:-01:00:00} --job-name=m-fs72-qwen25-72b-it $S --model Qwen/Qwen2.5-72B-Instruct --out runs/fewshot72_t07/qwen25-72b-it/samples.s{shard}.jsonl $D4 $FS $STD
    sbatch $G --time=${T72F:-01:00:00} --job-name=m-fs72-qwen25-72b-base $S --model Qwen/Qwen2.5-72B --out runs/fewshot72_t07/qwen25-72b-base/samples.s{shard}.jsonl $D4 $FS --completion --max-tokens 512 --max-model-len 6144 ;;
  *) echo "usage: $0 descendants|pair|prompts|fullpheno|olmo|full49|big32|fewshot|fewshot-smoke|mistral|mistral-smoke|think|think-smoke|matcht|matcht-smoke|fewshot-rf|fewshot-rf-smoke|greedy|greedy-matched|greedy-matched-smoke|medcase|medcase-smoke|medcase-judge|medcase-judge-smoke|medcase-an|frontier|mnm|r7-smoke|seeded|seeded-smoke|closed|closed-smoke|freelen"; exit 1 ;;
esac
