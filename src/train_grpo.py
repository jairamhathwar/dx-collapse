"""GRPO with differential-level rewards on DDXPlus (LoRA, vLLM colocated). Resumable.

    python train_grpo.py --model Qwen/Qwen3-1.7B --rewards acc \
        --train $THESIS/data/ddxplus/cases_train.jsonl --conditions $THESIS/data/ddxplus/conditions.json \
        --out $THESIS/runs/train/q3-1.7b-acc [--max-steps 300] [--dry-run]

Reward recipes (--rewards, comma-separated, equal weights unless --weights given):
    acc                 standard RLVR baseline (top-1 correctness)
    acc,brier           + proper scoring rule over the stated differential (calibration)
    acc,brier,overlap   + overlap with the ground-truth differential (coverage)
A 'format' reward is always added. Each reward is logged separately by TRL.
On preemption, resubmitting the same command resumes from the last checkpoint in --out.
When training finishes, the LoRA adapter is merged and saved to --out/merged for evaluation.
"""
import argparse
import json
import random
from pathlib import Path

from rewards import make_trl_reward
from sample_diagnoses import DDX_SYSTEM, build_ddx_prompt


def load_dataset_rows(path, conditions, tokenizer, n_dx, limit=None, seed=0):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    if limit:
        rows = rows[:limit]
    out = []
    for c in rows:
        msgs = [{"role": "system", "content": DDX_SYSTEM},
                {"role": "user", "content": build_ddx_prompt(c, n_dx, conditions)}]
        # pre-render the chat prompt so thinking mode is controlled here, not by the trainer
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
        out.append({"prompt": prompt, "gold": c["gold"],
                    "gt_differential": json.dumps(c["gt_differential"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--rewards", default="acc")
    ap.add_argument("--weights", help="comma-separated weights matching format + --rewards")
    ap.add_argument("--train", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dx", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--vllm-mem", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="3 steps on 32 prompts, no merge")
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="CPU-only end-to-end check (tiny model, no vLLM, 2 steps, incl. save/resume/merge)")
    a = ap.parse_args()
    if a.cpu_smoke:
        a.model, a.num_generations, a.prompts_per_step = "Qwen/Qwen3-0.6B", 2, 2
        a.max_completion_length, a.max_steps, a.save_steps, a.lora_r = 48, 2, 1, 8

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    conditions = json.load(open(a.conditions))
    tok = AutoTokenizer.from_pretrained(a.model)
    limit = 4 if a.cpu_smoke else 32 if a.dry_run else None
    rows = load_dataset_rows(a.train, conditions, tok, a.n_dx, limit=limit, seed=a.seed)
    ds = Dataset.from_list(rows)

    names = ["format"] + [r for r in a.rewards.split(",") if r != "format"]
    funcs = [make_trl_reward(n, conditions) for n in names]
    weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(funcs)
    assert len(weights) == len(funcs), f"need {len(funcs)} weights for {names}"

    per_device = a.num_generations * 2  # completions per micro-batch (2 prompts)
    accum = max(1, a.prompts_per_step * a.num_generations // per_device)
    gpu = not a.cpu_smoke
    cfg = GRPOConfig(
        use_cpu=not gpu,
        output_dir=a.out,
        max_steps=3 if a.dry_run else a.max_steps,
        learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0 if a.dry_run else 10,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        num_generations=a.num_generations,
        max_completion_length=a.max_completion_length,
        temperature=a.temperature,
        beta=a.beta,
        reward_weights=weights,
        use_vllm=gpu,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=a.vllm_mem,
        bf16=gpu,
        gradient_checkpointing=gpu,
        logging_steps=1,
        save_steps=a.save_steps,
        save_total_limit=2,
        log_completions=True,
        report_to=["tensorboard"],
        seed=a.seed,
    )
    trainer = GRPOTrainer(
        model=a.model,
        reward_funcs=funcs,
        args=cfg,
        train_dataset=ds,
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    ckpts = sorted(Path(a.out).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts and not a.dry_run else None)
    if a.dry_run:
        print("DRY RUN OK; reward names:", names, "weights:", weights)
        return
    trainer.save_model(a.out)
    merged = trainer.model.merge_and_unload()
    merged.to(torch.bfloat16).save_pretrained(Path(a.out) / "merged")
    tok.save_pretrained(Path(a.out) / "merged")
    print("merged model ->", Path(a.out) / "merged")
    if a.cpu_smoke:
        print("CPU SMOKE OK; reward names:", names, "log:", [h for h in trainer.state.log_history if "reward" in h][-1:])


if __name__ == "__main__":
    main()
