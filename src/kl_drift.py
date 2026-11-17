"""Drift of RL checkpoints from their starting model: KL on off-task and on-task prompts, per checkpoint.

    python kl_drift.py --base Qwen/Qwen3-1.7B --runs RUN_DIR [RUN_DIR ...] --off ambigqa_cases.jsonl \
        --on ddx_cases.jsonl --conditions conditions.json --out drift.json [--n-off 200] [--n-on 100] [--tokens 48]

For each prompt the base model's greedy continuation (--tokens tokens) is computed once. For every checkpoint-*
directory of each run (a LoRA adapter or a full model), KL(base || checkpoint) of the next-token distribution is
averaged over the continuation positions and over prompts: kl_off on unrelated prompts (AmbigQA questions), kl_on on
the training task's test prompts (DDXPlus). Each checkpoint is paired with the run's training reward around its step
(mean of rewards/r_acc/mean over the last 10 logged steps, from trainer_state.json), so runs can be compared at
matched reward. The final adapter/model in RUN_DIR itself is not scored (its last checkpoint is).
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sample_diagnoses import DDX_SYSTEM, QA_INSTRUCTIONS, QA_SYSTEM, build_ddx_prompt


@torch.no_grad()
def logits_on(model, seqs, starts, bs=16):
    out = []
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        L = max(len(x) for x in batch)
        ids = torch.full((len(batch), L), 0, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, x in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x)
            att[j, :len(x)] = 1
        lg = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits.float()
        for j, x in enumerate(batch):
            s = starts[i + j]
            out.append(F.log_softmax(lg[j, s - 1:len(x) - 1], dim=-1).cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-off", type=int, default=200)
    ap.add_argument("--n-on", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=48)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base)
    conditions = json.load(open(a.conditions))
    chat = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    off = [json.loads(l) for l in open(a.off)][:a.n_off]
    on = [json.loads(l) for l in open(a.on)][:a.n_on]
    prompts = [chat([{"role": "system", "content": QA_SYSTEM},
                     {"role": "user", "content": QA_INSTRUCTIONS.format(question=c["question"], n=5)}]) for c in off]
    prompts += [chat([{"role": "system", "content": DDX_SYSTEM},
                      {"role": "user", "content": build_ddx_prompt(c, 5, conditions)}]) for c in on]
    kind = ["off"] * len(off) + ["on"] * len(on)

    load = lambda p: AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    base = load(a.base)
    seqs, starts = [], []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(base.device)
            gen = base.generate(ids, max_new_tokens=a.tokens, do_sample=False)
            seqs.append(gen[0].tolist())
            starts.append(ids.shape[1])
    ref = logits_on(base, seqs, starts)
    del base
    torch.cuda.empty_cache()

    report = {}
    for run in a.runs:
        run = Path(run)
        state = None
        rows = []
        for ck in sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            step = int(ck.name.split("-")[1])
            if (ck / "adapter_config.json").exists():
                from peft import PeftModel
                model = PeftModel.from_pretrained(load(a.base), str(ck)).eval()
            else:
                model = load(str(ck))
            cur = logits_on(model, seqs, starts)
            kl = {"off": [], "on": []}
            for k, r, c in zip(kind, ref, cur):
                kl[k].append(float((r.exp() * (r - c)).sum(-1).mean()))
            state = json.load(open(ck / "trainer_state.json"))
            logs = [h for h in state["log_history"] if "rewards/r_acc/mean" in h and h.get("step", 0) <= step][-10:]
            rew = sum(float(h["rewards/r_acc/mean"]) for h in logs) / max(1, len(logs))
            rows.append({"step": step, "kl_off": sum(kl["off"]) / len(kl["off"]), "kl_on": sum(kl["on"]) / len(kl["on"]),
                         "train_acc": rew})
            print(run.name, json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in rows[-1].items()}), flush=True)
            del model
            torch.cuda.empty_cache()
        report[run.name] = rows
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
