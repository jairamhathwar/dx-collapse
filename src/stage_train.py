"""Post-training stages on a small model, for stage-arithmetic and DPO-pairing experiments (full fine-tuning).

    python stage_train.py sft  --model M --out DIR [--n 20000] [--seed 0] [--lr 1e-5]
    python stage_train.py dpo  --model M --out DIR [--n 10000] [--seed 0] [--lr 2e-6] [--beta 0.1]
                               [--pairing real|shuffled|pool|flipped] [--data-seed 0]
    python stage_train.py sft-chosen --model M --out DIR ...      # SFT on the chosen responses of the DPO pairs
    python stage_train.py arith --out DIR --terms "+A -B +C"      # weight arithmetic over saved models (state dicts)
    python stage_train.py prefacc --model M [--n 1000]             # held-out preference accuracy (chosen vs rejected)

Data: SFT = allenai/tulu-3-sft-olmo-2-mixture-0225 (messages); DPO = allenai/olmo-2-0425-1b-preference-mix
(chosen / rejected conversations). Subsamples are fixed by --data-seed, independent of the training seed; the last
2,000 pairs of the shuffled preference data are held out for prefacc and never trained on.
DPO --pairing: real = each prompt with its own rejected response; shuffled = rejected responses permuted across
prompts; pool = every rejected response drawn from a fixed pool of 100; flipped = chosen and rejected swapped.
The tokenizer (and chat template) always comes from --tokenizer (default allenai/OLMo-2-0425-1B-SFT).
"""
import argparse
import json
import random

import torch

SFT_DATA = "allenai/tulu-3-sft-olmo-2-mixture-0225"
PREF_DATA = "allenai/olmo-2-0425-1b-preference-mix"
HELD_OUT = 2000


def pref_rows(n, data_seed, held_out=False, max_chars=6000):
    from datasets import load_dataset
    ds = load_dataset(PREF_DATA, split="train").shuffle(seed=1234)
    idx = range(len(ds) - HELD_OUT, len(ds)) if held_out else range(len(ds) - HELD_OUT)
    rows = []
    order = list(idx)
    if not held_out:
        random.Random(data_seed).shuffle(order)
    for i in order:
        r = ds[i]
        ch, rj = r["chosen"], r["rejected"]
        if len(ch) < 2 or ch[-1]["role"] != "assistant" or rj[-1]["role"] != "assistant":
            continue
        if sum(len(m["content"]) for m in ch) > max_chars or len(rj[-1]["content"]) > max_chars:
            continue
        rows.append({"prompt": ch[:-1], "chosen": [ch[-1]], "rejected": [rj[-1]]})
        if len(rows) >= n:
            break
    return rows


def apply_pairing(rows, pairing, seed):
    rng = random.Random(seed)
    rej = [r["rejected"] for r in rows]
    if pairing == "shuffled":
        perm = list(range(len(rows)))
        rng.shuffle(perm)
        rej = [rej[j] for j in perm]
    elif pairing == "pool":
        pool = rng.sample(rej, 100)
        rej = [rng.choice(pool) for _ in rows]
    out = []
    for r, x in zip(rows, rej):
        if pairing == "flipped":
            out.append({"prompt": r["prompt"], "chosen": r["rejected"], "rejected": r["chosen"]})
        else:
            out.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": x})
    return out


def sft_rows(n, data_seed, max_chars=6000):
    from datasets import load_dataset
    ds = load_dataset(SFT_DATA, split="train").shuffle(seed=data_seed)
    rows = []
    for r in ds:
        m = r["messages"]
        if m and m[-1]["role"] == "assistant" and sum(len(x["content"]) for x in m) <= max_chars:
            rows.append({"messages": m})
        if len(rows) >= n:
            break
    return rows


def train(a):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    common = dict(output_dir=a.out, learning_rate=a.lr, num_train_epochs=1, per_device_train_batch_size=a.bs,
                  gradient_accumulation_steps=a.accum, lr_scheduler_type="linear", warmup_ratio=0.03, bf16=True,
                  logging_steps=10, save_strategy="no", report_to=[], seed=a.seed, gradient_checkpointing=True)
    if a.cmd == "dpo":
        from trl import DPOConfig, DPOTrainer
        rows = apply_pairing(pref_rows(a.n, a.data_seed), a.pairing, a.data_seed)
        cfg = DPOConfig(beta=a.beta, max_length=a.max_length, **common)
        trainer = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok)
    else:
        from trl import SFTConfig, SFTTrainer
        if a.cmd == "sft-chosen":
            rows = [{"messages": r["prompt"] + r["chosen"]} for r in pref_rows(a.n, a.data_seed)]
        else:
            rows = sft_rows(a.n, a.data_seed)
        cfg = SFTConfig(max_length=a.max_length, **common)
        trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(rows), processing_class=tok)
    print(f"{a.cmd}: {len(rows)} examples, model {a.model}", flush=True)
    trainer.train()
    trainer.model.to(torch.bfloat16).save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump({k: v for k, v in vars(a).items()}, open(f"{a.out}/stage_args.json", "w"))
    print("saved", a.out)


def arith(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    terms = a.terms.split()
    acc = None
    for t in terms:
        sign, path = (1.0 if t[0] == "+" else -1.0), t[1:]
        sd = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).state_dict()
        if acc is None:
            acc = {k: sign * v.clone() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                acc[k] += sign * v
        del sd
    model = AutoModelForCausalLM.from_pretrained(terms[0][1:], torch_dtype=torch.float32)
    model.load_state_dict(acc)
    model.to(torch.bfloat16).save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.tokenizer).save_pretrained(a.out)
    print("arith", a.terms, "->", a.out)


@torch.no_grad()
def prefacc(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    rows = pref_rows(a.n, 0, held_out=True)

    def logp(prompt, resp):
        p = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        pi = tok(p, return_tensors="pt").input_ids
        ri = tok(resp[0]["content"] + (tok.eos_token or ""), return_tensors="pt", add_special_tokens=False).input_ids[:, :a.max_length]
        ids = torch.cat([pi, ri], 1)[:, -4096:].to(model.device)
        lg = torch.log_softmax(model(ids).logits.float()[0, -ri.shape[1] - 1:-1], -1)
        return float(lg.gather(1, ids[0, -ri.shape[1]:, None]).sum()), ri.shape[1]

    win = winn = 0
    for r in rows:
        (lc, nc), (lr, nr) = logp(r["prompt"], r["chosen"]), logp(r["prompt"], r["rejected"])
        win += lc > lr
        winn += lc / nc > lr / nr
    out = {"model": a.model, "n": len(rows), "pref_acc": win / len(rows), "pref_acc_per_token": winn / len(rows)}
    print(json.dumps(out), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sft", "dpo", "sft-chosen", "arith", "prefacc"])
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--tokenizer", default="allenai/OLMo-2-0425-1B-SFT")
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--pairing", choices=["real", "shuffled", "pool", "flipped"], default="real")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--terms")
    a = ap.parse_args()
    if a.cmd == "arith":
        return arith(a)
    if a.cmd == "prefacc":
        a.n = a.n or 1000
        return prefacc(a)
    a.n = a.n or (10000 if a.cmd != "sft" else 20000)
    a.lr = a.lr or (2e-6 if a.cmd == "dpo" else 1e-5)
    train(a)


if __name__ == "__main__":
    main()
