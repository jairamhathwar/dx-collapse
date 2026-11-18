"""Pairwise functional distance between saved models: mean next-token KL on held-out prompts.

    python kl_pair.py --tokenizer T --prompts ambigqa_cases.jsonl --out out.json [--n 200] [--tokens 48] \
        name=PATH [name=PATH ...] --pairs a:b [a:b ...]

For each pair a:b, KL(a || b) is averaged over the positions of a's greedy continuation (--tokens tokens) of each chat
prompt (AmbigQA questions with the QA instructions), in float32. Logits of each model on each reference model's
continuations are computed once per (reference, model) combination that the pairs need.
"""
import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from sample_diagnoses import QA_INSTRUCTIONS, QA_SYSTEM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("models", nargs="+")
    ap.add_argument("--pairs", nargs="+", required=True)
    a = ap.parse_args()
    paths = dict(m.split("=", 1) for m in a.models)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    qs = [json.loads(l) for l in open(a.prompts)][:a.n]
    prompts = [tok.apply_chat_template([{"role": "system", "content": QA_SYSTEM},
                                        {"role": "user", "content": QA_INSTRUCTIONS.format(question=c["question"], n=5)}],
                                       tokenize=False, add_generation_prompt=True) for c in qs]
    pairs = [p.split(":") for p in a.pairs]
    refs = sorted({x for x, _ in pairs})
    load = lambda n: AutoModelForCausalLM.from_pretrained(paths[n], torch_dtype=torch.float32, device_map="cuda").eval()

    @torch.no_grad()
    def logp(model, seqs, starts):
        out = []
        for s, st in zip(seqs, starts):
            lg = model(torch.tensor([s], device=model.device)).logits.float()[0, st - 1:len(s) - 1]
            out.append(F.log_softmax(lg, -1).cpu())
        return out

    seqs, starts, ref_lp = {}, {}, {}
    for r in refs:
        m = load(r)
        seqs[r], starts[r] = [], []
        with torch.no_grad():
            for p in prompts:
                ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(m.device)
                gen = m.generate(ids, max_new_tokens=a.tokens, do_sample=False, pad_token_id=tok.eos_token_id)
                seqs[r].append(gen[0].tolist())
                starts[r].append(ids.shape[1])
        ref_lp[r] = logp(m, seqs[r], starts[r])
        del m
        torch.cuda.empty_cache()
    report = {}
    for b in sorted({y for _, y in pairs}):
        m = load(b)
        for r in refs:
            if [r, b] not in pairs:
                continue
            cur = logp(m, seqs[r], starts[r])
            kl = [float((x.exp() * (x - y)).sum(-1).mean()) for x, y in zip(ref_lp[r], cur)]
            report[f"{r}:{b}"] = sum(kl) / len(kl)
            print(f"KL({r} || {b}) = {report[f'{r}:{b}']:.5f}", flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
