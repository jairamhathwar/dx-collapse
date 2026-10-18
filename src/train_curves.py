"""Print GRPO training curves (rewards, entropy, length, grad norm) from a Slurm log,
and optionally which top-1 answers an evaluated model gives across cases.

    python train_curves.py LOG [LOG ...] [--samples SAMPLES_JSONL --conditions CONDITIONS_JSON]
"""
import argparse
import ast
import json
import re
from collections import Counter

KEYS = [("fmt", "rewards/r_format/mean"), ("acc", "rewards/r_acc/mean"), ("brier", "rewards/r_brier/mean"),
        ("overlap", "rewards/r_overlap/mean"), ("multi", "rewards/r_multi_rlvr/mean"),
        ("ent", "entropy"), ("len", "completions/mean_length"), ("zstd", "frac_reward_zero_std"),
        ("gnorm", "grad_norm"), ("kl", "kl")]


def curves(log, every=25):
    rows = [ast.literal_eval(m) for m in re.findall(r"\{'loss'[^{}]*\}", open(log).read())]
    print(f"== {log}  ({len(rows)} logged steps)")
    present = [(n, k) for n, k in KEYS if rows and k in rows[0]]
    print("  step " + " ".join(f"{n:>7s}" for n, _ in present))
    for i in sorted(set(list(range(0, len(rows), every)) + [len(rows) - 1])):
        print(f"  {i + 1:4d} " + " ".join(f"{float(rows[i][k]):7.3f}" for _, k in present))


def top1_mix(samples, conditions):
    from ddx_names import Matcher
    from parsing import parse_dx
    m = Matcher(json.load(open(conditions)))
    c = Counter()
    for rec in map(json.loads, open(samples)):
        for s in rec["samples"]:
            dx = parse_dx(s["text"])
            c[m(dx[0]["diagnosis"]) if dx else "<unparsed>"] += 1
    tot = sum(c.values())
    print(f"== top-1 answer mix in {samples}: {len(c)} distinct over {tot} samples")
    for name, n in c.most_common(6):
        print(f"   {n / tot:6.1%}  {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*")
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--conditions")
    a = ap.parse_args()
    for log in a.logs:
        curves(log)
    for s in a.samples:
        top1_mix(s, a.conditions)
