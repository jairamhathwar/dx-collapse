"""Breadth within one list versus across calls (DDXPlus, CPU only).

    python r16_breadth.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json --out r16.json

Per case and model, over the first k valid samples (k = 1, 5; for k = 32 all valid samples, up to 32):
  list_union_k     conditions named by the union of the k stated lists
  list_mass_k      reference probability of that union
  samp_distinct_k  distinct top answers among the k samples
  samp_mass_k      reference probability of those answers (Cov@k)
  warranted_k      share of the conditions the reference weights >= 0.1 that the union of lists names
Post-trained minus base with condition-clustered intervals (r14_an.Boot, 10,000 resamples). This separates breadth
within one list (k = 1) from breadth across calls (k = 5, 32) for both readouts.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from r14_an import Boot
from stated_vs_sampled import stated_dist

PAIRS = [("pair_q3", "qwen3-8b-base", "qwen3-8b"), ("pair25_t07", "qwen25-7b-base", "qwen25-7b-it"),
         ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it"), ("ddx_olmo", "olmo3-base", "olmo3-instruct"),
         ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct")]
KS = [1, 5, 32]


def load(path, refs, match):
    out = {}
    for rec in map(json.loads, open(path)):
        cid, ref = rec["case_id"], refs[rec["case_id"]]
        valid = []
        for s in rec["samples"]:
            d, top = stated_dist(parse_dx(s["text"]), match)
            if d is not None and top:
                valid.append((d, top))
        r = {}
        warr = {k for k, p in ref.items() if p >= 0.1}
        for k in KS:
            v = valid[:k]
            if not v or (k < 32 and len(v) < k):
                continue  # k = 32: all valid samples (base models often have fewer than 32)
            u = set().union(*(set(d) for d, _ in v))
            tops = {t for _, t in v}
            r[f"list_union_{k}"] = len(u)
            r[f"list_mass_{k}"] = sum(ref.get(c, 0) for c in u)
            r[f"samp_distinct_{k}"] = len(tops)
            r[f"samp_mass_{k}"] = sum(ref.get(c, 0) for c in tops)
            if warr:
                r[f"warranted_{k}"] = len(warr & u) / len(warr)
        out[cid] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "cases", "conditions-json", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    gold = {cid: c["gold"] for cid, c in cases.items()}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)
    boot = Boot(list(cases), gold, a.n_boot, rng)
    metrics = [f"{m}_{k}" for k in KS for m in ["list_union", "list_mass", "samp_distinct", "samp_mass", "warranted"]]
    report = {"models": {}, "pairs": {}}
    for d, b, p in PAIRS:
        R = {t: load(f"{a.runs}/{d}/{t}/samples.jsonl", refs, match) for t in (b, p)}
        for t in (b, p):
            report["models"][f"{d}/{t}"] = {m: float(np.mean([r[m] for r in R[t].values() if m in r])) for m in metrics}
            print(t, json.dumps({m: round(v, 3) for m, v in report["models"][f"{d}/{t}"].items()}), flush=True)
        pr = {}
        for m in metrics:
            v = {c: R[p][c][m] - R[b][c][m] for c in cases if m in R[b].get(c, {}) and m in R[p].get(c, {})}
            if len(v) < 50:
                continue
            pr[m] = boot.ratio(v, {c: 1.0 for c in v})
            pr[m]["n"] = len(v)
        report["pairs"][f"{d}:{b}>{p}"] = pr
        print(p, json.dumps({m: [round(x["diff"], 3), [round(y, 3) for y in x["cluster_ci"]]] for m, x in pr.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
