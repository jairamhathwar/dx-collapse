"""Phenopacket Store intervals clustered by disease (CPU only; reads score_samples.py's graded.jsonl).

    python r14_pheno.py --runs RUNS --out r14_pheno.json

For each comparison (later model minus earlier model, same cases), pass@k for k = 1..32 and distinct top answers, with
95% intervals from a bootstrap that resamples the diseases (gold disease ids) with all their cases (10,000 resamples),
and per model pass@1 / pass@32 by publication era of the source case report.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

from review2 import boot_p

KS = [1, 2, 4, 8, 16, 32]
COMP = [("go_full", "qwen25-7b-it", t) for t in ["alphamed-7b", "huatuo-o1-7b", "m1-7b"]] + \
       [("go_full", "qwen25-7b-base", "alphamed-7b-base-rl"), ("go_full", "qwen3-8b-base", "qwen3-8b"),
        ("go_full", "qwen25-7b-base", "qwen25-7b-it")] + \
       [("go_full_olmo", "olmo3-base", t) for t in ["olmo3-sft", "olmo3-dpo", "olmo3-instruct", "olmo3-rlzero-general"]]
ERA = lambda y: "unknown" if y is None else ("2025+" if y >= 2025 else ("2023-24" if y >= 2023 else "<=2022"))


def pass_at_k(n, c, k):
    if k > n:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(path):
    out = {}
    for r in map(json.loads, open(path)):
        row = {"gold": r["gold"], "era": ERA(r.get("pub_year")), "distinct": r["distinct"]}
        for k in KS:
            row[f"pass{k}"] = pass_at_k(r["n"], r["c_exact"], k)
        out[r["case_id"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cache, report = {}, {"models": {}, "pairs": {}}

    def get(d, t):
        if (d, t) not in cache:
            cache[(d, t)] = load(f"{a.runs}/{d}/{t}/graded.jsonl")
            R = cache[(d, t)]
            m = {k: float(np.mean([r[k] for r in R.values()])) for k in [f"pass{k}" for k in KS] + ["distinct"]}
            for era in ["<=2022", "2023-24", "2025+"]:
                rr = [r for r in R.values() if r["era"] == era]
                m[f"era_{era}"] = {"n": len(rr), "pass1": float(np.mean([r["pass1"] for r in rr])),
                                   "pass32": float(np.mean([r["pass32"] for r in rr]))}
            report["models"][f"{d}/{t}"] = m
            print(d, t, json.dumps(m), flush=True)
        return cache[(d, t)]

    W = None
    for d, b, p in COMP:
        B, P = get(d, b), get(d, p)
        cs = sorted(set(B) & set(P))
        keys = sorted({B[c]["gold"] for c in cs})
        if W is None or W.shape[1] != len(keys):
            W = rng.multinomial(len(keys), [1 / len(keys)] * len(keys), size=a.n_boot).astype(float)
        idx = {k: i for i, k in enumerate(keys)}
        N = np.zeros(len(keys))
        for c in cs:
            N[idx[B[c]["gold"]]] += 1
        pr = {"n_cases": len(cs), "n_diseases": len(keys)}
        for m in [f"pass{k}" for k in KS] + ["distinct"]:
            S = np.zeros(len(keys))
            for c in cs:
                S[idx[B[c]["gold"]]] += P[c][m] - B[c][m]
            bs = (W @ S) / (W @ N)
            pr[m] = {"diff": float(S.sum() / N.sum()), "cluster_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "p": boot_p(bs)}
        report["pairs"][f"{d}:{b}>{p}"] = pr
        print(d, b, p, json.dumps({m: [round(v["diff"] * (100 if m != "distinct" else 1), 2), [round(x * (100 if m != "distinct" else 1), 2) for x in v["cluster_ci"]]]
                                   for m, v in pr.items() if isinstance(v, dict)}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
