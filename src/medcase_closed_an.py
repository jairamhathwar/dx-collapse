"""Closed-set case reports: scoring without a judge (CPU only).

    python medcase_closed_an.py --runs RUNS --candidates candidates_closed.jsonl \
        --pairs "qwen3-8b-base>qwen3-8b ..." --out closed.json

Text after the first invented chat turn (a line 'user', 'assistant' or 'system') is dropped. Each answer name is mapped to the case's candidate list (normalized text equal, else the best rapidfuzz
token-set match with score >= 90); a sample is valid if its top answer maps to a candidate. Per model: valid share,
pass@1 and pass@32 (invalid counts as wrong), distinct top answers, recall@5 of one list, the share of the authors'
alternatives named by one list (mean over valid lists) and by any of the 32 top answers, distractors named per list,
Vote@32 gain, and AUROC of the stated probability and of 31-sample agreement for the single-call answer.
Pairs: post-trained minus base, paired case bootstrap (cases are independent reports), Holm over all pair tests;
within each model, alternatives covered by one list minus by 32 samples and stated minus agreement AUROC.
"""
import argparse
import json
import math
import re
from collections import Counter

import numpy as np
from rapidfuzz import fuzz, process

from consequences import auroc
from disease_normalizer import norm_text
from medcase_analysis import sample_files
from parsing import parse_dx

METRICS = ["valid", "pass1", "pass32", "distinct", "rec5", "alt_list", "alt_samples", "distractors_list", "vote_gain"]


def pass_at_k(n, c, k):
    if n < k or n == 0:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def boot(v, rng, n_boot):
    v = np.asarray(v, float)
    b = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    p = float(min(1.0, 2 * (1 + min((b <= 0).sum(), (b >= 0).sum())) / (n_boot + 1)))
    return {"diff": float(v.mean()), "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])], "p": p, "n": len(v)}


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "candidates", "pairs", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    C = {r["case_id"]: r for r in map(json.loads, open(a.candidates))}
    norms = {cid: [norm_text(x) for x in r["candidates"]] for cid, r in C.items()}
    cache = {}

    def match(cid, name):
        k = (cid, name)
        if k not in cache:
            n = norm_text(name)
            if n in norms[cid]:
                cache[k] = norms[cid].index(n)
            else:
                best = process.extractOne(n, norms[cid], scorer=fuzz.token_set_ratio)
                cache[k] = best[2] if best and best[1] >= 90 else None
        return cache[k]

    def load(tag):
        per = {}
        for rec in (json.loads(l) for f in sample_files(f"{a.runs}/{tag}") for l in open(f)):
            cid = rec["case_id"]
            roles = C[cid]["roles"]
            gi = roles.index("gold")
            alts = {i for i, r in enumerate(roles) if r == "alt"}
            tops, lists = [], []
            for s in rec["samples"]:
                # base models sometimes continue with invented chat turns; keep only the first answer
                dx = parse_dx(re.split(r"\n(?:user|assistant|system)\n", s["text"])[0])
                idx = [match(cid, d["diagnosis"]) for d in dx]
                if not dx or idx[0] is None:
                    tops.append(None)
                    continue
                ps = [d["probability"] for d, i in zip(dx, idx) if i is not None]
                ii = [i for i in idx if i is not None]
                if any(p is None or not 0 <= p <= 1 for p in ps) or sum(ps) <= 0:
                    ps = [1.0] * len(ii)
                tops.append(idx[0])
                lists.append((ii, ps[0] / sum(ps)))
            n, v = len(tops), [t for t in tops if t is not None]
            c = sum(t == gi for t in v)
            r = {"valid": len(v) / n, "pass1": pass_at_k(n, c, 1), "pass32": pass_at_k(n, c, min(32, n))}
            if v:
                r["distinct"] = float(len(set(v)))
                r["rec5"] = float(np.mean([gi in ii[:5] for ii, _ in lists]))
                r["distractors_list"] = float(np.mean([sum(roles[i] == "distractor" for i in set(ii[:5])) for ii, _ in lists]))
                if alts:
                    r["alt_list"] = float(np.mean([len(alts & set(ii[:5])) / len(alts) for ii, _ in lists]))
                    r["alt_samples"] = len(alts & set(v)) / len(alts)
                A = v[0]
                cnt = Counter(v)
                r["A_correct"] = float(A == gi)
                r["stated_p"] = lists[0][1]
                r["agree32"] = float(np.mean([t == A for t in v[1:]])) if len(v) > 1 else None
                r["vote_gain"] = float(cnt.most_common(1)[0][0] == gi) - float(A == gi)
            per[cid] = r
        return per

    rng = np.random.default_rng(0)
    report = {"models": {}, "pairs": {}, "within": {}}
    tests = []
    for pr in a.pairs.split():
        b, p = pr.split(">")
        B, P = load(b), load(p)
        for t, R in [(b, B), (p, P)]:
            m = {k: float(np.mean([r[k] for r in R.values() if r.get(k) is not None])) for k in METRICS}
            rr = [r for r in R.values() if r.get("agree32") is not None]
            y = np.array([r["A_correct"] for r in rr])
            m["auroc_stated"] = auroc(np.array([r["stated_p"] for r in rr]), y)
            m["auroc_agree32"] = auroc(np.array([r["agree32"] for r in rr]), y)
            report["models"][t] = m
            cs = [c for c, r in R.items() if r.get("alt_list") is not None]
            w = {"alt_list-samples": boot([R[c]["alt_list"] - R[c]["alt_samples"] for c in cs], rng, a.n_boot)}
            ix = np.arange(len(rr))
            sa, sg = np.array([r["stated_p"] for r in rr]), np.array([r["agree32"] for r in rr])
            bs = []
            for _ in range(2000):
                j = rng.integers(0, len(rr), len(rr))
                if 0 < y[j].sum() < len(j):
                    bs.append(auroc(sa[j], y[j]) - auroc(sg[j], y[j]))
            w["auroc_stated-agree32"] = {"diff": m["auroc_stated"] - m["auroc_agree32"], "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}
            report["within"][t] = w
            print(t, json.dumps({k: round(v, 4) for k, v in m.items()}), json.dumps(w), flush=True)
        d = {}
        for k in METRICS:
            cs = [c for c in B if c in P and B[c].get(k) is not None and P[c].get(k) is not None]
            d[k] = boot([P[c][k] - B[c][k] for c in cs], rng, a.n_boot)
            tests.append((pr, k))
        report["pairs"][pr] = d
    for (pr, k), ph in zip(tests, holm([report["pairs"][pr][k]["p"] for pr, k in tests])):
        report["pairs"][pr][k]["p_holm"] = ph
    for pr, d in report["pairs"].items():
        print(pr, json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["ci"]], round(v["p_holm"], 4)] for k, v in d.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
