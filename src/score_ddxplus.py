"""Grade DDXPlus samples (closed set of 49 conditions) against gold + ground-truth differential.

    python score_ddxplus.py --samples runs/ddx/<tag>/samples.jsonl \
        --cases data/ddxplus/cases_test.jsonl --conditions data/ddxplus/conditions.json \
        --out runs/ddx/<tag>/scores.json --graded-out runs/ddx/<tag>/graded.jsonl

graded.jsonl uses the same fields as score_samples.py (n, c_exact, c_related, list_hits,
distinct, entropy) so compare_runs.py works unchanged. Here c_related counts top-1 answers
that are in the ground-truth differential (plausible but not the gold pathology).
Extra DDXPlus metrics per case:
    ddx_cov@k   expected share of ground-truth differential probability mass covered by the
                distinct top-1 answers in k samples
    tv          total-variation distance between the empirical top-1 distribution over all
                samples and the ground-truth differential distribution
"""
import argparse
import json
import math
from collections import Counter

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def coverage_at_k(top1, ddx, k, n_perm=50, rng=None):
    """Expected GT-differential mass covered by distinct answers among k random samples."""
    valid = [t for t in top1]
    if k > len(valid):
        return None
    vals = []
    for _ in range(n_perm):
        pick = rng.choice(len(valid), size=k, replace=False)
        seen = {valid[i] for i in pick if valid[i]}
        vals.append(sum(p for c, p in ddx.items() if c in seen))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--graded-out")
    a = ap.parse_args()

    match = Matcher(json.load(open(a.conditions)))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    rng = np.random.default_rng(0)
    ks = [1, 2, 4, 8, 16, 32]

    graded, conf, corr = [], [], []
    unparsed = unmatched = 0
    unmatched_names = Counter()
    for rec in map(json.loads, open(a.samples)):
        case = cases[rec["case_id"]]
        gold = case["gold"]
        ddx = {c: p for c, p in case["gt_differential"]}
        tot = sum(ddx.values())
        ddx = {c: p / tot for c, p in ddx.items()}
        top1, list_hits, ddx_recall = [], 0, []
        for s in rec["samples"]:
            s["dx"] = parse_dx(s["text"])  # re-parse with the current parser
            if not s["dx"]:
                unparsed += 1
                top1.append(None)
                ddx_recall.append(0.0)
                continue
            names = [match(d["diagnosis"]) for d in s["dx"]]
            for d, m in zip(s["dx"], names):
                if m is None:
                    unmatched += 1
                    unmatched_names[d["diagnosis"]] += 1
            top1.append(names[0])
            list_hits += gold in names
            ddx_recall.append(sum(ddx.get(m, 0) for m in set(names) if m))
            p = s["dx"][0]["probability"]
            if p is not None and 0 <= p <= 1:
                conf.append(p)
                corr.append(names[0] == gold)
        dist = Counter(t for t in top1 if t)
        n_ok = sum(dist.values())
        emp = {c: v / n_ok for c, v in dist.items()} if n_ok else {}
        tv = 0.5 * sum(abs(emp.get(c, 0) - ddx.get(c, 0)) for c in set(emp) | set(ddx))
        ent = -sum(v * math.log(v) for v in emp.values())
        graded.append({
            "case_id": rec["case_id"], "gold": gold, "n": len(top1),
            "c_exact": sum(t == gold for t in top1),
            "c_related": sum(t in ddx for t in top1 if t),
            "list_hits": list_hits, "distinct": len(dist), "entropy": ent, "tv": tv,
            "gt_entropy": -sum(p * math.log(p) for p in ddx.values() if p > 0),
            "ddx_list_recall": float(np.mean(ddx_recall)),
            **{f"ddx_cov@{k}": coverage_at_k(top1, ddx, k, rng=rng) for k in ks if k <= len(top1)},
        })

    def mean(key):
        v = [g[key] for g in graded if g.get(key) is not None]
        return float(np.mean(v)) if v else None

    res = {
        "samples": a.samples, "n_cases": len(graded), "unparsed_samples": unparsed,
        "unmatched_names": unmatched, "top_unmatched": unmatched_names.most_common(15),
        "acc_top1": mean("c_exact") / graded[0]["n"] if graded else None,
        "tv_to_gt_ddx": mean("tv"), "entropy_top1": mean("entropy"), "gt_entropy": mean("gt_entropy"),
        "distinct_top1": mean("distinct"), "ddx_list_recall": mean("ddx_list_recall"),
        **{f"ddx_cov@{k}": mean(f"ddx_cov@{k}") for k in ks},
        "calibration": {"n": len(conf), "mean_conf": float(np.mean(conf)) if conf else None,
                        "accuracy": float(np.mean(corr)) if corr else None,
                        "ece": ece(conf, corr) if conf else None,
                        "brier": float(np.mean((np.array(conf) - np.array(corr, float)) ** 2)) if conf else None},
    }
    json.dump(res, open(a.out, "w"), indent=2)
    if a.graded_out:
        with open(a.graded_out, "w") as fh:
            for g in graded:
                fh.write(json.dumps(g) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "top_unmatched"}, indent=1))
    print("top unmatched:", res["top_unmatched"][:8])


if __name__ == "__main__":
    main()
