"""Grade sampled differentials and compute coverage / diversity / calibration metrics.

    python score_samples.py --samples runs/qwen3-8b/samples.jsonl \
        --cases data/phenopackets/cases.jsonl --normalizer data/ontology/normalizer.pkl \
        --out runs/qwen3-8b/scores.json

Per case, with n samples of which c have an exact top-1:
    pass@k       unbiased estimator 1 - C(n-c, k) / C(n, k)
    list_recall  fraction of samples whose whole differential contains the gold (exact)
    distinct     number of distinct resolved top-1 diagnoses across samples
    entropy      entropy (nats) of the resolved top-1 distribution
Calibration uses the stated probability of each sample's top-1 vs. exact correctness.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np

from disease_normalizer import DiseaseNormalizer, norm_text
from parsing import parse_dx

KS = [1, 2, 4, 8, 16, 32, 64]


def pass_at_k(n, c, k):
    if k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--graded-out", help="optional per-case graded jsonl")
    a = ap.parse_args()

    norm = DiseaseNormalizer.load(a.normalizer)
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}

    @lru_cache(maxsize=None)
    def grade(name, gold):
        return norm.grade(name, gold)

    per_case, conf, corr = [], [], []
    tiers, parse_fail = Counter(), 0
    for line in open(a.samples):
        rec = json.loads(line)
        case = cases[rec["case_id"]]
        gold = case["disease_id"]
        top1, top1_ids, in_list = [], [], []
        for s in rec["samples"]:
            s["dx"] = parse_dx(s["text"])  # re-parse with the current parser
            if not s["dx"]:
                parse_fail += 1
                top1.append("unparsed"); top1_ids.append(None); in_list.append(False)
                continue
            g = grade(s["dx"][0]["diagnosis"], gold)
            top1.append(g["tier"])
            # unmapped answers still count toward diversity, keyed by their normalized text
            top1_ids.append(g["mondo"] or "txt:" + norm_text(s["dx"][0]["diagnosis"]))
            tiers[g["tier"]] += 1
            in_list.append(any(grade(d["diagnosis"], gold)["tier"] == "exact" for d in s["dx"]))
            p = s["dx"][0]["probability"]
            if p is not None and 0 <= p <= 1:
                conf.append(p); corr.append(g["tier"] == "exact")
        n = len(top1)
        c_exact = sum(t == "exact" for t in top1)
        c_rel = sum(t in ("exact", "related") for t in top1)
        dist = Counter(i for i in top1_ids if i)
        tot = sum(dist.values())
        ent = -sum(v / tot * math.log(v / tot) for v in dist.values()) if tot else 0.0
        per_case.append({
            "case_id": rec["case_id"], "gold": gold, "pub_year": case.get("pub_year"), "n": n,
            "c_exact": c_exact, "c_related": c_rel, "list_hits": sum(in_list),
            "distinct": len(dist), "entropy": ent, "top1_tiers": top1,
        })

    def summarize(rows):
        out = {"n_cases": len(rows)}
        for k in KS:
            v = [pass_at_k(r["n"], r["c_exact"], k) for r in rows]
            v = [x for x in v if x is not None]
            if v:
                out[f"pass@{k}"] = float(np.mean(v))
            v = [pass_at_k(r["n"], r["c_related"], k) for r in rows]
            v = [x for x in v if x is not None]
            if v:
                out[f"pass_related@{k}"] = float(np.mean(v))
        out["list_recall"] = float(np.mean([r["list_hits"] / r["n"] for r in rows]))
        out["distinct_top1"] = float(np.mean([r["distinct"] for r in rows]))
        out["entropy_top1"] = float(np.mean([r["entropy"] for r in rows]))
        out["diseases_ever_solved"] = len({r["gold"] for r in rows if r["c_exact"] > 0})
        out["diseases_total"] = len({r["gold"] for r in rows})
        return out

    by_era = defaultdict(list)
    for r in per_case:
        y = r["pub_year"]
        by_era["unknown" if y is None else ("2025+" if y >= 2025 else ("2023-24" if y >= 2023 else "<=2022"))].append(r)

    result = {
        "samples": a.samples,
        "overall": summarize(per_case),
        "by_pub_era": {k: summarize(v) for k, v in sorted(by_era.items())},
        "top1_tier_counts": dict(tiers),
        "unparsed_samples": parse_fail,
        "calibration": {"n": len(conf), "mean_conf": float(np.mean(conf)) if conf else None,
                        "accuracy": float(np.mean(corr)) if corr else None,
                        "ece": ece(conf, corr) if conf else None,
                        "brier": float(np.mean((np.array(conf) - np.array(corr, float)) ** 2)) if conf else None},
    }
    json.dump(result, open(a.out, "w"), indent=2)
    if a.graded_out:
        with open(a.graded_out, "w") as fh:
            for r in per_case:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps(result["overall"], indent=2))
    print("tiers:", dict(tiers), "unparsed:", parse_fail, "calib:", result["calibration"])


if __name__ == "__main__":
    main()
