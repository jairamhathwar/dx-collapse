"""Case reports: coverage of the authors' alternatives by one list and by the samples at equal set size.

    python r23_medcase_eq.py --cases cases_test.jsonl --runs RUNS/medcase_t07 --judge judge.jsonl \
        --ddx-judge judge_ddx.jsonl --authors-ddx authors_ddx_clean.jsonl --pairs "base>post ..." --out eq.json

Grading as in medcase_analysis.py (judge model; an answer counts for an alternative if the judge matched it to that
item). Per case with at least one alternative, over valid samples:
  list_m, samples_m  m = min(3, distinct sampled answers): share of the alternatives named by the first m items of
                     one list (mean over valid lists) and by the m most frequent sampled answers
  list_3, samples_3  the same with sets capped at three
  union_lists        share named by the union of all valid lists (first five items each)
Within each model list minus samples; across the pair the post-trained list (first three items) minus the base
model's three most frequent answers. Case bootstrap (2,000 resamples), 95% percentile intervals.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from medcase_analysis import load


def case_eq(samples, alts):
    v = [s for s in samples if s and s["items"] is not None]
    if not v or not alts:
        return None
    n_alt = len(alts)
    order = [k for k, _ in Counter(s["key"] for s in v).most_common()]
    item_of = {}
    for s in v:
        item_of.setdefault(s["key"], s["items"][0])
    m = min(3, len(order))
    cov = lambda items: len(alts & set(items)) / n_alt
    return {"m": m,
            "list_m": float(np.mean([cov(s["items"][:m]) for s in v])),
            "samples_m": cov([item_of[k] for k in order[:m]]),
            "list_3": float(np.mean([cov(s["items"][:3]) for s in v])),
            "samples_3": cov([item_of[k] for k in order[:3]]),
            "union_lists": cov([i for s in v for i in s["items"][:5]])}


def boot(x, rng, n=2000):
    x = np.asarray(x, float)
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return {"diff": float(x.mean()), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "n": len(x)}


def main():
    ap = argparse.ArgumentParser()
    for x in ["cases", "runs", "judge", "ddx-judge", "authors-ddx", "pairs", "out"]:
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    J = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge))}
    K = {(r["case_id"], r["key"]): r["item"] for r in map(json.loads, open(a.ddx_judge))}
    ALTS = {r["case_id"]: frozenset(r.get("keep", range(2, r["n_alternatives"] + 2))) for r in map(json.loads, open(a.authors_ddx))}
    rng = np.random.default_rng(0)
    rows, report = {}, {"models": {}, "within": {}, "pairs": {}}
    for pair in a.pairs.split():
        b, p = pair.split(">")
        for t in (b, p):
            if t in rows:
                continue
            R = load(Path(a.runs) / t, J, K)
            rows[t] = {c: r for c, s in R.items() if (r := case_eq(s, ALTS.get(c, frozenset()))) is not None}
            m = {k: float(np.mean([r[k] for r in rows[t].values()])) for k in ["m", "list_m", "samples_m", "list_3", "samples_3", "union_lists"]}
            report["models"][t] = m
            report["within"][t] = {"list_minus_samples_m": boot([r["list_m"] - r["samples_m"] for r in rows[t].values()], rng),
                                   "list_minus_samples_3": boot([r["list_3"] - r["samples_3"] for r in rows[t].values()], rng)}
            print(t, json.dumps({k: round(v, 3) for k, v in m.items()}),
                  json.dumps({k: [round(v["diff"], 3), [round(x, 3) for x in v["ci"]]] for k, v in report["within"][t].items()}), flush=True)
        cs = [c for c in rows[p] if c in rows[b]]
        report["pairs"][pair] = {"post_list3_minus_base_samples3": boot([rows[p][c]["list_3"] - rows[b][c]["samples_3"] for c in cs], rng),
                                 "post_union_minus_base_samples3": boot([rows[p][c]["union_lists"] - rows[b][c]["samples_3"] for c in cs], rng)}
        print(pair, json.dumps({k: [round(v["diff"], 3), [round(x, 3) for x in v["ci"]]] for k, v in report["pairs"][pair].items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
