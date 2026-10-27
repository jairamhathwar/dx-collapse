"""Does one post-trained list reproduce its own base model's averaged list? (DDXPlus, CPU only)

    python packing_test.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --pairs "pair_q3:qwen3-8b-base>qwen3-8b ..." --out packing.json

For each case c, B(c) is the mean of the base model's valid stated lists for c (each renormalized over the
conditions it names). For each post-trained list q of case c we measure
  own        TV(q, B(c))                        distance to its own case's averaged base list
  same_gold  mean TV(q, B(c')) over the other cases c' with the same true pathology
  any_case   mean TV(q, B(c')) over 19 random other cases
  ref        TV(q, reference(c))
and, as within-model benchmarks,
  base_loo   TV(b_i, mean of the other base lists of c), for each base list b_i
  post_loo   TV(q_i, mean of the other post lists of c)
Top answers: share of post-trained lists whose first condition is the argmax of B(c), against the same share
for single base lists against the mean of the other base lists (leave-one-out).
Intervals resample the 49 true pathologies (condition-clustered bootstrap).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from review2 import boot_p
from stated_vs_sampled import stated_dist, tv


def mean_dist(ds):
    out = Counter()
    for d in ds:
        for k, v in d.items():
            out[k] += v / len(ds)
    return dict(out)


def argmax(d):
    return max(d.items(), key=lambda x: x[1])[0]


def load_lists(path, match):
    out = {}
    for rec in map(json.loads, open(path)):
        L = []
        for s in rec["samples"]:
            d, _ = stated_dist(parse_dx(s["text"]), match)
            if d is not None:
                L.append(d)
        out[rec["case_id"]] = L
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--max-lists", type=int, default=8, help="post-trained lists per case used (speed)")
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)
    report = {}
    for item in a.pairs.split():
        d, pr = item.split(":")
        bt, pt = pr.split(">")
        B = load_lists(Path(a.runs) / d / bt / "samples.jsonl", match)
        P = load_lists(Path(a.runs) / d / pt / "samples.jsonl", match)
        cids = [c for c in cases if len(B.get(c, [])) >= 2 and len(P.get(c, [])) >= 2]
        Bm = {c: mean_dist(B[c]) for c in cids}
        by_gold = defaultdict(list)
        for c in cids:
            by_gold[cases[c]["gold"]].append(c)
        rows = {}
        for c in cids:
            ql = P[c][:a.max_lists]
            same = [x for x in by_gold[cases[c]["gold"]] if x != c]
            anyc = [cids[i] for i in rng.choice(len(cids), size=19, replace=False) if cids[i] != c]
            r = {"own": np.mean([tv(q, Bm[c]) for q in ql]),
                 "same_gold": np.mean([tv(q, Bm[x]) for q in ql for x in same]) if same else np.nan,
                 "any_case": np.mean([tv(q, Bm[x]) for q in ql for x in anyc]),
                 "ref": np.mean([tv(q, refs[c]) for q in ql])}
            bl = B[c][:a.max_lists]
            r["base_loo"] = np.mean([tv(bl[i], mean_dist(B[c][:i] + B[c][i + 1:])) for i in range(len(bl))])
            r["post_loo"] = np.mean([tv(ql[i], mean_dist(P[c][:i] + P[c][i + 1:])) for i in range(len(ql))])
            top = argmax(Bm[c])
            r["post_top_eq_basemean"] = np.mean([argmax(q) == top for q in ql])
            r["base_top_eq_basemean_loo"] = np.mean([argmax(bl[i]) == argmax(mean_dist(B[c][:i] + B[c][i + 1:])) for i in range(len(bl))])
            r["gold"] = cases[c]["gold"]
            rows[c] = r
        keys = [k for k in next(iter(rows.values())) if k != "gold"]
        groups = defaultdict(list)
        for i, c in enumerate(cids):
            groups[rows[c]["gold"]].append(i)
        garr = [np.array(g) for g in groups.values()]
        draws = [np.concatenate([garr[j] for j in rng.choice(len(garr), len(garr))]) for _ in range(a.n_boot)]
        out = {"n_cases": len(cids), "means": {k: float(np.nanmean([rows[c][k] for c in cids])) for k in keys}, "diffs": {}}
        for x, y in [("own", "same_gold"), ("own", "any_case"), ("own", "ref"), ("own", "base_loo"), ("post_top_eq_basemean", "base_top_eq_basemean_loo")]:
            v = np.array([rows[c][x] - rows[c][y] for c in cids], float)
            b = np.array([np.nanmean(v[i]) for i in draws])
            out["diffs"][f"{x}-{y}"] = {"diff": float(np.nanmean(v)), "cluster_ci": [float(q) for q in np.percentile(b, [2.5, 97.5])], "p": boot_p(b)}
        report[item] = out
        print(item, json.dumps({k: round(v, 4) for k, v in out["means"].items()}),
              json.dumps({k: [round(v["diff"], 4), [round(q, 4) for q in v["cluster_ci"]]] for k, v in out["diffs"].items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
