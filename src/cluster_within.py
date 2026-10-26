"""Condition-clustered intervals for within-model comparisons quoted in the main text (DDXPlus, CPU only).

    python cluster_within.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --items "pair_q3:qwen3-8b-base,qwen3-8b ..." --out within.json

Per model (rows as in consequences.py; valid samples only):
  vote32 - single            majority answer over all valid samples vs the single-call answer
  top3_list1 - top3_vote@32  one list's top three vs the three most frequent sampled answers
  auroc(stated_p) - auroc(agree@32), auroc(stated_p) - auroc(agree@5)
Intervals resample the 49 true pathologies with all their cases; p-values use the add-one correction.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from consequences import auroc, case_row
from ddx_names import Matcher
from parsing import parse_dx
from review2 import boot_p
from stated_vs_sampled import entropy, stated_dist


def load(path, cases, refs, match):
    rows = []
    for rec in map(json.loads, open(path)):
        c = cases[rec["case_id"]]
        tops, lists = [], []
        for s in rec["samples"]:
            d, top = stated_dist(parse_dx(s["text"]), match)
            if d is not None and top is not None:
                tops.append(top)
                lists.append(d)
        r = case_row(tops, lists, c["gold"], entropy(refs[rec["case_id"]]))
        r["gold"] = c["gold"]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)
    out = {}
    for item in a.items.split():
        d, tags = item.split(":")
        for t in tags.split(","):
            rows = load(Path(a.runs) / d / t / "samples.jsonl", cases, refs, match)
            groups = defaultdict(list)
            for i, r in enumerate(rows):
                groups[r["gold"]].append(i)
            garr = [np.array(g) for g in groups.values()]
            draws = [np.concatenate([garr[j] for j in rng.choice(len(garr), len(garr))]) for _ in range(a.n_boot)]
            res = {}
            for name, x, y in [("vote32-single", "vote@32", "single"), ("top3list-top3vote32", "top3_list1", "top3_vote@32")]:
                v = np.array([r[x] - r[y] for r in rows])
                b = np.array([v[i].mean() for i in draws])
                res[name] = {"diff": float(v.mean()), "cluster_ci": [float(q) for q in np.percentile(b, [2.5, 97.5])], "p": boot_p(b)}
            ok = np.array([i for i, r in enumerate(rows) if r.get("consist@32") is not None])
            pos = np.full(len(rows), -1)
            pos[ok] = np.arange(len(ok))
            y = np.array([rows[i]["A_correct"] for i in ok])
            sp = np.array([rows[i]["stated_p"] for i in ok])
            for other in ("consist@32", "consist@5"):
                ag = np.array([rows[i][other] for i in ok])
                f = lambda idx: auroc(sp[idx], y[idx]) - auroc(ag[idx], y[idx])
                b = []
                for i in draws:
                    j = pos[i]
                    b.append(f(j[j >= 0]))
                b = np.array(b)
                res[f"auroc_stated-{other}"] = {"diff": float(f(np.arange(len(ok)))), "cluster_ci": [float(q) for q in np.nanpercentile(b, [2.5, 97.5])],
                                                 "p": boot_p(b[~np.isnan(b)])}
            out[f"{d}/{t}"] = res
            print(f"{d}/{t}", json.dumps({k: [round(v['diff'], 4), [round(q, 4) for q in v['cluster_ci']], round(v['p'], 4)] for k, v in res.items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
