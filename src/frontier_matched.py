"""Matched-accuracy comparison along the temperature frontier (DDXPlus, CPU only).

    python frontier_matched.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --temps 04 07 10 13 --pairs "qwen3-8b-base>qwen3-8b ..." --out frontier_matched.json

Each model was sampled with worked examples and top-p 1 at the listed temperatures (runs/frontier_t<T>/<tag>). For
each post-trained point (its pass@1 and pass@32 at one temperature) we interpolate the base model's pass@32 at the
same pass@1 along the base model's own frontier (linear in pass@1 between its temperature points); the matched
difference is post-trained pass@32 minus that value, undefined when the post-trained pass@1 lies outside the base
model's range. Intervals: condition-clustered bootstrap (one draw of the 49 pathologies shared by all models and
temperatures; 5,000 resamples); draws where the point falls outside the base range are dropped and counted.
"""
import argparse
import json
import math

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(path, gold, match):
    out = {}
    for rec in map(json.loads, open(path)):
        cid = rec["case_id"]
        tops = [stated_dist(parse_dx(s["text"]), match)[1] for s in rec["samples"]]
        n, c = len(tops), sum(t == gold[cid] for t in tops)
        out[cid] = (pass_at_k(n, c, 1), pass_at_k(n, c, min(32, n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "cases", "conditions-json", "pairs", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--temps", nargs="+", default=["04", "07", "10", "13"])
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = [json.loads(l) for l in open(a.cases)]
    gold = {c["case_id"]: c["gold"] for c in cases}
    keys = sorted(set(gold.values()))
    idx = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(keys), [1 / len(keys)] * len(keys), size=a.n_boot).astype(float)
    pairs = [p.split(">") for p in a.pairs.split()]
    tags = sorted({t for p in pairs for t in p})
    pts = {}  # (tag, temp) -> (point pass1, point pass32, boot pass1 [B], boot pass32 [B])
    for t in tags:
        for T in a.temps:
            R = load(f"{a.runs}/frontier_t{T}/{t}/samples.jsonl", gold, match)
            S1, S32, N = np.zeros(len(keys)), np.zeros(len(keys)), np.zeros(len(keys))
            for cid, (p1, p32) in R.items():
                i = idx[gold[cid]]
                S1[i] += p1
                S32[i] += p32
                N[i] += 1
            pts[(t, T)] = (S1.sum() / N.sum(), S32.sum() / N.sum(), (W @ S1) / (W @ N), (W @ S32) / (W @ N))
            print(t, T, round(pts[(t, T)][0], 4), round(pts[(t, T)][1], 4), flush=True)
    report = {"points": {f"{t}@{T}": {"pass1": v[0], "pass32": v[1]} for (t, T), v in pts.items()}, "matched": {}}

    def interp(x, xs, ys):
        o = np.argsort(xs)
        xs, ys = np.asarray(xs)[o], np.asarray(ys)[o]
        return float(np.interp(x, xs, ys)) if xs[0] <= x <= xs[-1] else float("nan")

    for b, p in pairs:
        res = {}
        for T in a.temps:
            x = pts[(p, T)][0]
            point = pts[(p, T)][1] - interp(x, [pts[(b, u)][0] for u in a.temps], [pts[(b, u)][1] for u in a.temps])
            bs = np.array([pts[(p, T)][3][j] - interp(pts[(p, T)][2][j], [pts[(b, u)][2][j] for u in a.temps],
                                                       [pts[(b, u)][3][j] for u in a.temps]) for j in range(a.n_boot)])
            ok = bs[~np.isnan(bs)]
            res[T] = {"post_pass1": x, "post_pass32": pts[(p, T)][1], "matched_diff": point,
                      "ci": [float(v) for v in np.percentile(ok, [2.5, 97.5])] if len(ok) > 100 else None,
                      "share_draws_in_range": float(len(ok) / a.n_boot)}
        report["matched"][f"{b}>{p}"] = res
        print(f"{b}>{p}", json.dumps({T: [round(v["post_pass1"], 3), None if math.isnan(v["matched_diff"]) else round(v["matched_diff"], 4),
                                          v["ci"] and [round(c, 4) for c in v["ci"]], round(v["share_draws_in_range"], 2)] for T, v in res.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
