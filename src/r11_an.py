"""Seventh-round checks requested by reviewers (DDXPlus, CPU only).

    python r11_an.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json --out r11.json

1. Stricter packing test. For each zero-shot T=0.7 post-trained model P and each other zero-shot model X (the five base
   and five post-trained models of the final pairs), the mean TV between P's lists and X's mean list for the same
   patient. Reports P's distance to its own base mean against the closest other base model, the closest other
   post-trained model and the mean over the other post-trained models, with condition-clustered intervals, and each
   mean list's own TV to the reference.
2. Patient-shuffled pass@32. Each model's pass@32 when every case is scored against the true pathology of another case
   with a different true pathology (20 random permutations), so the gap to the real pass@32 is the part of the
   sampled readout that depends on the patient.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

from ddx_names import Matcher
from r8_controls import load, mean_dist, cdiff
from stated_vs_sampled import tv

PAIRS = [("pair_q3", "qwen3-8b-base", "qwen3-8b"), ("pair25_t07", "qwen25-7b-base", "qwen25-7b-it"),
         ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it"), ("ddx_olmo", "olmo3-base", "olmo3-instruct"),
         ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct")]


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--max-lists", type=int, default=8)
    ap.add_argument("--shuffle-only", help='only the shuffled pass@32, for these runs: "dir:tag,tag dir:tag,..."')
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
    if a.shuffle_only:
        tags = [(d, t) for item in a.shuffle_only.split() for d in [item.split(":")[0]] for t in item.split(":")[1].split(",")]
    else:
        tags = [(d, b) for d, b, _ in PAIRS] + [(d, p) for d, _, p in PAIRS]
    M = {t: load(f"{a.runs}/{t[0]}/{t[1]}/samples.jsonl", match) for t in tags}
    means = {} if a.shuffle_only else {t: {c: mean_dist([x["dist"] for x in M[t][c] if x]) for c in cases if any(M[t].get(c, []))} for t in tags}
    report = {"packing_strict": {}, "shuffled_pass32": {}}

    # 1. stricter packing
    for d, b, p in ([] if a.shuffle_only else PAIRS):
        P = (d, p)
        rows = {}
        for c in cases:
            ql = [x["dist"] for x in M[P].get(c, []) if x][:a.max_lists]
            if not ql:
                continue
            rows[c] = {f"{t[0]}/{t[1]}": float(np.mean([tv(q, means[t][c]) for q in ql])) for t in tags if t != P and c in means[t]}
        own = f"{d}/{b}"
        others_base = [f"{t[0]}/{t[1]}" for t in tags[:5] if f"{t[0]}/{t[1]}" != own]
        others_post = [f"{t[0]}/{t[1]}" for t in tags[5:] if t != P]
        cids = [c for c in rows if own in rows[c] and all(k in rows[c] for k in others_base + others_post)]
        avg = {k: float(np.mean([rows[c][k] for c in cids])) for k in [own] + others_base + others_post}
        nb = min(others_base, key=lambda k: avg[k])
        npst = min(others_post, key=lambda k: avg[k])
        col = lambda f: {c: f(c) for c in cids}
        out = {"n": len(cids), "mean_tv_to": avg,
               "ref_tv_of_mean": {f"{t[0]}/{t[1]}": float(np.mean([tv(means[t][c], refs[c]) for c in cids if c in means[t]])) for t in tags},
               "closest_other_base": nb, "closest_other_post": npst,
               "own-closest_other_base": cdiff(col(lambda c: rows[c][own]), col(lambda c: rows[c][nb]), cids, gold, rng, a.n_boot),
               "own-closest_other_post": cdiff(col(lambda c: rows[c][own]), col(lambda c: rows[c][npst]), cids, gold, rng, a.n_boot),
               "own-mean_other_post": cdiff(col(lambda c: rows[c][own]), col(lambda c: float(np.mean([rows[c][k] for k in others_post]))), cids, gold, rng, a.n_boot)}
        report["packing_strict"][f"{d}:{b}>{p}"] = out
        print("packing", p, json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["cluster_ci"]]] for k, v in out.items() if isinstance(v, dict) and "diff" in v}),
              "closest base", nb, "closest post", npst, flush=True)

    # 2. patient-shuffled pass@32
    cl = list(cases)
    for t in tags:
        def p32(lab):
            vals = []
            for c in cl:
                rows = M[t].get(c, [])
                n = len(rows)
                k = sum(1 for x in rows if x and x["top"] == lab[c])
                v = pass_at_k(n, k, 32)
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals))
        real = p32(gold)
        sh = []
        for _ in range(20):
            perm = list(rng.permutation(cl))
            # re-draw until no case keeps a label of its own pathology class
            lab = {}
            for c, c2 in zip(cl, perm):
                lab[c] = gold[c2]
            bad = [c for c in cl if lab[c] == gold[c]]
            for c in bad:
                while lab[c] == gold[c]:
                    lab[c] = gold[cl[rng.integers(len(cl))]]
            sh.append(p32(lab))
        report["shuffled_pass32"][f"{t[0]}/{t[1]}"] = {"real": real, "shuffled": float(np.mean(sh)), "shuffled_sd": float(np.std(sh)),
                                                        "patient_dependent": real - float(np.mean(sh))}
        print("shuffled", t, round(real, 4), round(float(np.mean(sh)), 4), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
