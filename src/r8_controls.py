"""Controls requested in the fifth review round (DDXPlus, CPU only).

    python r8_controls.py --spec spec.json --cases cases_test.jsonl --conditions-json release_conditions.json --out r8.json

spec.json: {"models": {name: path_to_samples.jsonl}, "families": {name: family},
            "pairs": [[base, post], ...], "fewshot_pairs": [[base, post], ...], "defaults": [post, ...]}

1. Cross-family packing: for each post-trained model P with base B, the TV between each P list and the mean of the
   lists of every zero-shot base model for the same case; own base vs base models of the other family.
2. Severe true pathology: recall@1 and recall@5 of the stated list and sampled pass@1 / pass@32 on cases whose true
   pathology has DDXPlus severity 1-2, base vs post-trained, with intervals clustered over the 17 pathologies.
3. Length-matched must-not-miss control: the union of a post-trained model's first two valid standard lists
   (warranted severe named, unwarranted severe named, length).
4. Complete lists: recall@5, reference mass named, warranted severe named and DDF1 on lists naming exactly five
   listed conditions with valid probabilities, base vs post-trained.
"""
import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from review2 import boot_p
from stated_vs_sampled import tv


def parse(text, match):
    dx = parse_dx(text)
    names = [match(d["diagnosis"]) for d in dx]
    pairs = [(n, d["probability"]) for n, d in zip(names, dx) if n]
    if not pairs:
        return None
    probs = [p for _, p in pairs]
    ok = all(p is not None and 0 <= p <= 1 for p in probs) and sum(probs) > 0
    if not ok:
        probs = [1.0] * len(pairs)
    d = {}
    for (n, _), p in zip(pairs, probs):
        d[n] = d.get(n, 0) + p
    z = sum(d.values())
    d = {k: v / z for k, v in d.items()}
    return {"dist": d, "top": pairs[0][0], "complete": len(d) == 5 and ok}


def load(path, match):
    out = {}
    for rec in map(json.loads, open(path)):
        out[rec["case_id"]] = [parse(s["text"], match) for s in rec["samples"]]
    return out


def mean_dist(ds):
    m = Counter()
    for d in ds:
        for k, v in d.items():
            m[k] += v / len(ds)
    return dict(m)


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def cdiff(a, b, cids, gold, rng, n_boot):
    ok = [c for c in cids if a.get(c) is not None and b.get(c) is not None]
    if not ok:
        return None
    v = np.array([a[c] - b[c] for c in ok])
    g = defaultdict(list)
    for i, c in enumerate(ok):
        g[gold[c]].append(i)
    ga = [np.array(x) for x in g.values()]
    bs = np.array([v[np.concatenate([ga[j] for j in rng.choice(len(ga), len(ga))])].mean() for _ in range(n_boot)])
    return {"diff": float(v.mean()), "cluster_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "p": boot_p(bs), "n": len(ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--max-lists", type=int, default=8)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    match = Matcher(sorted(severity))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    gold = {cid: c["gold"] for cid, c in cases.items()}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)
    M = {name: load(path, match) for name, path in spec["models"].items()}
    fam = spec["families"]
    report = {}

    # 1. cross-family packing
    bases = [b for b, _ in spec["pairs"]]
    bmean = {b: {c: mean_dist([x["dist"] for x in M[b][c] if x]) for c in cases if any(M[b][c])} for b in bases}
    pk = {}
    for b, p in spec["pairs"]:
        rows = {}
        for c in cases:
            ql = [x["dist"] for x in M[p].get(c, []) if x][:a.max_lists]
            if not ql or c not in bmean[b]:
                continue
            r = {"own": float(np.mean([tv(q, bmean[b][c]) for q in ql]))}
            other = [o for o in bases if fam[o] != fam[b] and c in bmean[o]]
            same = [o for o in bases if fam[o] == fam[b] and o != b and c in bmean[o]]
            if other:
                r["other_family"] = float(np.mean([tv(q, bmean[o][c]) for q in ql for o in other]))
            if same:
                r["same_family"] = float(np.mean([tv(q, bmean[o][c]) for q in ql for o in same]))
            # base-to-base benchmark: own base lists vs other-family base means
            bl = [x["dist"] for x in M[b][c] if x][:a.max_lists]
            if other and bl:
                r["base_other_family"] = float(np.mean([tv(q, bmean[o][c]) for q in bl for o in other]))
            rows[c] = r
        cids = list(rows)
        col = lambda k: {c: rows[c].get(k) for c in cids}
        out = {"means": {k: float(np.mean([r[k] for r in rows.values() if k in r])) for k in ["own", "other_family", "same_family", "base_other_family"]
                         if any(k in r for r in rows.values())}}
        out["own-other_family"] = cdiff(col("own"), col("other_family"), cids, gold, rng, a.n_boot)
        if any("same_family" in r for r in rows.values()):
            out["own-same_family"] = cdiff(col("own"), col("same_family"), cids, gold, rng, a.n_boot)
        pk[f"{b}>{p}"] = out
        print("packing", b, p, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["means"].items()}),
              json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["cluster_ci"]]] for k, v in out.items() if k != "means" and v}), flush=True)
    report["packing_cross_family"] = pk

    # 2. severe true pathology
    def case_stats(rows, c):
        valid = [x for x in rows if x]
        if not valid:
            return None
        n, k = len(rows), sum(1 for x in rows if x and x["top"] == gold[c])
        order = [sorted(x["dist"], key=lambda q: -x["dist"][q]) for x in valid]
        return {"rec1": float(np.mean([o[0] == gold[c] for o in order])), "rec5": float(np.mean([gold[c] in o[:5] for o in order])),
                "pass1": pass_at_k(n, k, 1), "pass32": pass_at_k(n, k, 32)}
    sv = {}
    for grp in ["pairs", "fewshot_pairs"]:
        for b, p in spec.get(grp, []):
            for stratum, sel in [("severe", lambda c: severity.get(gold[c], 9) <= 2), ("other", lambda c: severity.get(gold[c], 9) > 2)]:
                cids = [c for c in cases if sel(c)]
                SB = {c: case_stats(M[b][c], c) for c in cids if c in M[b]}
                SP = {c: case_stats(M[p][c], c) for c in cids if c in M[p]}
                cc = [c for c in cids if SB.get(c) and SP.get(c)]
                out = {"n_cases": len(cc)}
                for k in ["rec1", "rec5", "pass1", "pass32"]:
                    out[f"base_{k}"] = float(np.mean([SB[c][k] for c in cc if SB[c][k] is not None]))
                    out[f"post_{k}"] = float(np.mean([SP[c][k] for c in cc if SP[c][k] is not None]))
                    out[f"d_{k}"] = cdiff({c: SP[c][k] for c in cc}, {c: SB[c][k] for c in cc}, cc, gold, rng, a.n_boot)
                sv[f"{grp}:{b}>{p}:{stratum}"] = out
                print("severe-true", grp, b, p, stratum, json.dumps({k: round(v, 3) for k, v in out.items() if isinstance(v, float)}),
                      json.dumps({k: [round(v["diff"], 3), [round(x, 3) for x in v["cluster_ci"]]] for k, v in out.items() if k.startswith("d_") and v}), flush=True)
    report["severe_true"] = sv

    # 3. length-matched must-not-miss control: union of the first two valid standard lists
    lm = {}
    for p in spec.get("defaults", []):
        sevn, unw, ln = [], [], []
        for c in cases:
            valid = [x for x in M[p].get(c, []) if x]
            if len(valid) < 2:
                continue
            ref = refs[c]
            warranted = [k for k, q in ref.items() if q >= a.tau and severity.get(k, 9) <= 2]
            u = set(valid[0]["dist"]) | set(valid[1]["dist"])
            ln.append(len(u))
            unw.append(sum(1 for k in u if severity.get(k, 9) <= 2 and ref.get(k, 0) < a.tau))
            if warranted:
                sevn.append(np.mean([k in u for k in warranted]))
        lm[p] = {"len": float(np.mean(ln)), "sev_named": float(np.mean(sevn)), "unwarranted": float(np.mean(unw))}
        print("length-matched", p, json.dumps({k: round(v, 3) for k, v in lm[p].items()}), flush=True)
    report["length_matched_union2"] = lm

    # 4. complete lists only
    cl = {}
    for b, p in spec["pairs"]:
        def stats(rows, c):
            comp = [x["dist"] for x in rows if x and x["complete"]]
            if not comp:
                return None
            ref = refs[c]
            warranted = [k for k, q in ref.items() if q >= a.tau and severity.get(k, 9) <= 2]
            support = {k for k, q in ref.items() if q > 0}
            f1 = []
            for d in comp:
                h = len(support & set(d))
                f1.append(0.0 if h == 0 else 2 * (h / len(support)) * (h / len(d)) / (h / len(support) + h / len(d)))
            return {"rec5": float(np.mean([gold[c] in d for d in comp])), "mass": float(np.mean([sum(ref.get(k, 0) for k in d) for d in comp])),
                    "sev": float(np.mean([np.mean([k in d for k in warranted]) for d in comp])) if warranted else None,
                    "ddf1": float(np.mean(f1)), "tv": float(np.mean([tv(d, ref) for d in comp]))}
        SB = {c: stats(M[b][c], c) for c in cases if c in M[b]}
        SP = {c: stats(M[p][c], c) for c in cases if c in M[p]}
        cc = [c for c in cases if SB.get(c) and SP.get(c)]
        out = {"n_cases": len(cc)}
        for k in ["rec5", "mass", "sev", "ddf1", "tv"]:
            out[k] = cdiff({c: SP[c][k] for c in cc}, {c: SB[c][k] for c in cc}, cc, gold, rng, a.n_boot)
        cl[f"{b}>{p}"] = out
        print("complete-only", b, p, len(cc), json.dumps({k: [round(v["diff"], 3), [round(x, 3) for x in v["cluster_ci"]]] for k, v in out.items() if isinstance(v, dict) and v}), flush=True)
    report["complete_lists"] = cl
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
