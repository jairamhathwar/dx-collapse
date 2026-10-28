"""Recovery experiment: does a seeded pick from the model's own list restore sampled breadth? (DDXPlus, CPU only)

    python seeded_an.py --cases cases_test.jsonl --conditions-json release_conditions.json \
        --spec "qwen3-8b=SEEDED_RUN:DEFAULT_RUN:BASE_RUN" ... --out seeded.json

Each seeded call (sample_diagnoses.py --prompt-variant seeded) states a list and then picks one condition with a
supplied random number u (32 stratified numbers per case). Per model:
  follows     share of calls whose pick is the condition the cumulative rule selects from the call's own top five
  pick_*      the 32 picks as the sampled answers (invalid or missing picks count as wrong for pass@k)
  rule_*      the rule applied by us to each call's list (what an exact follower would give)
  list_*      the stated list of each call (recall@5, TV to the reference): does the extra instruction change it?
  default_*   the same readouts from the model's ordinary zero-shot samples, and base_* from its base model's
Agreement AUROC: correctness of the first call's list top item, scored by the share of the other 31 picks naming it.
Differences (seeded minus default, seeded minus base) with condition-clustered 95% intervals.
"""
import argparse
import json

import numpy as np

from consequences import auroc
from ddx_names import Matcher
from r8_controls import cdiff, load, parse, pass_at_k
from stated_vs_sampled import tv

def rule_pick(dist, u):
    items = list(dist.items())[:5]
    z = sum(p for _, p in items)
    run = 0.0
    for k, p in items:
        run += p / z
        if run >= u - 1e-9:
            return k
    return items[-1][0]


def seeded_rows(path, match):
    out = {}
    for rec in map(json.loads, open(path)):
        rows = []
        for s in rec["samples"]:
            lst = parse(s["text"], match)
            pick = match(s["pick"]) if s.get("pick") else None
            rows.append({"list": lst, "pick": pick, "u": s["u"],
                         "rule": rule_pick(lst["dist"], s["u"]) if lst else None})
        out[rec["case_id"]] = rows
    return out


def readouts(tops, gold, n):
    """tops: answers (None = invalid) for one case; n = number of calls"""
    v = [t for t in tops if t]
    c = sum(t == gold for t in v)
    return {"pass1": pass_at_k(n, c, 1), "pass32": pass_at_k(n, c, min(32, n)),
            "distinct": float(len(set(v))) if v else None}


def auroc_diff(A, B, cids, gold, rng, n_boot):
    """AUROC(A) - AUROC(B) on common cases; A, B: {case: (score, correct)}; condition-clustered bootstrap"""
    by = {}
    for i, c in enumerate(cids):
        by.setdefault(gold[c], []).append(i)
    groups = [np.array(v) for v in by.values()]
    sa, ya = np.array([A[c][0] for c in cids]), np.array([A[c][1] for c in cids], float)
    sb, yb = np.array([B[c][0] for c in cids]), np.array([B[c][1] for c in cids], float)
    f = lambda ix: auroc(sa[ix], ya[ix]) - auroc(sb[ix], yb[ix])
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[j] for j in rng.choice(len(groups), len(groups))])
        if 0 < ya[ix].sum() < len(ix) and 0 < yb[ix].sum() < len(ix):
            bs.append(f(ix))
    bs = np.array(bs)
    p = float(min(1.0, 2 * (1 + min((bs <= 0).sum(), (bs >= 0).sum())) / (len(bs) + 1)))
    return {"diff": float(f(np.arange(len(cids)))), "cluster_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "p": p, "n": len(cids)}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="tag=seeded.jsonl:default.jsonl[:base.jsonl]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
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
    report = {"models": {}, "diffs": {}}
    for spec in a.spec:
        tag, paths = spec.split("=")
        paths = paths.split(":")
        S = seeded_rows(paths[0], match)
        D = load(paths[1], match)
        B = load(paths[2], match) if len(paths) > 2 else None
        cids = [c for c in S if c in gold]
        per = {k: {} for k in ["pick", "rule", "default", "base"]}
        extra = {"follows": [], "pick_in_list": [], "pick_present": [], "list_valid": [], "list_rec5": [], "list_tv": [],
                 "default_rec5": [], "default_tv": []}
        sig = {"pick": ([], []), "default": ([], []), "base": ([], [])}
        cids_sig = {"pick": [], "default": [], "base": []}
        for c in cids:
            rows, n = S[c], len(S[c])
            per["pick"][c] = readouts([r["pick"] for r in rows], gold[c], n)
            per["rule"][c] = readouts([r["rule"] for r in rows], gold[c], n)
            dr = D.get(c, [])
            per["default"][c] = readouts([x["top"] if x else None for x in dr], gold[c], len(dr)) if dr else {}
            if B is not None and B.get(c):
                per["base"][c] = readouts([x["top"] if x else None for x in B[c]], gold[c], len(B[c]))
            for r in rows:
                extra["pick_present"].append(r["pick"] is not None)
                extra["list_valid"].append(r["list"] is not None)
                if r["list"]:
                    extra["list_rec5"].append(gold[c] in list(r["list"]["dist"])[:5])
                    extra["list_tv"].append(tv(r["list"]["dist"], refs[c]))
                    if r["pick"]:
                        extra["follows"].append(r["pick"] == r["rule"])
                        extra["pick_in_list"].append(r["pick"] in list(r["list"]["dist"])[:5])
            for x in dr:
                if x:
                    extra["default_rec5"].append(gold[c] in list(x["dist"])[:5])
                    extra["default_tv"].append(tv(x["dist"], refs[c]))
            # agreement signal for the single-call answer (first valid list's top / first valid default sample)
            first = next((r["list"]["top"] for r in rows if r["list"]), None)
            if first:
                others = [r["pick"] for r in rows[1:] if r["pick"]]
                if others:
                    sig["pick"][0].append(np.mean([p == first for p in others]))
                    sig["pick"][1].append(first == gold[c])
                    cids_sig["pick"].append(c)
            for nm, src in [("default", dr), ("base", B.get(c, []) if B else [])]:
                tops = [x["top"] for x in src if x]
                if len(tops) > 1:
                    sig[nm][0].append(np.mean([t == tops[0] for t in tops[1:]]))
                    sig[nm][1].append(tops[0] == gold[c])
                    cids_sig[nm].append(c)
        m = {"n_cases": len(cids)}
        for nm in per:
            for k in ["pass1", "pass32", "distinct"]:
                vals = [r[k] for r in per[nm].values() if r.get(k) is not None]
                if vals:
                    m[f"{nm}_{k}"] = float(np.mean(vals))
        for k, v in extra.items():
            if v:
                m[k] = float(np.mean(v))
        for nm, (s, y) in sig.items():
            if len(set(y)) == 2:
                m[f"{nm}_auroc_agree"] = auroc(np.array(s), np.array(y, float))
        report["models"][tag] = m
        print(tag, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
        dd = {}
        for k in ["pass1", "pass32", "distinct"]:
            for other in ["default", "base"]:
                x = {c: per["pick"][c].get(k) for c in cids}
                y = {c: per[other].get(c, {}).get(k) for c in cids}
                if any(v is not None for v in y.values()):
                    r = cdiff(x, y, cids, gold, rng, a.n_boot)
                    if r:
                        dd[f"seeded-{other}_{k}"] = r
        # agreement AUROC: seeded picks vs the ordinary samples' agreement (and the base model's), clustered
        sc = {nm: {c: (a_, y_) for c, a_, y_ in zip(cids_sig[nm], *sig[nm])} for nm in sig}
        for other in ["default", "base"]:
            common = [c for c in sc["pick"] if c in sc[other]]
            if len(common) > 50:
                dd[f"seeded-{other}_auroc_agree"] = auroc_diff(sc["pick"], sc[other], common, gold, rng, a.n_boot)
        report["diffs"][tag] = dd
        print(tag, json.dumps(dd), flush=True)
    tests = [(t, k) for t, d in report["diffs"].items() for k in d]
    for (t, k), p in zip(tests, holm([report["diffs"][t][k]["p"] for t, k in tests])):
        report["diffs"][t][k]["p_holm"] = p
    print("Holm over", len(tests), "tests", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
