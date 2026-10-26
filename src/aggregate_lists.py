"""Aggregating stated lists across a few calls, against reading one list or sampling top answers (DDXPlus, CPU only).

    python aggregate_lists.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --pairs "pair_q3:qwen3-8b-base>qwen3-8b ..." --out aggregate.json

For each model, case and budget m (calls): take the first m valid samples and build
  agg     the mean of their stated lists (each renormalized over the conditions it names)
  agg5    agg truncated to its five most probable conditions and renormalized (same size as one list)
  samp    the empirical distribution of their top answers
and score each against the reference differential:
  tv, tv_nogold   total variation, also with the true pathology removed from both and renormalized
  mass            reference mass on the named conditions (for samp: Cov@m)
  size            number of named conditions
  ddf1            set F1 against the reference's support
  sev             share of severe conditions (severity <= 2) with reference >= tau that are named
  sev_any         share of severe conditions with any reference weight that are named
  top1, top3      the true pathology is the most probable / among the three most probable conditions
A random set of the same size names s/49 of the reference mass on average; we report that baseline.
Per pair: case- and condition-clustered bootstrap intervals for post-trained agg/agg5 at m=5 against the
post-trained model's single list and against the base model's 32 samples and its single list, and per severe
condition the number named less often after post-training under each readout.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from review2 import renorm_without
from stated_vs_sampled import stated_dist, tv

BUDGETS = (1, 3, 5, 10, 32)
READOUTS = ("agg", "agg5", "samp")
METRICS = ("tv", "tv_nogold", "mass", "size", "ddf1", "sev", "sev_any", "top1", "top3")


def score(d, ref, gold, support, sev_tau, sev_any):
    named = set(d)
    r = {"tv": tv(d, ref), "mass": sum(p for k, p in ref.items() if k in named), "size": len(named)}
    dng, rng_ = renorm_without(d, gold), renorm_without(ref, gold)
    r["tv_nogold"] = tv(dng, rng_) if (dng and rng_) else None
    hit = len(support & named)
    rec, prec = hit / len(support), hit / len(named)
    r["ddf1"] = 0.0 if hit == 0 else 2 * rec * prec / (rec + prec)
    r["sev"] = float(np.mean([k in named for k in sev_tau])) if sev_tau else None
    r["sev_any"] = float(np.mean([k in named for k in sev_any])) if sev_any else None
    order = sorted(d, key=lambda k: -d[k])
    r["top1"] = float(order[0] == gold)
    r["top3"] = float(gold in order[:3])
    return r


def mean_dist(ds):
    out = Counter()
    for d in ds:
        for k, v in d.items():
            out[k] += v / len(ds)
    return dict(out)


def top_k(d, k):
    t = dict(sorted(d.items(), key=lambda x: -x[1])[:k])
    z = sum(t.values())
    return {c: v / z for c, v in t.items()}


def load(path, cases, refs, severity, match, tau):
    rows = {}
    for rec in map(json.loads, open(path)):
        cid = rec["case_id"]
        ref, gold = refs[cid], cases[cid]["gold"]
        support = {k for k, p in ref.items() if p > 0}
        sev_tau = [k for k, p in ref.items() if p >= tau and severity.get(k, 9) <= 2]
        sev_any = [k for k, p in ref.items() if p > 0 and severity.get(k, 9) <= 2]
        valid = []
        for s in rec["samples"]:
            d, top = stated_dist(parse_dx(s["text"]), match)
            if d is not None:
                valid.append((d, top))
        if not valid:
            rows[cid] = None
            continue
        r = {"gold": gold, "named": {}}
        for m in BUDGETS:
            sub = valid[:m]
            agg = mean_dist([d for d, _ in sub])
            samp = Counter(t for _, t in sub)
            samp = {k: v / len(sub) for k, v in samp.items()}
            for name, dist in (("agg", agg), ("agg5", top_k(agg, 5)), ("samp", samp)):
                for k, v in score(dist, ref, gold, support, sev_tau, sev_any).items():
                    r[f"{name}@{m}:{k}"] = v
                r["named"][f"{name}@{m}"] = set(dist)
        r["sev_any_list"] = sev_any
        rows[cid] = r
    return rows


def model_means(rows):
    out = {}
    keys = [k for k in next(r for r in rows.values() if r) if ":" in k]
    for k in keys:
        v = [r[k] for r in rows.values() if r and r[k] is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


def diff_ci(xa, xb, cids, gold_of, rng, n_boot=1000):
    """Mean of xa - xb over cases with both defined; case and condition-clustered 95% intervals."""
    ok = [c for c in cids if xa.get(c) is not None and xb.get(c) is not None]
    d = np.array([xa[c] - xb[c] for c in ok])
    if not len(d):
        return None
    case = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    by = defaultdict(list)
    for c, v in zip(ok, d):
        by[gold_of[c]].append(v)
    groups = [np.array(v) for v in by.values()]
    clus = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        clus.append(np.concatenate([groups[i] for i in pick]).mean())
    return {"diff": float(d.mean()), "case_ci": [float(x) for x in np.percentile(case, [2.5, 97.5])],
            "cluster_ci": [float(x) for x in np.percentile(clus, [2.5, 97.5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--pairs", required=True, help='space-separated "dir:base>post"; dir may be absolute')
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    match = Matcher(sorted(severity))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    gold_of = {cid: c["gold"] for cid, c in cases.items()}
    severe = sorted(k for k, s in severity.items() if s <= 2)
    rng = np.random.default_rng(0)
    report = {"budgets": BUDGETS, "n_conditions": len(severity), "models": {}, "pairs": {}}
    for item in a.pairs.split():
        d, pr = item.split(":")
        bt, pt = pr.split(">")
        rows = {}
        for t in (bt, pt):
            key = f"{d}/{t}"
            rows[t] = load(Path(a.runs) / d / t / "samples.jsonl", cases, refs, severity, match, a.tau)
            if key not in report["models"]:
                mm = model_means(rows[t])
                for m in BUDGETS:
                    mm[f"random_same_size@{m}:mass"] = mm[f"agg@{m}:size"] / len(severity)
                report["models"][key] = mm
                print(key, " ".join(f"{r}@{m}:tv={mm[f'{r}@{m}:tv']:.3f},mass={mm[f'{r}@{m}:mass']:.3f},size={mm[f'{r}@{m}:size']:.1f},"
                                    f"sev={mm[f'{r}@{m}:sev']:.3f},ddf1={mm[f'{r}@{m}:ddf1']:.3f},top1={mm[f'{r}@{m}:top1']:.3f}"
                                    for m in (1, 5, 32) for r in READOUTS), flush=True)
        B, P = rows[bt], rows[pt]
        cids = [c for c in cases if B.get(c) and P.get(c)]
        col = lambda R, k: {c: R[c][k] for c in cids}
        comps = {}
        for met in ("tv", "tv_nogold", "mass", "ddf1", "sev", "sev_any", "top1", "top3"):
            for r in ("agg", "agg5"):
                comps[f"post_{r}@5-post_list:{met}"] = diff_ci(col(P, f"{r}@5:{met}"), col(P, f"agg@1:{met}"), cids, gold_of, rng, a.n_boot)
                comps[f"post_{r}@5-base_samp@32:{met}"] = diff_ci(col(P, f"{r}@5:{met}"), col(B, f"samp@32:{met}"), cids, gold_of, rng, a.n_boot)
                comps[f"post_{r}@5-base_list:{met}"] = diff_ci(col(P, f"{r}@5:{met}"), col(B, f"agg@1:{met}"), cids, gold_of, rng, a.n_boot)
                comps[f"post_{r}@5-base_{r}@5:{met}"] = diff_ci(col(P, f"{r}@5:{met}"), col(B, f"{r}@5:{met}"), cids, gold_of, rng, a.n_boot)
        # severe conditions named less often after post-training, threshold-free, per readout
        lower = {}
        for b_ro, p_ro in (("agg@1", "agg@1"), ("agg@1", "agg5@5"), ("agg@1", "agg@5"), ("agg5@5", "agg5@5"), ("samp@32", "agg@5")):
            n_lower, n_cond = 0, 0
            for k in severe:
                elig = [c for c in cids if k in B[c]["sev_any_list"]]
                if not elig:
                    continue
                n_cond += 1
                b = np.mean([k in B[c]["named"][b_ro] for c in elig])
                p = np.mean([k in P[c]["named"][p_ro] for c in elig])
                n_lower += p < b
            lower[f"base_{b_ro}>post_{p_ro}"] = [int(n_lower), n_cond]
        report["pairs"][item] = {"comparisons": comps, "severe_lower": lower}
        print(item, json.dumps({k: round(v["diff"], 4) for k, v in comps.items() if v and k.endswith((":tv", ":mass", ":sev", ":sev_any", ":ddf1", ":top1"))}), flush=True)
        print("  severe named less often (lower, of):", json.dumps(lower), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
