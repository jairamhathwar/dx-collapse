"""Stated vs sampled readouts on published case reports (MedCaseReasoning test, 897 cases), graded by a judge model.

    python medcase_analysis.py --cases cases_test.jsonl --runs RUNS/medcase_t07 --judge judge.jsonl \
        --pairs "qwen3-8b-base>qwen3-8b ..." [--judge2 judge2.jsonl] --out medcase.json

There is no reference differential, so only readouts that need the true diagnosis are available. A name is correct if
the judge says it names the final diagnosis (judge_medcase.py). Per model, over the 32 samples of each case:
  valid            share of samples with a parsable list
  pass1, pass32    unbiased pass@k of the top answer (invalid samples count as wrong, as on DDXPlus)
  distinct         distinct top answers among valid samples (correct answers merged into one; others by normalized text)
  rec1, rec5       one stated list: share of valid lists whose first item / any of the first five items is correct
  vote32 - single  majority top answer over valid samples vs the first valid sample's top answer
  AUROC for the first valid sample's top answer being correct, of its stated probability (renormalized over the
  list) and of agreement among the other valid samples (agree5: next 4, agree32: all)
Pairs: post-trained minus base, paired case bootstrap; Holm over all pair tests. By publication year: pass1, pass32
and rec5 for cases published up to 2022 and from 2023.
--judge2: agreement (Cohen's kappa) between the two judges on the pairs both graded; with --strict a name is correct
only if both judges say so.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from consequences import auroc
from disease_normalizer import norm_text
from parsing import parse_dx
from review2 import boot_p

def sample_files(d):
    """the merged samples.jsonl if present, else the shards samples.s*.jsonl"""
    d = Path(d)
    return [d / "samples.jsonl"] if (d / "samples.jsonl").exists() else sorted(d.glob("samples.s*.jsonl"))


METRICS = ["valid", "pass1", "pass32", "distinct", "rec1", "rec5"]


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def load(path, J):
    rows = {}
    for rec in (json.loads(l) for f in sample_files(path) for l in open(f)):
        cid, samples = rec["case_id"], []
        for s in rec["samples"]:
            dx = parse_dx(s["text"])
            if not dx:
                samples.append(None)
                continue
            ok = [J.get((cid, norm_text(d["diagnosis"])), False) for d in dx]
            ps = [d["probability"] for d in dx]
            if any(p is None or not (0 <= p <= 1) for p in ps) or sum(ps) <= 0:
                ps = [1.0] * len(dx)
            # list order as given (the prompt asks for most likely first)
            key = "GOLD" if ok[0] else "txt:" + norm_text(dx[0]["diagnosis"])
            samples.append({"key": key, "ok": ok, "p": ps[0] / sum(ps)})
        rows[cid] = samples
    return rows


def case_row(samples):
    v = [s for s in samples if s]
    n, c = len(samples), sum(1 for s in v if s["ok"][0])
    r = {"valid": len(v) / n, "pass1": pass_at_k(n, c, 1), "pass32": pass_at_k(n, c, min(32, n))}
    if not v:
        r.update({"distinct": None, "rec1": None, "rec5": None})
        return r, None
    keys = [s["key"] for s in v]
    r["distinct"] = float(len(set(keys)))
    r["rec1"] = float(np.mean([s["ok"][0] for s in v]))
    r["rec5"] = float(np.mean([any(s["ok"][:5]) for s in v]))
    A = keys[0]
    sig = {"single": float(v[0]["ok"][0]), "vote32": float(Counter(keys).most_common(1)[0][0] == "GOLD"),
           "stated_p": v[0]["p"],
           "agree5": float(np.mean([k == A for k in keys[1:5]])) if len(keys) > 1 else None,
           "agree32": float(np.mean([k == A for k in keys[1:]])) if len(keys) > 1 else None}
    return r, sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--judge2")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--strict", action="store_true", help="a name is correct only if both judges say so")
    a = ap.parse_args()
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    J = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge))}
    rng = np.random.default_rng(0)
    report = {"n_judged": len(J), "judge_match_rate": float(np.mean(list(J.values()))), "models": {}, "pairs": {}}
    if a.judge2:
        J2 = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge2))}
        both = [k for k in J2 if k in J]
        x, y = np.array([J[k] for k in both]), np.array([J2[k] for k in both])
        po = float((x == y).mean())
        pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
        report["judge_agreement"] = {"n": len(both), "agree": po, "kappa": (po - pe) / (1 - pe),
                                     "match_rate_judge1": float(x.mean()), "match_rate_judge2": float(y.mean())}
        print("judge agreement", json.dumps(report["judge_agreement"]), flush=True)
        if a.strict:
            J = {k: v and J2.get(k, False) for k, v in J.items()}
            report["strict"] = True
    cache = {}

    def get(tag):
        if tag not in cache:
            rows = load(Path(a.runs) / tag, J)
            per, sig = {}, {}
            for cid, s in rows.items():
                per[cid], sg = case_row(s)
                if sg:
                    sig[cid] = sg
            m = {k: float(np.mean([r[k] for r in per.values() if r[k] is not None])) for k in METRICS}
            for k in ["single", "vote32"]:
                m[k] = float(np.mean([sig[c][k] for c in per if c in sig] + [0.0] * sum(1 for c in per if c not in sig)))
            sub = [c for c in sig if sig[c]["agree32"] is not None]
            y = np.array([sig[c]["single"] for c in sub])
            for s in ["stated_p", "agree5", "agree32"]:
                m[f"auroc_{s}"] = auroc(np.array([sig[c][s] for c in sub]), y)
            dd = []
            S, A32 = np.array([sig[c]["stated_p"] for c in sub]), np.array([sig[c]["agree32"] for c in sub])
            for _ in range(2000):
                i = rng.integers(0, len(sub), len(sub))
                dd.append(auroc(S[i], y[i]) - auroc(A32[i], y[i]))
            m["d_auroc_stated-agree32"] = [m["auroc_stated_p"] - m["auroc_agree32"], *map(float, np.nanpercentile(dd, [2.5, 97.5]))]
            for lo, hi, nm in [(0, 2022, "le2022"), (2023, 9999, "ge2023")]:
                cc = [c for c in per if cases[c]["year"] and lo <= cases[c]["year"] <= hi]
                m[f"n_{nm}"] = len(cc)
                for k in ["pass1", "pass32", "rec5"]:
                    vals = [per[c][k] for c in cc if per[c][k] is not None]
                    m[f"{k}_{nm}"] = float(np.mean(vals)) if vals else None
            cache[tag] = (per, sig)
            report["models"][tag] = m
            print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}), flush=True)
        return cache[tag]

    tests = []
    for item in a.pairs.split():
        bt, pt = item.split(">")
        (PB, SB), (PP, SP) = get(bt), get(pt)
        out = {}
        for k in METRICS + ["vote32-single"]:
            if k == "vote32-single":  # within the post-trained model, as on DDXPlus
                cids = [c for c in SP]
                v = np.array([SP[c]["vote32"] - SP[c]["single"] for c in cids])
            else:
                cids = [c for c in cases if PB.get(c, {}).get(k) is not None and PP.get(c, {}).get(k) is not None]
                v = np.array([PP[c][k] - PB[c][k] for c in cids])
            b = v[rng.integers(0, len(v), size=(a.n_boot, len(v)))].mean(1)
            out[k] = {"diff": float(v.mean()), "ci": [float(x) for x in np.percentile(b, [2.5, 97.5])], "p": boot_p(b), "n": len(v)}
            tests.append((item, k))
        report["pairs"][item] = out
    ph = holm([report["pairs"][i][k]["p"] for i, k in tests])
    for (i, k), p in zip(tests, ph):
        report["pairs"][i][k]["p_holm"] = p
    for item, out in report["pairs"].items():
        print(item, json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["ci"]], round(v["p_holm"], 4)] for k, v in out.items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
