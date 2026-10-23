"""Stated vs. sampled differentials on DDXPlus: does a model's single listed differential match the
reference better than the distribution of its sampled top-1 answers?

    python stated_vs_sampled.py --runs-dir RUNS/pair_q3 --tags qwen3-8b-base,qwen3-8b \
        --cases cases_test.jsonl --conditions-json release_conditions.json [--out out.json]

Per case and model (valid samples only: at least one listed name maps to one of the 49 conditions):
    tv_sampled      TV(empirical top-1 distribution over the samples, reference)
    tv_list         mean over samples of TV(one sample's stated list, reference); the list's stated
                    probabilities over matched names are renormalized to sum to 1 (uniform if absent)
    tv_list_mean    TV(mean of the stated lists over samples, reference)
    rec_list        reference mass covered by the names in one list (mean over samples)
    cov5_sampled    reference mass covered by the distinct top-1 answers of 5 random samples
                    (same 5-name budget as one list)
    sev_list        severe (1-2) conditions with reference p >= tau: share of single lists naming them
    sev_top1_5      same, named as top-1 in at least one of 5 random samples
    h_list, h_sampled   entropy of one stated list (mean) and of the sampled top-1 distribution
Oracles: 'oracle-sampler' draws as many top-1 answers from the reference as the model has valid
samples; 'oracle-top5' states the reference's top 5, renormalized.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx

METRICS = ["tv_sampled", "tv_list", "tv_list_mean", "rec_list", "cov5_sampled", "sev_list", "sev_top1_5",
           "h_list", "h_sampled", "valid"]


def tv(p, q):
    return 0.5 * sum(abs(p.get(c, 0) - q.get(c, 0)) for c in set(p) | set(q))


def entropy(p):
    return -sum(v * math.log(v) for v in p.values() if v > 0)


def stated_dist(dx, match):
    names = [match(d["diagnosis"]) for d in dx]
    pairs = [(n, d["probability"]) for n, d in zip(names, dx) if n]
    if not pairs:
        return None, None
    probs = [p for _, p in pairs]
    if any(p is None or not (0 <= p <= 1) for p in probs) or sum(probs) <= 0:
        probs = [1.0] * len(pairs)
    dist = Counter()
    for (n, _), p in zip(pairs, probs):
        dist[n] += p
    z = sum(dist.values())
    return {c: v / z for c, v in dist.items()}, names[0]


def case_metrics(tops, lists, ref, severe_refs, rng, n_perm=50):
    """tops: top-1 names of valid samples; lists: stated dists of valid samples."""
    n = len(tops)
    if n == 0:
        return None
    emp = Counter(tops)
    emp = {c: v / n for c, v in emp.items()}
    mean_list = Counter()
    for d in lists:
        for c, v in d.items():
            mean_list[c] += v / n
    k = min(5, n)
    cov5, sev5 = [], []
    for _ in range(n_perm):
        seen = {tops[i] for i in rng.choice(n, size=k, replace=False)}
        cov5.append(sum(p for c, p in ref.items() if c in seen))
        if severe_refs:
            sev5.append(np.mean([c in seen for c in severe_refs]))
    return {
        "tv_sampled": tv(emp, ref),
        "tv_list": float(np.mean([tv(d, ref) for d in lists])),
        "tv_list_mean": tv(dict(mean_list), ref),
        "rec_list": float(np.mean([sum(p for c, p in ref.items() if c in d) for d in lists])),
        "cov5_sampled": float(np.mean(cov5)),
        "sev_list": float(np.mean([np.mean([c in d for c in severe_refs]) for d in lists])) if severe_refs else None,
        "sev_top1_5": float(np.mean(sev5)) if severe_refs else None,
        "h_list": float(np.mean([entropy(d) for d in lists])),
        "h_sampled": entropy(emp),
        "valid": n,
    }


def summarize(rows, rng, n_boot=2000):
    out = {}
    for m in METRICS:
        v = np.array([r[m] for r in rows if r[m] is not None], float)
        out[m] = float(v.mean()) if len(v) else None
    return out


def paired_ci(a_rows, b_rows, m, rng, n_boot=2000):
    """mean(a - b) over cases where both are defined, with a case bootstrap."""
    d = np.array([a[m] - b[m] for a, b in zip(a_rows, b_rows) if a and b and a[m] is not None and b[m] is not None])
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--tags", required=True, help="comma-separated; the first is the reference model for CIs")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--out")
    a = ap.parse_args()

    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    match = Matcher(sorted(severity))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    tags = a.tags.split(",")
    rng = np.random.default_rng(0)

    per_model = {}
    oracle_s, oracle_t5 = {}, {}
    for t in tags:
        rows = {}
        for rec in map(json.loads, open(Path(a.runs_dir) / t / "samples.jsonl")):
            case = cases[rec["case_id"]]
            z = sum(p for _, p in case["gt_differential"])
            ref = {c: p / z for c, p in case["gt_differential"]}
            severe = [c for c, p in ref.items() if p >= a.tau and severity.get(c, 9) <= 2]
            tops, lists = [], []
            for s in rec["samples"]:
                d, top = stated_dist(parse_dx(s["text"]), match)
                if d is None or top is None:
                    continue
                tops.append(top)
                lists.append(d)
            rows[rec["case_id"]] = case_metrics(tops, lists, ref, severe, rng)
            if t == tags[0]:
                # oracles use the reference model's valid-sample count, so finite-sample noise matches
                n = max(len(tops), 1)
                cs, ps = zip(*ref.items())
                draw = list(rng.choice(len(cs), size=n, p=np.array(ps) / sum(ps)))
                top5 = dict(sorted(ref.items(), key=lambda x: -x[1])[:5])
                zz = sum(top5.values())
                top5 = {c: v / zz for c, v in top5.items()}
                oracle_s[rec["case_id"]] = case_metrics([cs[i] for i in draw], [top5] * n, ref, severe, rng)
        per_model[t] = rows

    common = sorted(set.intersection(*(set(r) for r in per_model.values())))
    report = {"runs_dir": a.runs_dir, "n_cases": len(common), "models": {}, "diffs": {}}
    for t in tags + ["oracle"]:
        rows = [(per_model[t] if t != "oracle" else oracle_s)[c] for c in common]
        rows = [r for r in rows if r]
        report["models"][t] = summarize(rows, rng) | {"n_cases_valid": len(rows)}
    ref_t = tags[0]
    for t in tags[1:]:
        A = [per_model[t][c] for c in common]
        B = [per_model[ref_t][c] for c in common]
        report["diffs"][f"{t} - {ref_t}"] = {m: paired_ci(A, B, m, rng) for m in
                                              ["tv_sampled", "tv_list", "rec_list", "cov5_sampled", "sev_list", "sev_top1_5"]}
    # within-model gap: sampled TV minus single-list TV (positive = list is closer to the reference)
    for t in tags:
        rows = [per_model[t][c] for c in common]
        d = np.array([r["tv_sampled"] - r["tv_list"] for r in rows if r])
        boots = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(1)
        report["models"][t]["gap_tv_sampled_minus_list"] = [float(d.mean()), *map(float, np.percentile(boots, [2.5, 97.5]))]

    print(f"{a.runs_dir}  n_cases={len(common)}  (TV lower is better; rec/cov/sev higher is better)")
    hdr = ["tv_sampled", "tv_list", "tv_list_mean", "rec_list", "cov5_sampled", "sev_list", "sev_top1_5", "h_list", "h_sampled", "valid"]
    print(f"  {'model':22s}" + "".join(f"{h:>13s}" for h in hdr) + "   gap(TVs-TVl)")
    for t, v in report["models"].items():
        g = v.get("gap_tv_sampled_minus_list")
        gs = f"   {g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]" if g else ""
        print(f"  {t:22s}" + "".join(f"{v[h]:13.3f}" if v[h] is not None else f"{'-':>13s}" for h in hdr) + gs)
    for k, v in report["diffs"].items():
        print(f"  {k}: " + "  ".join(f"{m}={x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for m, x in v.items()))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
