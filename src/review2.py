"""Second-round review analyses on DDXPlus (CPU only).

    python review2.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --pairs "pair_q3:qwen3-8b-base>qwen3-8b ..." --out review2.json

For each base > post-trained pair (valid samples = a listed name maps to one of the 49 conditions):
 1. Chance baselines: Cov@32, TV of 32 sampled answers and distinct answers for a uniform guesser and
    for a guesser drawing from the reference averaged over test cases; for the stated readout, a fixed
    "prior list" (top five of the average reference) with its TV, mass and severe naming.
 2. TV without the true pathology: remove it from the list (or the samples) and from the reference,
    renormalize both, and recompute TV (so a model cannot gain by naming the gold diagnosis).
 3. DDXPlus set metrics per list: DDR (share of reference conditions named), DDP (share of named
    conditions in the reference), DDF1.
 4. Over-naming per list: number of named severe conditions with reference probability < 0.1.
 5. Severe inclusion per condition without a threshold: for each severe condition, the share of lists
    naming it among cases where the reference gives it any probability.
 6. Base -> post differences for the headline metrics with case-bootstrap and condition-clustered
    bootstrap intervals (clusters = true pathology, 49 x 20 cases), bootstrap p-values, and Holm
    correction across all pairs and metrics.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from consequences import auroc
from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist, tv

METRICS = ["pass1", "pass32", "distinct", "cov32", "tv_samples", "tv_list", "mass_list", "sev_named",
           "tv_list_nogold", "tv_samples_nogold", "ddf1", "overname_count", "auroc_stated", "auroc_agree32"]


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def renorm_without(d, gold):
    d = {k: v for k, v in d.items() if k != gold}
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else None


def load_model(path, cases, refs, severity, match, tau):
    rows = {}
    for rec in map(json.loads, open(path)):
        cid = rec["case_id"]
        ref, gold = refs[cid], cases[cid]["gold"]
        lists, tops = [], []
        for s in rec["samples"]:
            d, top = stated_dist(parse_dx(s["text"]), match)
            lists.append(d)
            tops.append(top)
        n = len(tops)
        valid = [(d, t) for d, t in zip(lists, tops) if d is not None]
        r = {"gold": gold, "n": n, "c": sum(t == gold for t in tops)}
        r["pass1"] = pass_at_k(n, r["c"], 1)
        r["pass32"] = pass_at_k(n, r["c"], 32)
        seen = {t for t in tops if t}
        r["distinct"] = len(seen)
        r["cov32"] = sum(p for k, p in ref.items() if k in seen)
        if valid:
            emp = Counter(t for _, t in valid)
            emp = {k: v / len(valid) for k, v in emp.items()}
            r["tv_samples"] = tv(emp, ref)
            ref_ng = renorm_without(ref, gold)
            emp_ng = renorm_without(emp, gold)
            r["tv_samples_nogold"] = tv(emp_ng, ref_ng) if (ref_ng and emp_ng) else None
            tl, ms, sv, tng, f1, oc = [], [], [], [], [], []
            sev_elig = [k for k, p in ref.items() if p >= tau and severity.get(k, 9) <= 2]
            for d, _ in valid:
                tl.append(tv(d, ref))
                ms.append(sum(p for k, p in ref.items() if k in d))
                if sev_elig:
                    sv.append(np.mean([k in d for k in sev_elig]))
                dng = renorm_without(d, gold)
                if ref_ng and dng:
                    tng.append(tv(dng, ref_ng))
                support = {k for k, p in ref.items() if p > 0}
                rec_ = len(support & set(d)) / len(support)
                prec = len(support & set(d)) / len(d)
                f1.append(0.0 if rec_ + prec == 0 else 2 * rec_ * prec / (rec_ + prec))
                oc.append(sum(1 for k in d if severity.get(k, 9) <= 2 and ref.get(k, 0) < tau))
            r.update({"tv_list": np.mean(tl), "mass_list": np.mean(ms), "sev_named": np.mean(sv) if sv else None,
                      "tv_list_nogold": np.mean(tng) if tng else None, "ddf1": np.mean(f1), "overname_count": np.mean(oc)})
            d0, A = valid[0]
            others = [t for _, t in valid[1:]]
            r["A_correct"] = float(A == gold)
            r["stated_p"] = d0.get(A, 0.0)
            r["agree32"] = float(np.mean([t == A for t in others])) if others else None
            r["lists"] = [d for d, _ in valid]
        rows[cid] = r
    return rows


def case_metric(r, m):
    return r.get(m)


def pair_diffs(B, P, cids, rng, n_boot):
    """returns {metric: dict(diff, case_ci, cluster_ci, p_cluster)} for post - base"""
    clusters = defaultdict(list)
    for i, c in enumerate(cids):
        clusters[B[c]["gold"]].append(i)
    cl_keys = list(clusters)
    cl_arr = [np.array(clusters[k]) for k in cl_keys]
    draw_clusters = lambda: np.concatenate([cl_arr[k] for k in rng.choice(len(cl_arr), len(cl_arr))])
    out = {}
    for m in METRICS:
        if m.startswith("auroc"):
            sig = "stated_p" if m == "auroc_stated" else "agree32"
            ok = [i for i, c in enumerate(cids) if B[c].get(sig) is not None and P[c].get(sig) is not None]
            yb = np.array([B[cids[i]]["A_correct"] for i in ok]); sb = np.array([B[cids[i]][sig] for i in ok])
            yp = np.array([P[cids[i]]["A_correct"] for i in ok]); spp = np.array([P[cids[i]][sig] for i in ok])
            posarr = np.full(len(cids), -1)
            posarr[np.array(ok)] = np.arange(len(ok))
            f = lambda idx: auroc(spp[idx], yp[idx]) - auroc(sb[idx], yb[idx])
            full = np.arange(len(ok))
            point = f(full)
            boots_case = [f(rng.integers(0, len(ok), len(ok))) for _ in range(n_boot)]
            boots_cl = []
            for _ in range(n_boot):
                idx = posarr[draw_clusters()]
                boots_cl.append(f(idx[idx >= 0]))
        else:
            v = np.array([(P[c][m] - B[c][m]) if (P[c].get(m) is not None and B[c].get(m) is not None) else np.nan for c in cids], float)
            point = float(np.nanmean(v))
            boots_case = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
            boots_cl = []
            for _ in range(n_boot):
                boots_cl.append(np.nanmean(v[draw_clusters()]))
        bc, bl = np.array(boots_case), np.array(boots_cl)
        p_cl = float(min(1.0, 2 * min(np.mean(bl <= 0), np.mean(bl >= 0))))
        out[m] = {"diff": float(point), "case_ci": [float(x) for x in np.nanpercentile(bc, [2.5, 97.5])],
                  "cluster_ci": [float(x) for x in np.nanpercentile(bl, [2.5, 97.5])], "p_cluster": p_cl}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--pairs", required=True, help='space-separated "dir:base>post"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    conds = sorted(severity)
    match = Matcher(conds)
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)

    # chance baselines (model-independent)
    avg = Counter()
    for r in refs.values():
        for k, p in r.items():
            avg[k] += p / len(refs)
    ac, ap_ = zip(*sorted(avg.items())); ap_ = np.array(ap_) / sum(ap_)
    prior_list = dict(sorted(avg.items(), key=lambda x: -x[1])[:5]); z = sum(prior_list.values())
    prior_list = {k: v / z for k, v in prior_list.items()}
    base = {"uniform": defaultdict(list), "prior": defaultdict(list), "prior_list": defaultdict(list)}
    for cid, ref in refs.items():
        for name, draw in [("uniform", [conds[i] for i in rng.integers(0, len(conds), 32)]),
                           ("prior", [ac[i] for i in rng.choice(len(ac), 32, p=ap_)])]:
            seen = set(draw); emp = Counter(draw); emp = {k: v / 32 for k, v in emp.items()}
            base[name]["cov32"].append(sum(p for k, p in ref.items() if k in seen))
            base[name]["tv_samples"].append(tv(emp, ref))
            base[name]["distinct"].append(len(seen))
            base[name]["pass32"].append(float(cases[cid]["gold"] in seen))
        sev_elig = [k for k, p in ref.items() if p >= a.tau and severity.get(k, 9) <= 2]
        base["prior_list"]["tv_list"].append(tv(prior_list, ref))
        base["prior_list"]["mass_list"].append(sum(p for k, p in ref.items() if k in prior_list))
        if sev_elig:
            base["prior_list"]["sev_named"].append(np.mean([k in prior_list for k in sev_elig]))
        support = {k for k, p in ref.items() if p > 0}
        rec_ = len(support & set(prior_list)) / len(support); prec = len(support & set(prior_list)) / 5
        base["prior_list"]["ddf1"].append(0 if rec_ + prec == 0 else 2 * rec_ * prec / (rec_ + prec))
    report = {"chance": {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in base.items()},
              "prior_list": list(prior_list), "pairs": {}, "models": {}, "severe_per_condition": {}}
    print("chance", json.dumps(report["chance"]), flush=True)

    tests = []
    for item in a.pairs.split():
        d, pr = item.split(":")
        bt, pt = pr.split(">")
        B = load_model(Path(a.runs) / d / bt / "samples.jsonl", cases, refs, severity, match, a.tau)
        P = load_model(Path(a.runs) / d / pt / "samples.jsonl", cases, refs, severity, match, a.tau)
        cids = sorted(set(B) & set(P))
        for tag, M in [(bt, B), (pt, P)]:
            report["models"][f"{d}/{tag}"] = {m: float(np.nanmean([M[c][m] for c in cids if M[c].get(m) is not None]))
                                              for m in METRICS if not m.startswith("auroc")}
        # severe inclusion per condition, threshold-free (reference gives it any probability)
        per = {}
        for k in conds:
            if severity[k] > 2:
                continue
            vals = {}
            for tag, M in [("base", B), ("post", P)]:
                inc = [np.mean([k in L for L in M[c]["lists"]]) for c in cids if refs[c].get(k, 0) > 0 and M[c].get("lists")]
                vals[tag] = float(np.mean(inc)) if inc else None
            per[k] = vals
        report["severe_per_condition"][f"{d}:{bt}>{pt}"] = per
        diffs = pair_diffs(B, P, cids, rng, a.n_boot)
        report["pairs"][f"{d}:{bt}>{pt}"] = diffs
        for m, v in diffs.items():
            tests.append((f"{d}:{bt}>{pt}", m, v["p_cluster"]))
        print(f"{d}:{bt}>{pt}", json.dumps({m: [round(v['diff'], 4), [round(x, 4) for x in v['cluster_ci']], round(v['p_cluster'], 4)] for m, v in diffs.items()}), flush=True)
        dec = sum(1 for v in per.values() if v["base"] is not None and v["post"] is not None and v["post"] < v["base"])
        print(f"  severe conditions with lower inclusion after post-training: {dec}/{sum(1 for v in per.values() if v['base'] is not None)}", flush=True)
    # Holm across all tests
    order = sorted(range(len(tests)), key=lambda i: tests[i][2])
    m_ = len(tests); adj = [None] * m_; running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m_ - rank) * tests[i][2]))
        adj[i] = running
    for (pair, m, p), q in zip(tests, adj):
        report["pairs"][pair][m]["p_holm"] = q
    report["n_tests"] = m_
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
