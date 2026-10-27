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
EXTRA = ["within_list", "cond_tv", "rec1", "rec3", "rec5"]  # reported with intervals, Holm-corrected as a separate family:
# within_list = TV - (1 - mass); cond_tv = TV between the list and the reference restricted to the named conditions and
# renormalized (scale-free allocation error); rec_k = the true pathology is among the list's k most probable conditions


def boot_p(b):
    """two-sided bootstrap p-value with the add-one correction, so it never reaches 0"""
    b = np.asarray(b)
    return float(min(1.0, 2 * (1 + min(np.sum(b <= 0), np.sum(b >= 0))) / (len(b) + 1)))


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
            tl, ms, sv, tng, f1, oc, ctv, rk = [], [], [], [], [], [], [], {1: [], 3: [], 5: []}
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
                mL = sum(ref.get(k, 0) for k in d)
                if mL > 0:
                    ctv.append(tv(d, {k: ref[k] / mL for k in d if k in ref}))
                order = sorted(d, key=lambda k: -d[k])
                for kk in rk:
                    rk[kk].append(float(gold in order[:kk]))
            r.update({"tv_list": np.mean(tl), "mass_list": np.mean(ms), "sev_named": np.mean(sv) if sv else None,
                      "tv_list_nogold": np.mean(tng) if tng else None, "ddf1": np.mean(f1), "overname_count": np.mean(oc)})
            r["within_list"] = r["tv_list"] - (1 - r["mass_list"])
            r["cond_tv"] = float(np.mean(ctv)) if ctv else None
            for kk in rk:
                r[f"rec{kk}"] = float(np.mean(rk[kk]))
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


def pair_diffs(B, P, cids, rng, n_boot, n_boot_case=2000):
    """returns {metric: dict(diff, case_ci, cluster_ci, p_cluster)} for post - base"""
    clusters = defaultdict(list)
    for i, c in enumerate(cids):
        clusters[B[c]["gold"]].append(i)
    cl_keys = list(clusters)
    cl_arr = [np.array(clusters[k]) for k in cl_keys]
    draw_clusters = lambda: np.concatenate([cl_arr[k] for k in rng.choice(len(cl_arr), len(cl_arr))])
    out = {}
    for m in METRICS + EXTRA:
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
            boots_case = [f(rng.integers(0, len(ok), len(ok))) for _ in range(n_boot_case)]
            boots_cl = []
            for _ in range(n_boot):
                idx = posarr[draw_clusters()]
                boots_cl.append(f(idx[idx >= 0]))
        else:
            v = np.array([(P[c][m] - B[c][m]) if (P[c].get(m) is not None and B[c].get(m) is not None) else np.nan for c in cids], float)
            point = float(np.nanmean(v))
            boots_case = [np.nanmean(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot_case)]
            boots_cl = []
            for _ in range(n_boot):
                boots_cl.append(np.nanmean(v[draw_clusters()]))
        bc, bl = np.array(boots_case), np.array(boots_cl)
        p_cl = boot_p(bl)
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
    ap.add_argument("--n-boot", type=int, default=10000, help="cluster-bootstrap resamples (p-values resolve to 2/(n+1))")
    ap.add_argument("--n-boot-case", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
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
                                              for m in METRICS + EXTRA if not m.startswith("auroc")}
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
        # per-condition test: clustered bootstrap over true pathologies, BH-FDR across conditions, and a
        # case-level swap null for the number of conditions named less often
        sev_keys = [k for k in conds if severity[k] <= 2]
        inc = {}
        for k in sev_keys:
            el = [c for c in cids if refs[c].get(k, 0) > 0 and B[c].get("lists") and P[c].get("lists")]
            if not el:
                continue
            inc[k] = (el, np.array([np.mean([k in L for L in B[c]["lists"]]) for c in el]),
                      np.array([np.mean([k in L for L in P[c]["lists"]]) for c in el]))
        for k, (el, bv, pv) in inc.items():
            groups = defaultdict(list)
            for i, c in enumerate(el):
                groups[B[c]["gold"]].append(i)
            garr = [np.array(g) for g in groups.values()]
            dv = pv - bv
            boots = [dv[np.concatenate([garr[j] for j in rng.choice(len(garr), len(garr))])].mean() for _ in range(a.n_boot)]
            per[k].update({"n": len(el), "diff": float(dv.mean()), "cluster_ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                           "p": boot_p(boots)})
        ks = list(inc)
        ps = np.array([per[k]["p"] for k in ks]); order = np.argsort(ps); q = np.empty(len(ps)); run = 1.0
        for r_, i in reversed(list(enumerate(order))):
            run = min(run, ps[i] * len(ps) / (r_ + 1)); q[i] = run
        for k, qq in zip(ks, q):
            per[k]["q_bh"] = float(qq)
        obs_lower = sum(per[k]["diff"] < 0 for k in ks)
        cid_index = {c: i for i, c in enumerate(cids)}
        idx = {k: np.array([cid_index[c] for c in inc[k][0]]) for k in ks}
        null, null_mean = [], []
        obs_mean = float(np.mean([per[k]["diff"] for k in ks]))
        for _ in range(a.n_perm):
            flip = rng.random(len(cids)) < 0.5
            cnt, tot = 0, 0.0
            for k in ks:
                _, bv, pv = inc[k]
                f = flip[idx[k]]
                dm = np.mean(np.where(f, bv - pv, pv - bv))
                cnt += dm < 0
                tot += dm
            null.append(cnt)
            null_mean.append(tot / len(ks))
        null, null_mean = np.array(null), np.array(null_mean)
        report.setdefault("severe_test", {})[f"{d}:{bt}>{pt}"] = {
            "n_conditions": len(ks), "lower": int(obs_lower), "higher": int(len(ks) - obs_lower),
            "sig_lower_q05": int(sum(per[k]["diff"] < 0 and per[k]["q_bh"] < 0.05 for k in ks)),
            "sig_higher_q05": int(sum(per[k]["diff"] > 0 and per[k]["q_bh"] < 0.05 for k in ks)),
            "null_lower_mean": float(null.mean()), "null_lower_p_ge_obs": float((1 + np.sum(null >= obs_lower)) / (len(null) + 1)),
            "mean_shift": obs_mean, "mean_shift_p": float(min(1.0, 2 * (1 + min(np.sum(null_mean <= obs_mean), np.sum(null_mean >= obs_mean))) / (len(null_mean) + 1))),
            "worst": sorted(((per[k]["diff"], k) for k in ks))[:3]}
        print("  severe test:", json.dumps(report["severe_test"][f"{d}:{bt}>{pt}"]), flush=True)
        report["severe_per_condition"][f"{d}:{bt}>{pt}"] = per
        diffs = pair_diffs(B, P, cids, rng, a.n_boot, a.n_boot_case)
        report["pairs"][f"{d}:{bt}>{pt}"] = diffs
        for m, v in diffs.items():
            if m in METRICS:
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
    ext = [(pr, m, v["p_cluster"]) for pr, d_ in report["pairs"].items() for m, v in d_.items() if m in EXTRA]
    order = sorted(range(len(ext)), key=lambda i: ext[i][2]); running = 0.0
    for rank_, i in enumerate(order):
        running = max(running, min(1.0, (len(ext) - rank_) * ext[i][2]))
        report["pairs"][ext[i][0]][ext[i][1]]["p_holm_extra"] = running
    report["n_tests_extra"] = len(ext)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
