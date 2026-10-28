"""One pipeline for the DDXPlus appendix tables (CPU only): every level and difference from review2.py's per-case
loader, every interval from the condition-clustered bootstrap (49 true pathologies x 20 cases).

    python r14_an.py --runs RUNS --overflow OVERFLOW_RUNS --cases cases_test.jsonl \
        --conditions-json release_conditions.json --out r14.json

Per model (levels): pass@1, pass@32 (invalid samples count as wrong), distinct top answers, entropy (nats) of the valid
top answers, TV of the sampled answers and of one list, Cov@32, ECE of the first item's stated probability (10 bins),
invalid shares (top answer not one of the 49 conditions; list naming none of them), Considered@32 by severity tier
(share of case-condition pairs with reference weight >= 0.1 whose condition is a top answer at least once), Vote@32
gain, and the stated-list robustness readouts (complete lists, TV on complete lists, TV when the top answer is right
or wrong, excess TV over the oracle floors, share of named severe conditions the reference gives < 0.1, pass@32 with
each top answer drawn from its own list), and for the full-distribution prompt the AUROCs and top-3 comparison.
Per comparison (later model minus earlier model on the same cases): differences with 95% clustered intervals and
add-one bootstrap p-values (10,000 resamples), incl. pass@1 over valid samples and pass@32 on cases where every sample
of both models is valid.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from consequences import auroc, vote
from ddx_names import Matcher
from parsing import parse_dx, sort_by_prob
from review2 import boot_p, load_model, pass_at_k
from stated_vs_sampled import stated_dist, tv

# (run dir, model tag); "@" marks a run dir under --overflow
Q3T1, Q25T1, Q3, Q25 = "ddx_qwen3_t1", "pair25_t1", "pair_q3", "pair25_t07"
COMPARISONS = {
    "general": [(Q3T1, "qwen3-8b-base", "qwen3-8b"), (Q25T1, "qwen25-7b-base", "qwen25-7b-it"),
                (Q3, "qwen3-8b-base", "qwen3-8b"), (Q25, "qwen25-7b-base", "qwen25-7b-it")],
    "prompts": [(d, "qwen3-8b-base", "qwen3-8b") for d in ["pv_plain_t07", "pv_plain_t1", "pv_cot_t07", "pv_cot_t1"]],
    "stages": [(d, a, b) for d in ["ddx_olmo", "ddx_olmo_t1"] for a, b in
               [("olmo3-base", "olmo3-sft"), ("olmo3-base", "olmo3-dpo"), ("olmo3-base", "olmo3-instruct"),
                ("olmo3-base", "olmo3-rlzero-general"), ("olmo3-sft", "olmo3-dpo"), ("olmo3-dpo", "olmo3-instruct")]],
    "big32": [("big32_ddx", a, b) for a, b in
              [("qwen25-32b-base", "qwen25-32b-it"), ("olmo31-32b-base", "olmo31-32b-sft"), ("olmo31-32b-base", "olmo31-32b-dpo"),
               ("olmo31-32b-base", "olmo31-32b-instruct"), ("olmo31-32b-sft", "olmo31-32b-dpo"), ("olmo31-32b-dpo", "olmo31-32b-instruct")]],
    "medical": [("ddx", "qwen25-7b-it", t) for t in ["alphamed-7b", "huatuo-o1-7b", "m1-7b"]] +
               [("ddx_base_rl", "qwen25-7b-base", "alphamed-7b-base-rl"), ("ddx_base_rl", "alphamed-7b", "alphamed-7b-base-rl"),
                ("ddx_base_rl", "qwen25-7b-base", "qwen25-7b-it"), ("ddx_base_rl", "qwen25-7b-base", "alphamed-7b")],
    "grpo": [("e9_ddx", "qwen3-8b-base", t) for t in ["b8-acc-s0", "b8-gm-s0"]],
    "robust": [(Q3, "qwen3-8b-base", "qwen3-8b"), (Q25, "qwen25-7b-base", "qwen25-7b-it"),
               ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it"), ("ddx_olmo", "olmo3-base", "olmo3-instruct"),
               ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct")],
    "full": [("pv_full_t07", "qwen3-8b-base", "qwen3-8b"), ("pv_full_t07", "qwen25-7b-base", "qwen25-7b-it"),
             ("@pv_full_t07", "olmo3-base", "olmo3-instruct")],
}
TIERS = {"severe": (1, 2), "moderate": (3, 3), "mild": (4, 5)}
DIFF = ["pass1", "pass32", "distinct", "entropy", "cov32", "tv_samples", "tv_list", "rec5", "vote_gain",
        "pass1_valid", "pass32_allvalid", "invalid_top", "pass32_listsampled"] + [f"cons_{t}" for t in TIERS]


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def emp_tv_floor(ref, n, rng, cache, n_draw=200):
    """expected TV between n draws from ref and ref"""
    key = (tuple(sorted(ref.items())), n)
    if key not in cache:
        ks, ps = zip(*ref.items())
        draws = rng.multinomial(n, ps, size=n_draw) / n
        cache[key] = float(np.mean([0.5 * (np.abs(d - np.array(ps)).sum()) for d in draws]))
    return cache[key]


def load(path, cases, refs, severity, match, rng, top_by_prob, floor_cache):
    R = load_model(path, cases, refs, severity, match, 0.1) if not top_by_prob else {}
    order = sort_by_prob if top_by_prob else list
    out = {}
    for rec in map(json.loads, open(path)):
        cid = rec["case_id"]
        gold, ref = cases[cid]["gold"], refs[cid]
        r = R.get(cid, {"gold": gold})
        tops, lists, raw_p, complete = [], [], [], []
        for s in rec["samples"]:
            dx = order(parse_dx(s["text"]))
            d, top = stated_dist(dx, match)
            tops.append(top)
            lists.append(d)
            p = dx[0]["probability"] if dx else None
            raw_p.append(p if (top and p is not None and 0 <= p <= 1) else None)
            names = [match(x["diagnosis"]) for x in dx]
            ps = [x["probability"] for x in dx]
            complete.append(d is not None and len(set(n for n in names if n)) >= 5 and all(n for n in names[:5])
                            and all(q is not None and 0 <= q <= 1 for q in ps[:5]))
        n = len(tops)
        vt = [t for t in tops if t]
        valid = [(d, t) for d, t in zip(lists, tops) if d is not None and t]
        c = sum(t == gold for t in tops)
        r.update({"gold": gold, "n": n, "c": c, "pass1": pass_at_k(n, c, 1), "pass32": pass_at_k(n, c, min(32, n)),
                  "distinct": len(set(vt)), "invalid_top": 1 - len(vt) / n, "invalid_list": sum(d is None for d in lists) / n,
                  "all_valid": len(vt) == n})
        seen = set(vt)
        r["cov32"] = sum(p for k, p in ref.items() if k in seen)
        r["conf"] = [(p, t == gold) for p, t in zip(raw_p, tops) if p is not None]
        for tn, (lo, hi) in TIERS.items():
            el = [k for k, p in ref.items() if p >= 0.1 and lo <= severity.get(k, 9) <= hi]
            r[f"cons_{tn}_hits"], r[f"cons_{tn}_n"] = sum(k in seen for k in el), len(el)
        if vt:
            cnt = Counter(vt)
            r["entropy"] = -sum(v / len(vt) * math.log(v / len(vt)) for v in cnt.values())
            r["pass1_valid"] = c / len(vt)
            emp = {k: v / len(vt) for k, v in cnt.items()}
            r.setdefault("tv_samples", tv(emp, ref))
            r["excess_samples"] = tv(emp, ref) - emp_tv_floor(ref, len(vt), rng, floor_cache)
        if valid:
            ls, ts = [d for d, _ in valid], [t for _, t in valid]
            r.setdefault("tv_list", float(np.mean([tv(d, ref) for d in ls])))
            r["rec5"] = r.get("rec5", float(np.mean([gold in sorted(d, key=lambda k: -d[k])[:5] for d in ls])))
            top5 = dict(sorted(ref.items(), key=lambda x: -x[1])[:5])
            z = sum(top5.values())
            floor_list = tv({k: v / z for k, v in top5.items()}, ref)
            r["excess_list"] = float(np.mean([tv(d, ref) for d in ls])) - floor_list
            cl = [tv(d, ref) for d, ok in zip(lists, complete) if ok]
            r["tv_complete"] = float(np.mean(cl)) if cl else None
            r["complete"] = float(np.mean([ok for ok, d in zip(complete, lists) if d is not None]))
            right = [tv(d, ref) for d, t in valid if t == gold]
            wrong = [tv(d, ref) for d, t in valid if t != gold]
            r["tv_right"] = float(np.mean(right)) if right else None
            r["tv_wrong"] = float(np.mean(wrong)) if wrong else None
            sl = []
            for d in ls:
                sev = [k for k in d if severity.get(k, 9) <= 2]
                if sev:
                    sl.append(np.mean([ref.get(k, 0) < 0.1 for k in sev]))
            r["sev_low_share"] = float(np.mean(sl)) if sl else None
            draws = [rng.choice(list(d), p=np.array(list(d.values())) / sum(d.values())) for d in ls]
            r["pass32_listsampled"] = pass_at_k(n, sum(x == gold for x in draws), min(32, n))
            A = ts[0]
            r["A_correct"] = float(A == gold)
            r["stated_p"] = ls[0].get(A, 0.0)
            r["agree5"] = float(np.mean([t == A for t in ts[1:5]])) if len(ts) > 1 else None
            r["agree32"] = float(np.mean([t == A for t in ts[1:]])) if len(ts) > 1 else None
            r["vote_gain"] = float(vote(ts, ls) == gold) - float(A == gold)
            freq = [k for k, _ in Counter(ts).most_common(3)]
            r["top3_list_minus_samples"] = float(gold in sorted(ls[0], key=lambda k: -ls[0][k])[:3]) - float(gold in freq)
        r.pop("lists", None)
        out[cid] = r
    return out


class Boot:
    """condition-clustered bootstrap via cluster counts: mean = (W @ S) / (W @ N)"""
    def __init__(self, cids, gold, n_boot, rng):
        self.keys = sorted({gold[c] for c in cids})
        self.idx = {k: i for i, k in enumerate(self.keys)}
        self.W = rng.multinomial(len(self.keys), [1 / len(self.keys)] * len(self.keys), size=n_boot).astype(float)
        self.gold = gold

    def ratio(self, num, den):
        """num, den: {case: value}; returns point, ci, p (for differences: test against 0)"""
        S, N = np.zeros(len(self.keys)), np.zeros(len(self.keys))
        for c in num:
            i = self.idx[self.gold[c]]
            S[i] += num[c]
            N[i] += den[c]
        point = S.sum() / N.sum()
        b = (self.W @ S) / np.maximum(self.W @ N, 1e-12)
        b = b[(self.W @ N) > 0]
        return {"diff": float(point), "cluster_ci": [float(x) for x in np.percentile(b, [2.5, 97.5])], "p": boot_p(b)}


def pair_diff(B, P, boot):
    out = {}
    for m in DIFF:
        if m.startswith("cons_"):
            cs = [c for c in B if c in P and B[c][m + "_n"]]
            if cs:
                hb = {c: P[c][m + "_hits"] - B[c][m + "_hits"] for c in cs}
                out[m] = boot.ratio(hb, {c: B[c][m + "_n"] for c in cs})
            continue
        if m == "pass32_allvalid":
            cs = [c for c in B if c in P and B[c]["all_valid"] and P[c]["all_valid"]]
            v = {c: P[c]["pass32"] - B[c]["pass32"] for c in cs}
        else:
            v = {c: P[c][m] - B[c][m] for c in B if c in P and B[c].get(m) is not None and P[c].get(m) is not None}
        if v:
            out[m] = boot.ratio(v, {c: 1.0 for c in v})
            out[m]["n"] = len(v)
    return out


def levels(R):
    m = {}
    for k in ["pass1", "pass32", "distinct", "entropy", "cov32", "tv_samples", "tv_list", "rec5", "invalid_top", "invalid_list",
              "pass1_valid", "vote_gain", "complete", "tv_complete", "tv_right", "tv_wrong", "excess_list", "excess_samples",
              "sev_low_share", "pass32_listsampled", "top3_list_minus_samples", "sev_named", "cond_tv"]:
        v = [r[k] for r in R.values() if r.get(k) is not None]
        if v:
            m[k] = float(np.mean(v))
    for tn in TIERS:
        m[f"cons_{tn}"] = sum(r[f"cons_{tn}_hits"] for r in R.values()) / max(1, sum(r[f"cons_{tn}_n"] for r in R.values()))
    conf = [x for r in R.values() for x in r["conf"]]
    if conf:
        m["ece"] = ece([p for p, _ in conf], [y for _, y in conf])
        m["mean_conf"] = float(np.mean([p for p, _ in conf]))
    m["n_all_valid"] = sum(r["all_valid"] for r in R.values())
    for sig in ["stated_p", "agree5", "agree32"]:
        rr = [r for r in R.values() if r.get(sig) is not None and r.get("A_correct") is not None]
        if rr:
            m[f"auroc_{sig}"] = auroc(np.array([r[sig] for r in rr]), np.array([r["A_correct"] for r in rr]))
    return m


def auroc_gap(R, gold, rng, n_boot, s1="stated_p", s2="agree32"):
    """within-model AUROC(s1) - AUROC(s2), condition-clustered interval"""
    cs = [c for c, r in R.items() if r.get(s1) is not None and r.get(s2) is not None]
    by = defaultdict(list)
    for i, c in enumerate(cs):
        by[gold[c]].append(i)
    groups = [np.array(v) for v in by.values()]
    y = np.array([R[c]["A_correct"] for c in cs])
    a, b = np.array([R[c][s1] for c in cs]), np.array([R[c][s2] for c in cs])
    f = lambda ix: auroc(a[ix], y[ix]) - auroc(b[ix], y[ix])
    bs = []
    for _ in range(n_boot):
        ix = np.concatenate([groups[j] for j in rng.choice(len(groups), len(groups))])
        if 0 < y[ix].sum() < len(ix):
            bs.append(f(ix))
    return {"diff": float(f(np.arange(len(cs)))), "cluster_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "p": boot_p(bs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--overflow", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--groups", default=",".join(COMPARISONS))
    a = ap.parse_args()
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
    boot = Boot(list(cases), gold, a.n_boot, rng)
    cache, floor_cache = {}, {}
    report = {"models": {}, "pairs": {}, "auroc_gap": {}}

    def get(d, t):
        key = f"{d}/{t}"
        if key not in cache:
            root = a.overflow if d.startswith("@") else a.runs
            path = Path(root) / d.lstrip("@") / t / "samples.jsonl"
            cache[key] = load(path, cases, refs, severity, match, rng, d.lstrip("@").startswith("pv_full"), floor_cache)
            report["models"][key] = levels(cache[key])
            print(key, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in report["models"][key].items()}), flush=True)
        return cache[key]

    for g in a.groups.split(","):
        for d, b, p in COMPARISONS[g]:
            B, P = get(d, b), get(d, p)
            key = f"{d}:{b}>{p}"
            if key not in report["pairs"]:
                report["pairs"][key] = pair_diff(B, P, boot)
                print(key, json.dumps({m: [round(v["diff"], 4), [round(x, 4) for x in v["cluster_ci"]]] for m, v in report["pairs"][key].items()}), flush=True)
            if g == "full":
                for t in (b, p):
                    report["auroc_gap"][f"{d}/{t}"] = {"stated-agree32": auroc_gap(cache[f"{d}/{t}"], gold, rng, 2000),
                                                       "stated-agree5": auroc_gap(cache[f"{d}/{t}"], gold, rng, 2000, s2="agree5")}
        json.dump(report, open(a.out, "w"), indent=1)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
