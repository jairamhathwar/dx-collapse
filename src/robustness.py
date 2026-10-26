"""Robustness checks requested in review (DDXPlus, CPU only).

    python robustness.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --spec "pair_q3:qwen3-8b-base,qwen3-8b ..." --out robustness.json

Per model (valid samples = a listed name maps to one of the 49 conditions):
 1. Chance and list-sampling baselines for the sampled readout:
      pass@1, pass@32, distinct, severe Considered@32 for the model's own samples (all n samples,
      invalid = no answer), for a uniform guesser over the 49 conditions, for a guesser drawing from
      the reference differential's average over test cases ("prior"), and for "list sampling":
      each valid sample's top-1 replaced by a draw from its own stated list.
 2. List format: invalid-sample rate, mean number of named conditions, share of lists with exactly 5
    distinct on-list conditions and valid probabilities ("complete"), share with missing/invalid
    probabilities, off-list names per list.
 3. Stated readout on complete lists only (tv_list, mass, severe named) and on all valid lists.
 4. Floor-matched TV: TV minus the expected TV of the matching oracle (the reference's top 5 for one
    list; as many draws from the reference as the model has valid samples for the samples).
 5. Accuracy-stratified list TV: mean tv_list over samples whose top-1 is correct vs incorrect.
 6. Over-naming: share of conditions named in one list whose reference probability is < 0.01, and
    share of severe conditions named whose reference probability is < 0.1.
 7. Single call without skipping invalid samples: accuracy of the first sample (invalid = wrong) and
    the vote@32 gain against it.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import tv


def pass_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def parse_list(text, match):
    dx = parse_dx(text)
    names = [match(d["diagnosis"]) for d in dx]
    pairs = [(n, d["probability"]) for n, d in zip(names, dx) if n]
    offlist = sum(1 for n in names if n is None)
    if not pairs:
        return None
    probs = [p for _, p in pairs]
    prob_ok = all(p is not None and 0 <= p <= 1 for p in probs) and sum(probs) > 0
    if not prob_ok:
        probs = [1.0] * len(pairs)
    dist = Counter()
    for (c, _), p in zip(pairs, probs):
        dist[c] += p
    z = sum(dist.values())
    dist = {c: v / z for c, v in dist.items()}
    return {"top": pairs[0][0], "dist": dist, "n_named": len(dist), "prob_ok": prob_ok, "offlist": offlist,
            "complete": len(dist) == 5 and prob_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
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
    prior = Counter()
    for r in refs.values():
        for k, p in r.items():
            prior[k] += p / len(refs)
    prior_c, prior_p = zip(*sorted(prior.items()))
    prior_p = np.array(prior_p) / sum(prior_p)
    rng = np.random.default_rng(0)

    # oracle TV for n draws from each case's reference, cached by (case, n)
    oracle_cache = {}

    def oracle_tv(cid, n, sims=20):
        if (cid, n) not in oracle_cache:
            ref = refs[cid]
            cs, ps = zip(*ref.items())
            ps = np.array(ps)
            v = []
            for _ in range(sims):
                draw = Counter(np.array(cs)[rng.choice(len(cs), size=n, p=ps)])
                v.append(tv({k: x / n for k, x in draw.items()}, ref))
            oracle_cache[(cid, n)] = float(np.mean(v))
        return oracle_cache[(cid, n)]

    def top5_floor(cid):
        ref = refs[cid]
        t = dict(sorted(ref.items(), key=lambda x: -x[1])[:5])
        z = sum(t.values())
        return tv({k: v / z for k, v in t.items()}, ref)

    out = {}
    for item in a.spec.split():
        d, tags = item.split(":")
        for t in tags.split(","):
            p = Path(a.runs) / d / t / "samples.jsonl"
            if not p.exists():
                print("missing", p)
                continue
            acc = {k: [] for k in ["p1_model", "p32_model", "p1_ls", "p32_ls", "p32_uni", "p32_prior",
                                   "dist_model", "dist_ls", "dist_uni", "dist_prior"]}
            sev = {k: [] for k in ["model", "ls", "uni", "prior"]}
            fmt = Counter()
            tvl_all, tvl_cmp, rec_all, rec_cmp, sevn_all, sevn_cmp = [], [], [], [], [], []
            ex_list, ex_samp = [], []
            tvl_correct, tvl_wrong = [], []
            over_low, over_sev = [], []
            first_any, vote = [], []
            for rec in map(json.loads, open(p)):
                cid = rec["case_id"]
                ref, gold = refs[cid], cases[cid]["gold"]
                lists = [parse_list(s["text"], match) for s in rec["samples"]]
                n = len(lists)
                valid = [L for L in lists if L]
                fmt["samples"] += n
                fmt["invalid"] += n - len(valid)
                tops = [L["top"] if L else None for L in lists]
                ls_tops = [L and str(rng.choice(list(L["dist"]), p=np.array(list(L["dist"].values())))) for L in lists]
                uni = [conds[i] for i in rng.integers(0, len(conds), size=n)]
                pri = [prior_c[i] for i in rng.choice(len(prior_c), size=n, p=prior_p)]
                for name, seq in [("model", tops), ("ls", ls_tops), ("uni", uni), ("prior", pri)]:
                    c = sum(x == gold for x in seq)
                    if name in ("model", "ls"):
                        acc[f"p1_{name}"].append(pass_at_k(n, c, 1))
                    acc[f"p32_{name}"].append(pass_at_k(n, c, 32))
                    acc[f"dist_{name}"].append(len({x for x in seq if x}))
                    seen = set(seq)
                    for k, pr in ref.items():
                        if pr >= a.tau and severity.get(k, 9) <= 2:
                            sev[name].append(k in seen)
                first_any.append(float(tops[0] == gold))
                if valid:
                    cnt = Counter(L["top"] for L in valid)
                    vote.append(float(cnt.most_common(1)[0][0] == gold))
                else:
                    vote.append(0.0)
                if not valid:
                    continue
                fl = top5_floor(cid)
                emp = Counter(L["top"] for L in valid)
                emp = {k: v / len(valid) for k, v in emp.items()}
                ex_samp.append(tv(emp, ref) - oracle_tv(cid, len(valid)))
                for L in valid:
                    fmt["valid"] += 1
                    fmt["named"] += L["n_named"]
                    fmt["complete"] += L["complete"]
                    fmt["prob_bad"] += (not L["prob_ok"])
                    fmt["offlist"] += L["offlist"]
                    tl = tv(L["dist"], ref)
                    rc = sum(pr for k, pr in ref.items() if k in L["dist"])
                    sv = [k in L["dist"] for k, pr in ref.items() if pr >= a.tau and severity.get(k, 9) <= 2]
                    tvl_all.append(tl); rec_all.append(rc); sevn_all += sv
                    ex_list.append(tl - fl)
                    (tvl_correct if L["top"] == gold else tvl_wrong).append(tl)
                    if L["complete"]:
                        tvl_cmp.append(tl); rec_cmp.append(rc); sevn_cmp += sv
                    over_low.append(np.mean([ref.get(k, 0) < 0.01 for k in L["dist"]]))
                    named_sev = [k for k in L["dist"] if severity.get(k, 9) <= 2]
                    if named_sev:
                        over_sev.append(np.mean([ref.get(k, 0) < a.tau for k in named_sev]))
            m = lambda x: float(np.mean([v for v in x if v is not None])) if x else None
            r = {
                "pass1_model": m(acc["p1_model"]), "pass32_model": m(acc["p32_model"]),
                "pass1_listsample": m(acc["p1_ls"]), "pass32_listsample": m(acc["p32_ls"]),
                "pass32_uniform": m(acc["p32_uni"]), "pass32_prior": m(acc["p32_prior"]),
                "distinct_model": m(acc["dist_model"]), "distinct_listsample": m(acc["dist_ls"]),
                "distinct_uniform": m(acc["dist_uni"]), "distinct_prior": m(acc["dist_prior"]),
                "severeC32_model": m(sev["model"]), "severeC32_listsample": m(sev["ls"]),
                "severeC32_uniform": m(sev["uni"]), "severeC32_prior": m(sev["prior"]),
                "invalid_rate": fmt["invalid"] / fmt["samples"], "mean_named": fmt["named"] / max(fmt["valid"], 1),
                "complete_rate": fmt["complete"] / max(fmt["valid"], 1), "prob_missing_rate": fmt["prob_bad"] / max(fmt["valid"], 1),
                "offlist_per_list": fmt["offlist"] / max(fmt["valid"], 1),
                "tv_list_all": m(tvl_all), "tv_list_complete": m(tvl_cmp), "n_complete_lists": len(tvl_cmp),
                "mass_all": m(rec_all), "mass_complete": m(rec_cmp), "severe_named_all": m(sevn_all), "severe_named_complete": m(sevn_cmp),
                "excess_tv_list": m(ex_list), "excess_tv_samples": m(ex_samp),
                "tv_list_top1_correct": m(tvl_correct), "tv_list_top1_wrong": m(tvl_wrong),
                "named_ref_below_0.01": m(over_low), "severe_named_ref_below_tau": m(over_sev),
                "single_first_sample_any": m(first_any), "vote32_minus_first_any": m(vote) - m(first_any),
            }
            out[f"{d}/{t}"] = r
            print(f"{d}/{t}", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
