"""Eighth-round checks (DDXPlus, CPU only), zero-shot T=0.7 final pairs.

    python r12_an.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --test-patients release_test_patients --out r12.json

1. Tie-robust error signal: AUROC of the stated probability and of 31-sample agreement for the single-call answer,
   on all cases and on cases whose valid samples do not all give the same top answer (where agreement is not tied).
2. Size-matched top-k: with m = min(3, distinct sampled top answers), the first m items of the first valid list against
   the m most frequent sampled answers (accuracy = true pathology included).
3. One validity convention: pass@1 and pass@32 counting invalid samples as wrong (as in the paper) and over valid
   samples only (pass@1 = share of valid samples correct; pass@32 = any valid sample correct).
4. Reference ceiling: share of cases whose reference top five contains the true pathology.
5. Natural prevalence: headline metrics reweighted from the class-balanced test sample (20 cases per pathology) to the
   pathology frequencies of the full DDXPlus test split; condition-clustered intervals.
6. What replaces acute pulmonary edema: on cases whose reference gives it at least 0.1, the conditions whose share of
   lists naming them rises most from base to post-trained model.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict

import numpy as np

from consequences import auroc
from ddx_names import Matcher
from r8_controls import load
from stated_vs_sampled import tv

PAIRS = [("pair_q3", "qwen3-8b-base", "qwen3-8b"), ("pair25_t07", "qwen25-7b-base", "qwen25-7b-it"),
         ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it"), ("ddx_olmo", "olmo3-base", "olmo3-instruct"),
         ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct")]
APE = "Acute pulmonary edema"


def pass_at_k(n, c, k):
    if n < k or n == 0:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def case_metrics(rows, gold, ref):
    v = [x for x in rows if x]
    n, c = len(rows), sum(1 for x in v if x["top"] == gold)
    r = {"pass1": pass_at_k(n, c, 1), "pass32": pass_at_k(n, c, min(32, n))}
    if not v:
        return r
    r["pass1_valid"] = c / len(v)
    r["pass32_valid"] = float(c > 0)
    tops = [x["top"] for x in v]
    A = tops[0]
    r["correct"] = float(A == gold)
    r["stated_p"] = v[0]["dist"][A]
    r["agree32"] = float(np.mean([t == A for t in tops[1:]])) if len(tops) > 1 else None
    r["unanimous"] = len(set(tops)) == 1
    freq = [k for k, _ in Counter(tops).most_common()]
    m = min(3, len(freq))
    r["topm_list"] = float(gold in list(v[0]["dist"])[:m])
    r["topm_samples"] = float(gold in freq[:m])
    r["rec5"] = float(np.mean([gold in list(x["dist"])[:5] for x in v]))
    r["tv_list"] = float(np.mean([tv(x["dist"], ref) for x in v]))
    r["named"] = [set(x["dist"]) for x in v]
    return r


def wboot(vals, gold_of, cids, w, rng, n_boot=10000):
    """weighted mean difference with a bootstrap over pathologies"""
    by = defaultdict(list)
    for c in cids:
        by[gold_of[c]].append(c)
    keys = list(by)
    def wm(sel):
        num = sum(w[gold_of[c]] * vals[c] for c in sel)
        den = sum(w[gold_of[c]] for c in sel)
        return num / den
    est = wm(cids)
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys))
        sel = [c for j in pick for c in by[keys[j]]]
        bs.append(wm(sel))
    return {"diff": est, "cluster_ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--test-patients", required=True)
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
    prev = Counter(row["PATHOLOGY"] for row in csv.DictReader(open(a.test_patients)))
    tot = sum(prev.values())
    w = {k: prev[k] / tot for k in prev}
    rng = np.random.default_rng(0)
    report = {"reference_ceiling": {}, "models": {}, "pairs": {}}
    report["reference_ceiling"]["top5_contains_gold"] = float(np.mean([gold[c] in sorted(refs[c], key=lambda k: -refs[c][k])[:5] for c in cases]))
    report["reference_ceiling"]["top5_contains_gold_prevalence_weighted"] = float(
        sum(w[gold[c]] * (gold[c] in sorted(refs[c], key=lambda k: -refs[c][k])[:5]) for c in cases) / sum(w[gold[c]] for c in cases))
    print("reference ceiling", report["reference_ceiling"], flush=True)
    for d, b, p in PAIRS:
        CM = {}
        for t in (b, p):
            rows = load(f"{a.runs}/{d}/{t}/samples.jsonl", match)
            CM[t] = {c: case_metrics(rows.get(c, []), gold[c], refs[c]) for c in cases}
            per = CM[t]
            out = {}
            for subset, sel in [("all", lambda r: True), ("non_unanimous", lambda r: not r["unanimous"])]:
                rr = [r for r in per.values() if "correct" in r and r["agree32"] is not None and sel(r)]
                y = np.array([r["correct"] for r in rr])
                out[f"auroc_stated_{subset}"] = auroc(np.array([r["stated_p"] for r in rr]), y)
                out[f"auroc_agree32_{subset}"] = auroc(np.array([r["agree32"] for r in rr]), y)
                out[f"n_{subset}"] = len(rr)
            for k in ["pass1", "pass32", "pass1_valid", "pass32_valid", "topm_list", "topm_samples", "rec5", "tv_list"]:
                vals = [r[k] for r in per.values() if r.get(k) is not None]
                out[k] = float(np.mean(vals))
                wv = [(w[gold[c]], r[k]) for c, r in per.items() if r.get(k) is not None]
                out[k + "_prev"] = float(sum(x * y for x, y in wv) / sum(x for x, _ in wv))
            report["models"][f"{d}/{t}"] = out
            print(t, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}), flush=True)
        pr = {}
        for k in ["pass1", "pass32", "rec5", "tv_list", "pass1_valid", "pass32_valid"]:
            cids = [c for c in cases if CM[b][c].get(k) is not None and CM[p][c].get(k) is not None]
            vals = {c: CM[p][c][k] - CM[b][c][k] for c in cids}
            pr[k + "_prev"] = wboot(vals, gold, cids, w, rng, a.n_boot)
        cids = [c for c in cases if CM[p][c].get("topm_list") is not None]
        pr["topm_list-samples_post"] = wboot({c: CM[p][c]["topm_list"] - CM[p][c]["topm_samples"] for c in cids}, gold, cids, {k: 1.0 for k in w}, rng, a.n_boot)
        cids = [c for c in cases if CM[b][c].get("topm_list") is not None]
        pr["topm_list-samples_base"] = wboot({c: CM[b][c]["topm_list"] - CM[b][c]["topm_samples"] for c in cids}, gold, cids, {k: 1.0 for k in w}, rng, a.n_boot)
        # what replaces acute pulmonary edema
        ape = [c for c in cases if refs[c].get(APE, 0) >= 0.1 and CM[b][c].get("named") and CM[p][c].get("named")]
        rise = Counter()
        for c in ape:
            for k in set().union(*CM[b][c]["named"], *CM[p][c]["named"]):
                rise[k] += (np.mean([k in s for s in CM[p][c]["named"]]) - np.mean([k in s for s in CM[b][c]["named"]])) / len(ape)
        pr["ape_cases"] = len(ape)
        pr["ape_change"] = float(rise[APE])
        pr["ape_replacements"] = [(k, round(float(v), 3)) for k, v in sorted(rise.items(), key=lambda x: -x[1])[:5]]
        report["pairs"][f"{d}:{b}>{p}"] = pr
        print(p, json.dumps({k: (v if not isinstance(v, dict) else [round(v["diff"], 4), [round(x, 4) for x in v["cluster_ci"]]]) for k, v in pr.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
