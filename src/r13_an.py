"""Ninth-round checks (DDXPlus, CPU only), using review2.py's per-case loader so values match the headline tables.

    python r13_an.py --runs RUNS --overflow OVERFLOW --cases cases_test.jsonl --conditions-json release_conditions.json \
        --union-conditions release_conditions_mnm_union.json --out r13.json

Groups: zero-shot T=0.7 (five final pairs), worked examples (six pairs incl. Mistral), another model's examples (six).
1. Stated-list concentration: per case, mean entropy (nats) of one list's renormalized probabilities and mean top-1 stated
   probability; post - base with condition-clustered intervals.
2. Joint cluster bootstrap of pooled effects: one draw of the 49 pathologies is applied to every pair of a group; the
   pooled effect is the mean of the pairs' differences, and the family contrast is the Qwen mean minus the others' mean.
3. Common cases: (a) cases with at least 16 valid samples for both models: pass@16 and distinct answers over the first 16
   valid samples, recall@5, one list's TV and severe named; (b) agreement and stated-probability AUROC on cases whose
   valid samples are not unanimous for either model.
4. Selective prediction: accuracy of the single-call answer at 50% and 80% coverage and the area under the risk-coverage
   curve (AURC, lower is better), ranking cases by the stated probability or by 31-sample agreement.
5. Dangerous misses: on cases whose true pathology is severe (DDXPlus severity 1-2) or in the published must-not-miss
   union, the share of lists that do not contain it among their five items; and, on cases whose reference warrants at
   least one severe condition (weight >= 0.1), the share of lists that name none of them.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from consequences import auroc
from ddx_names import Matcher
from review2 import load_model, pass_at_k

ZS = [("pair_q3", "qwen3-8b-base", "qwen3-8b"), ("pair25_t07", "qwen25-7b-base", "qwen25-7b-it"),
      ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it"), ("ddx_olmo", "olmo3-base", "olmo3-instruct"),
      ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct")]
FAM = [("qwen3-8b-base", "qwen3-8b"), ("qwen25-7b-base", "qwen25-7b-it"), ("qwen25-32b-base", "qwen25-32b-it"),
       ("olmo3-base", "olmo3-instruct"), ("olmo31-32b-base", "olmo31-32b-instruct"), ("mistral7b-base", "mistral7b-it")]
POOL = ["pass32", "distinct", "rec5", "tv_list", "cond_tv", "sev_named", "auroc_agree32"]


def entropy(d):
    return -sum(p * math.log(p) for p in d.values() if p > 0)


def extra(r, gold, ref, severity, union, tau):
    lists = r.get("lists") or []
    if not lists:
        return
    r["ent"] = float(np.mean([entropy(d) for d in lists]))
    r["top1p"] = float(np.mean([max(d.values()) for d in lists]))
    top5 = [set(sorted(d, key=lambda k: -d[k])[:5]) for d in lists]
    if severity.get(gold, 9) <= 2:
        r["miss_severe_true"] = float(np.mean([gold not in s for s in top5]))
    if gold in union:
        r["miss_union_true"] = float(np.mean([gold not in s for s in top5]))
    warr = [k for k, p in ref.items() if p >= tau and severity.get(k, 9) <= 2]
    if warr:
        r["miss_all_warranted"] = float(np.mean([not any(k in d for k in warr) for d in lists]))
    tops = r["_tops_valid"]
    r["n_valid"] = len(tops)
    r["unanimous"] = len(set(tops)) == 1
    if len(tops) >= 16:
        t16 = tops[:16]
        r["pass16_valid"] = pass_at_k(16, sum(t == gold for t in t16), 16)
        r["distinct16_valid"] = float(len(set(t16)))


def load(path, cases, refs, severity, match, union, tau):
    rows = load_model(path, cases, refs, severity, match, tau)
    # valid top answers in sample order (load_model keeps only lists; recover tops from the lists' argmax order)
    from parsing import parse_dx
    from stated_vs_sampled import stated_dist
    for rec in map(json.loads, open(path)):
        cid = rec["case_id"]
        tops = []
        for s in rec["samples"]:
            d, t = stated_dist(parse_dx(s["text"]), match)
            if d is not None:
                tops.append(t)
        rows[cid]["_tops_valid"] = tops
        extra(rows[cid], cases[cid]["gold"], refs[cid], severity, union, tau)
    return rows


def clusters_of(cids, gold):
    g = defaultdict(list)
    for i, c in enumerate(cids):
        g[gold[c]].append(i)
    return g


def cl_diff(vals, cids, gold, rng, n_boot):
    v = np.array([vals[c] for c in cids], float)
    g = clusters_of(cids, gold); ks = list(g); arr = [np.array(g[k]) for k in ks]
    bs = [v[np.concatenate([arr[j] for j in rng.choice(len(arr), len(arr))])].mean() for _ in range(n_boot)]
    return {"diff": float(v.mean()), "ci": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "n": len(cids)}


def selective(rows):
    sub = [r for r in rows.values() if r.get("agree32") is not None]
    y = np.array([r["A_correct"] for r in sub])
    out = {"n": len(sub), "acc@100": float(y.mean())}
    rng = np.random.default_rng(1)
    for s in ["stated_p", "agree32"]:
        sc = np.array([r[s] for r in sub])
        order = np.argsort(-sc + 1e-9 * rng.random(len(sc)))
        acc = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
        out[f"aurc_{s}"] = float(np.mean(1 - acc))
        for cov in (0.5, 0.8):
            out[f"acc@{int(cov * 100)}_{s}"] = float(acc[int(cov * len(acc)) - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "overflow", "cases", "conditions-json", "union-conditions", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--tau", type=float, default=0.1)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    union = {c["cond-name-eng"] for c in json.load(open(a.union_conditions)).values() if int(c["severity"]) <= 2}
    match = Matcher(sorted(severity))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    gold = {c: v["gold"] for c, v in cases.items()}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)
    groups = {"zero-shot": [(f"{a.runs}/{d}/{b}/samples.jsonl", f"{a.runs}/{d}/{p}/samples.jsonl", b) for d, b, p in ZS],
              "worked examples": [(f"{a.overflow}/fewshot_t07/{b}/samples.jsonl", f"{a.overflow}/fewshot_t07/{p}/samples.jsonl", b) for b, p in FAM],
              "examples from another model": [(f"{a.overflow}/fewshot_rf_t07/{b}/samples.jsonl", f"{a.overflow}/fewshot_rf_t07/{p}/samples.jsonl", b) for b, p in FAM]}
    report = {}
    for gname, items in groups.items():
        G = {"pairs": {}, "pooled": {}}
        loaded = []
        for pb, pp, btag in items:
            B = load(pb, cases, refs, severity, match, union, a.tau)
            P = load(pp, cases, refs, severity, match, union, a.tau)
            loaded.append((btag, B, P))
            res = {"models": {"base": {}, "post": {}}, "diff": {}}
            for lab, R in (("base", B), ("post", P)):
                for k in ["ent", "top1p", "miss_severe_true", "miss_union_true", "miss_all_warranted"]:
                    vals = [r[k] for r in R.values() if r.get(k) is not None]
                    res["models"][lab][k] = float(np.mean(vals)) if vals else None
                res["models"][lab]["selective"] = selective(R)
            for k in ["ent", "top1p", "miss_severe_true", "miss_union_true", "miss_all_warranted", "rec5", "tv_list", "sev_named"]:
                cids = [c for c in cases if B[c].get(k) is not None and P[c].get(k) is not None]
                if cids:
                    res["diff"][k] = cl_diff({c: P[c][k] - B[c][k] for c in cids}, cids, gold, rng, a.n_boot)
            # common cases with >= 16 valid samples in both
            c16 = [c for c in cases if B[c].get("pass16_valid") is not None and P[c].get("pass16_valid") is not None]
            res["common16_n"] = len(c16)
            for k in ["pass16_valid", "distinct16_valid", "rec5", "tv_list", "sev_named"]:
                cc = [c for c in c16 if B[c].get(k) is not None and P[c].get(k) is not None]
                if cc:
                    res["diff"][f"common16:{k}"] = cl_diff({c: P[c][k] - B[c][k] for c in cc}, cc, gold, rng, a.n_boot)
            # AUROC on cases non-unanimous for both
            nu = [c for c in cases if B[c].get("agree32") is not None and P[c].get("agree32") is not None
                  and not B[c]["unanimous"] and not P[c]["unanimous"]]
            res["nonunanimous_both_n"] = len(nu)
            for lab, R in (("base", B), ("post", P)):
                y = np.array([R[c]["A_correct"] for c in nu])
                for s in ["stated_p", "agree32"]:
                    res["models"][lab][f"auroc_{s}_nonunanimous_both"] = auroc(np.array([R[c][s] for c in nu]), y)
            G["pairs"][btag] = res
            print(gname, btag, json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["ci"]]] for k, v in res["diff"].items()}),
                  "| ent", {l: round(res["models"][l]["ent"], 3) for l in ("base", "post")},
                  "| sel", {l: {k: round(v, 3) for k, v in res["models"][l]["selective"].items() if k != "n"} for l in ("base", "post")},
                  "| nu_both", res["nonunanimous_both_n"], {l: (round(res["models"][l]["auroc_agree32_nonunanimous_both"], 3), round(res["models"][l]["auroc_stated_p_nonunanimous_both"], 3)) for l in ("base", "post")},
                  flush=True)
        # joint cluster bootstrap of pooled effects
        allc = list(cases)
        g = clusters_of(allc, gold); ks = list(g); arr = [np.array(g[k]) for k in ks]
        nanf = lambda r, k: (np.nan if r.get(k) is None else float(r[k]))
        def vec(B, P, m):
            if m == "auroc_agree32":
                ok = np.array([B[c].get("agree32") is not None and P[c].get("agree32") is not None for c in allc])
                return {"ok": ok, "yb": np.array([B[c].get("A_correct", 0) for c in allc]), "yp": np.array([P[c].get("A_correct", 0) for c in allc]),
                        "sb": np.array([nanf(B[c], "agree32") for c in allc]), "sp": np.array([nanf(P[c], "agree32") for c in allc])}
            return np.array([nanf(P[c], m) - nanf(B[c], m) for c in allc])
        def pair_diff(v, idx, m):
            if m == "auroc_agree32":
                i = idx[v["ok"][idx]]
                return auroc(v["sp"][i], v["yp"][i]) - auroc(v["sb"][i], v["yb"][i])
            return float(np.nanmean(v[idx]))
        isq = np.array(["qwen" in t for t, _, _ in loaded])
        full = np.arange(len(allc))
        for m in POOL:
            V = [vec(B, P, m) for _, B, P in loaded]
            est = np.array([pair_diff(v, full, m) for v in V])
            boots = []
            for _ in range(a.n_boot):
                idx = np.concatenate([arr[j] for j in rng.choice(len(arr), len(arr))])
                boots.append([pair_diff(v, idx, m) for v in V])
            boots = np.array(boots)
            pooled = boots.mean(1)
            contrast = boots[:, isq].mean(1) - boots[:, ~isq].mean(1)
            G["pooled"][m] = {"pooled": float(est.mean()), "pooled_ci": [float(x) for x in np.percentile(pooled, [2.5, 97.5])],
                              "qwen_minus_other": float(est[isq].mean() - est[~isq].mean()),
                              "contrast_ci": [float(x) for x in np.percentile(contrast, [2.5, 97.5])]}
            print(gname, "pooled", m, json.dumps({k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in G["pooled"][m].items()}), flush=True)
        report[gname] = G
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
