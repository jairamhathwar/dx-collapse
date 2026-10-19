"""Per-condition analysis of DDXPlus differentials: which diagnoses drop out after post-training,
and whether the losses concentrate in severe ("can't-miss") conditions.

    python per_condition.py --cases cases_test.jsonl --conditions-json release_conditions.json \
        --model base=PATH/samples.jsonl --model instruct=PATH/samples.jsonl --ref base \
        --tau 0.1 --out-dir OUT

For each model and condition c:
    gold_pass1 / gold_pass32   mean pass@1 / pass@32 over cases whose true pathology is c
    considered_top1            over cases where the reference differential gives c >= tau,
                               fraction where c is the top-1 answer of at least one of the n samples
    considered_listed          same, but c appears anywhere in any sample's listed differential
    pred_rate vs ref_rate      share of all top-1 answers equal to c vs. mean reference probability of c
Severity (DDXPlus: 1 = most severe) aggregates use a case-level bootstrap.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_model(path, match):
    """case_id -> (list of top-1 names, set of all listed names)"""
    out = {}
    for rec in map(json.loads, open(path)):
        tops, listed = [], set()
        for s in rec["samples"]:
            dx = parse_dx(s["text"])
            names = [match(d["diagnosis"]) for d in dx]
            tops.append(names[0] if names else None)
            listed.update(n for n in names if n)
        out[rec["case_id"]] = (tops, listed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True, help="DDXPlus release_conditions.json (for severity)")
    ap.add_argument("--model", action="append", required=True, help="tag=samples.jsonl")
    ap.add_argument("--ref", required=True, help="tag to compare against (e.g. the base model)")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    conds = sorted(severity)
    match = Matcher(conds)
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    models = dict(m.split("=", 1) for m in a.model)
    data = {t: load_model(p, match) for t, p in models.items()}
    common = sorted(set.intersection(*(set(d) for d in data.values())))

    # per-case indicator arrays for bootstrap: considered[t][case][cond] for conds with ref p >= tau
    per_case = {t: [] for t in data}   # list of (case_id, {cond: (top1_hit, listed_hit)})
    for t, d in data.items():
        for cid in common:
            tops, listed = d[cid]
            case = cases[cid]
            ref = {c: p for c, p in case["gt_differential"]}
            z = sum(ref.values())
            hits = {c: (c in tops, c in listed) for c, p in ref.items() if p / z >= a.tau and c in severity}
            per_case[t].append((cid, hits))

    out = {"n_cases": len(common), "tau": a.tau, "models": list(data), "conditions": {}}
    for c in conds:
        entry = {"severity": severity[c]}
        gold_cases = [cid for cid in common if cases[cid]["gold"] == c]
        for t, d in data.items():
            n_tot = sum(len(d[cid][0]) for cid in common)
            pred = sum(tp == c for cid in common for tp in d[cid][0]) / n_tot
            cons = [hits[c] for _, hits in per_case[t] if c in hits]
            g1 = [pass_at_k(len(d[cid][0]), sum(x == c for x in d[cid][0]), 1) for cid in gold_cases]
            g32 = [pass_at_k(len(d[cid][0]), sum(x == c for x in d[cid][0]), 32) for cid in gold_cases]
            entry[t] = {
                "gold_pass1": float(np.mean(g1)) if g1 else None,
                "gold_pass32": float(np.mean(g32)) if g32 else None,
                "considered_top1": float(np.mean([x[0] for x in cons])) if cons else None,
                "considered_listed": float(np.mean([x[1] for x in cons])) if cons else None,
                "n_considered_cases": len(cons),
                "pred_rate": pred,
            }
        entry["ref_rate"] = float(np.mean([dict(cases[cid]["gt_differential"]).get(c, 0) /
                                           sum(p for _, p in cases[cid]["gt_differential"]) for cid in common]))
        out["conditions"][c] = entry

    # severity aggregates with case bootstrap: mean of considered_top1 over (case, cond) pairs in the group
    rng = np.random.default_rng(0)
    groups = {"severe (1-2)": {1, 2}, "moderate (3)": {3}, "mild (4-5)": {4, 5}}
    agg = {}
    for g, levels in groups.items():
        agg[g] = {}
        stats = {}
        for t in data:
            arr = [[float(h[0]) for cc, h in hits.items() if severity[cc] in levels] for _, hits in per_case[t]]
            stats[t] = arr
            flat = [x for r in arr for x in r]
            agg[g][t] = {"considered_top1": float(np.mean(flat)) if flat else None, "pairs": len(flat)}
        for t in data:
            if t == a.ref:
                continue
            diffs = []
            idx_all = np.arange(len(common))
            for _ in range(a.n_boot):
                idx = rng.choice(idx_all, len(idx_all))
                ft = [x for i in idx for x in stats[t][i]]
                fr = [x for i in idx for x in stats[a.ref][i]]
                if ft and fr:
                    diffs.append(np.mean(ft) - np.mean(fr))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            agg[g][t]["delta_vs_ref"] = [float(agg[g][t]["considered_top1"] - agg[g][a.ref]["considered_top1"]),
                                         float(lo), float(hi)]
    out["severity"] = agg

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "per_condition.json", "w"), indent=1)

    # text report
    others = [t for t in data if t != a.ref]
    lines = [f"n_cases={len(common)} tau={a.tau}  considered@n = condition appears as a top-1 answer in >=1 of n samples "
             f"when the reference differential gives it >= tau", ""]
    lines.append("By severity (considered@n, top-1):")
    for g, v in agg.items():
        s = f"  {g:14s} " + "  ".join(f"{t}={v[t]['considered_top1']:.3f}" for t in data)
        for t in others:
            d = v[t]["delta_vs_ref"]
            s += f"   Δ{t}={d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]"
        lines.append(s + f"   (pairs={v[a.ref]['pairs']})")
    for t in others:
        lines.append(f"\nLargest drops in considered@n ({t} vs {a.ref}):")
        drops = sorted(((out["conditions"][c][t]["considered_top1"] or 0) - (out["conditions"][c][a.ref]["considered_top1"] or 0), c)
                       for c in conds if out["conditions"][c][a.ref]["considered_top1"] is not None)
        for dv, c in drops[:12]:
            e = out["conditions"][c]
            lines.append(f"  sev{e['severity']} {c:42s} {e[a.ref]['considered_top1']:.2f} -> {e[t]['considered_top1']:.2f} ({dv:+.2f})"
                         f"  gold p@32 {e[a.ref]['gold_pass32']:.2f} -> {e[t]['gold_pass32']:.2f}  n={e[t]['n_considered_cases']}")
        lines.append(f"\nMost over-predicted by {t} (top-1 share vs reference rate):")
        over = sorted(((out["conditions"][c][t]["pred_rate"] - out["conditions"][c]["ref_rate"], c) for c in conds), reverse=True)
        for dv, c in over[:6]:
            e = out["conditions"][c]
            lines.append(f"  {c:42s} {t} {e[t]['pred_rate']:.3f}  {a.ref} {e[a.ref]['pred_rate']:.3f}  reference {e['ref_rate']:.3f}")
    (od / "per_condition.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = others[0]
        items = sorted(((out["conditions"][c][t]["considered_top1"] or 0) - (out["conditions"][c][a.ref]["considered_top1"] or 0),
                        c) for c in conds if out["conditions"][c][a.ref]["considered_top1"] is not None)
        colors = {1: "#b2182b", 2: "#ef8a62", 3: "#bdbdbd", 4: "#67a9cf", 5: "#2166ac"}
        fig, ax = plt.subplots(figsize=(7, 10))
        ax.barh(range(len(items)), [d for d, _ in items], color=[colors[severity[c]] for _, c in items])
        ax.set_yticks(range(len(items)), [f"{c} (sev {severity[c]})" for _, c in items], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Δ considered@n ({t} − {a.ref})")
        ax.set_title("Which diagnoses drop out of the sampled differential")
        fig.tight_layout()
        fig.savefig(od / "per_condition_delta.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
