"""When does the stated probability beat sample agreement as an error signal, and can a small audit tell?

    python audit.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --spec "pair_q3:qwen3-8b-base,qwen3-8b ddx_olmo:olmo3-base,..." --out audit.json

Per model (valid samples only; single-call answer A = top-1 of the first valid sample):
  auroc_stated, auroc_agree5, auroc_agree32   as in consequences.py
  scr5, scr32      single-answer rate: share of cases whose first 5 / all valid samples share one top-1
  distinct         mean distinct top-1 answers
Across models: Spearman correlation of the gap (stated - agreement) with scr and with auroc_stated.
Audit simulation: draw n cases (n = 50, 100, 200) and use only each case's first 5 samples; the audit picks
the signal with the higher AUROC (stated vs agreement over the other 4). We report how often that pick
matches the full-data pick (980 cases, same 5-sample budget) and the AUROC lost when it does not.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from consequences import auroc, case_row, spearman
from ddx_names import Matcher
from parsing import parse_dx, sort_by_prob
from stated_vs_sampled import entropy, stated_dist


def load(path, cases, match, top_by_prob):
    rows = []
    for rec in map(json.loads, open(path)):
        case = cases[rec["case_id"]]
        z = sum(p for _, p in case["gt_differential"])
        ref = {c: p / z for c, p in case["gt_differential"]}
        tops, lists = [], []
        for s in rec["samples"]:
            d, top = stated_dist((sort_by_prob if top_by_prob else list)(parse_dx(s["text"])), match)
            if d is not None and top is not None:
                tops.append(top)
                lists.append(d)
        r = case_row(tops, lists, case["gold"], entropy(ref))
        r["scr5"] = float(len(tops[:5]) >= 2 and len(set(tops[:5])) == 1) if tops else None
        r["scr32"] = float(len(tops) >= 2 and len(set(tops)) == 1) if tops else None
        r["n_distinct"] = len(set(tops)) if tops else None
        rows.append(r)
    return rows


def usable(rows):
    return [r for r in rows if r.get("A_correct") is not None and r["consist@5"] is not None and r["consist@32"] is not None]


def model_summary(rows, rng, sizes=(50, 100, 200), n_draw=500):
    sub = usable(rows)
    y = np.array([r["A_correct"] for r in sub])
    sp = np.array([r["stated_p"] for r in sub])
    a5 = np.array([r["consist@5"] for r in sub])
    a32 = np.array([r["consist@32"] for r in sub])
    out = {
        "n": len(sub), "acc": float(y.mean()),
        "auroc_stated": auroc(sp, y), "auroc_agree5": auroc(a5, y), "auroc_agree32": auroc(a32, y),
        "scr5": float(np.mean([r["scr5"] for r in sub])), "scr32": float(np.mean([r["scr32"] for r in sub])),
        "distinct": float(np.mean([r["n_distinct"] for r in sub])),
    }
    full_pick_stated = out["auroc_stated"] > out["auroc_agree5"]
    best, worst = max(out["auroc_stated"], out["auroc_agree5"]), min(out["auroc_stated"], out["auroc_agree5"])
    out["full_pick"] = "stated" if full_pick_stated else "agree5"
    out["audit"] = {}
    for n in sizes:
        agree, loss = [], []
        for _ in range(n_draw):
            i = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
            s_, a_ = auroc(sp[i], y[i]), auroc(a5[i], y[i])
            if np.isnan(s_) or np.isnan(a_):
                continue
            pick_stated = s_ > a_
            agree.append(pick_stated == full_pick_stated)
            loss.append(0.0 if pick_stated == full_pick_stated else best - worst)
        out["audit"][n] = {"match_rate": float(np.mean(agree)), "mean_auroc_loss": float(np.mean(loss))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--spec", required=True, help='space-separated "dir:tag1,tag2" items; dir may end with @prob for --top-by-prob')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    rng = np.random.default_rng(0)
    res = {}
    for item in a.spec.split():
        d, tags = item.split(":")
        by_prob = d.endswith("@prob")
        d = d.removesuffix("@prob")
        for t in tags.split(","):
            p = Path(a.runs) / d / t / "samples.jsonl"
            if not p.exists():
                print("missing", p)
                continue
            res[f"{d}/{t}"] = model_summary(load(p, cases, match, by_prob), rng)
            v = res[f"{d}/{t}"]
            print(f"{d}/{t:24s} acc={v['acc']:.3f} stated={v['auroc_stated']:.3f} agree5={v['auroc_agree5']:.3f} "
                  f"agree32={v['auroc_agree32']:.3f} scr5={v['scr5']:.3f} scr32={v['scr32']:.3f} distinct={v['distinct']:.2f} "
                  f"pick={v['full_pick']} audit100={v['audit'][100]['match_rate']:.2f}", flush=True)
    keys = list(res)
    gap5 = np.array([res[k]["auroc_stated"] - res[k]["auroc_agree5"] for k in keys])
    gap32 = np.array([res[k]["auroc_stated"] - res[k]["auroc_agree32"] for k in keys])
    summary = {"n_models": len(keys)}
    for name, x in [("scr5", "scr5"), ("scr32", "scr32"), ("auroc_stated", "auroc_stated"), ("distinct", "distinct")]:
        xv = np.array([res[k][x] for k in keys])
        summary[f"rho_gap5_{name}"] = spearman(gap5, xv)
        summary[f"rho_gap32_{name}"] = spearman(gap32, xv)
    for n in (50, 100, 200):
        summary[f"audit{n}_match_mean"] = float(np.mean([res[k]["audit"][n]["match_rate"] for k in keys]))
        summary[f"audit{n}_match_min"] = float(np.min([res[k]["audit"][n]["match_rate"] for k in keys]))
        summary[f"audit{n}_loss_mean"] = float(np.mean([res[k]["audit"][n]["mean_auroc_loss"] for k in keys]))
    print(json.dumps(summary, indent=1))
    json.dump({"models": res, "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
