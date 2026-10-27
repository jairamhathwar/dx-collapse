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

from consequences import auroc, case_row, rank, spearman
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


def fit_logistic(X, y, l2=1e-2, iters=50):
    """Newton's method for L2-regularized logistic regression with an intercept."""
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w


def model_summary(rows, rng, sizes=(50, 100, 200), n_draw=500, agree_key="consist@5"):
    sub = usable(rows)
    y = np.array([r["A_correct"] for r in sub])
    sp = np.array([r["stated_p"] for r in sub])
    a5 = np.array([r[agree_key] for r in sub])  # the agreement signal used by the audit and the combinations
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
        agree, loss, agree_ho, loss_ho = [], [], [], []
        ho = {"audit": [], "always_agree": [], "always_stated": [], "rank_avg": [], "logistic": []}
        for _ in range(n_draw):
            i = rng.choice(len(sub), size=min(n, len(sub)), replace=False)
            s_, a_ = auroc(sp[i], y[i]), auroc(a5[i], y[i])
            if np.isnan(s_) or np.isnan(a_):
                continue
            pick_stated = s_ > a_
            agree.append(pick_stated == full_pick_stated)
            loss.append(0.0 if pick_stated == full_pick_stated else best - worst)
            # held out: judge the pick on the cases not used for the audit
            rest = np.setdiff1d(np.arange(len(sub)), i)
            sr, ar = auroc(sp[rest], y[rest]), auroc(a5[rest], y[rest])
            agree_ho.append(pick_stated == (sr > ar))
            loss_ho.append(max(sr, ar) - (sr if pick_stated else ar))
            # combined signals, fitted or ranked on the audit cases and scored on the rest
            ho["audit"].append(sr if pick_stated else ar)
            ho["always_agree"].append(ar)
            ho["always_stated"].append(sr)
            ho["rank_avg"].append(auroc(rank(sp[rest]) + rank(a5[rest]), y[rest]))
            mu, sd = np.array([sp[i].mean(), a5[i].mean()]), np.array([sp[i].std() + 1e-6, a5[i].std() + 1e-6])
            Xa = (np.column_stack([sp[i], a5[i]]) - mu) / sd
            Xr = (np.column_stack([sp[rest], a5[rest]]) - mu) / sd
            w = fit_logistic(Xa, y[i])
            ho["logistic"].append(auroc(Xr @ w[1:], y[rest]))
        out["audit"][n] = {"match_rate": float(np.mean(agree)), "mean_auroc_loss": float(np.mean(loss)),
                           "heldout_match_rate": float(np.mean(agree_ho)), "heldout_mean_auroc_loss": float(np.mean(loss_ho)),
                           "heldout_auroc": {k: float(np.nanmean(v)) for k, v in ho.items()}}
    out["gap5"] = out["auroc_stated"] - out["auroc_agree5"]
    out["always_agree_loss"] = max(0.0, out["gap5"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--spec", required=True, help='space-separated "dir:tag1,tag2" items; dir may end with @prob for --top-by-prob')
    ap.add_argument("--out", required=True)
    ap.add_argument("--agree", choices=["5", "32"], default="5", help="agreement over the next 4 or all other samples")
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
            res[f"{d}/{t}"] = model_summary(load(p, cases, match, by_prob), rng, agree_key=f"consist@{a.agree}")
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
    post = [k for k in keys if "base" not in k.split("/")[-1] and "1.7b" not in k]
    summary["always_agree_correct_share"] = float(np.mean([res[k]["full_pick"] == "agree5" for k in keys]))
    summary["always_agree_mean_auroc_loss"] = float(np.mean([res[k]["always_agree_loss"] for k in keys]))
    summary["post_n"] = len(post)
    summary["post_always_agree_correct_share"] = float(np.mean([res[k]["full_pick"] == "agree5" for k in post]))
    summary["post_always_agree_mean_auroc_loss"] = float(np.mean([res[k]["always_agree_loss"] for k in post]))
    for n in (50, 100, 200):
        summary[f"audit{n}_heldout_match_mean"] = float(np.mean([res[k]["audit"][n]["heldout_match_rate"] for k in keys]))
        summary[f"audit{n}_heldout_loss_mean"] = float(np.mean([res[k]["audit"][n]["heldout_mean_auroc_loss"] for k in keys]))
        summary[f"post_audit{n}_heldout_match_mean"] = float(np.mean([res[k]["audit"][n]["heldout_match_rate"] for k in post]))
        summary[f"post_audit{n}_heldout_loss_mean"] = float(np.mean([res[k]["audit"][n]["heldout_mean_auroc_loss"] for k in post]))
    base = [k for k in keys if k not in post]
    summary["base_n"] = len(base)
    for strat in ["audit", "always_agree", "always_stated", "rank_avg", "logistic"]:
        summary[f"base_heldout100_auroc_{strat}"] = float(np.mean([res[k]["audit"][100]["heldout_auroc"][strat] for k in base]))
        summary[f"heldout100_auroc_{strat}"] = float(np.mean([res[k]["audit"][100]["heldout_auroc"][strat] for k in keys]))
        summary[f"post_heldout100_auroc_{strat}"] = float(np.mean([res[k]["audit"][100]["heldout_auroc"][strat] for k in post]))
        summary[f"post_heldout100_wins_vs_audit_{strat}"] = int(sum(res[k]["audit"][100]["heldout_auroc"][strat] > res[k]["audit"][100]["heldout_auroc"]["audit"] for k in post))
    # match rate by size of the true gap
    for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 1)]:
        sel = [k for k in keys if lo <= abs(res[k]["gap5"]) < hi]
        summary[f"audit100_match_gap_{lo}-{hi}"] = [len(sel), float(np.mean([res[k]["audit"][100]["match_rate"] for k in sel])) if sel else None]
    for n in (50, 100, 200):
        summary[f"audit{n}_match_mean"] = float(np.mean([res[k]["audit"][n]["match_rate"] for k in keys]))
        summary[f"audit{n}_match_min"] = float(np.min([res[k]["audit"][n]["match_rate"] for k in keys]))
        summary[f"audit{n}_loss_mean"] = float(np.mean([res[k]["audit"][n]["mean_auroc_loss"] for k in keys]))
    print(json.dumps(summary, indent=1))
    json.dump({"models": res, "summary": summary}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
