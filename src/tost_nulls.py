"""Equivalence and non-inferiority tests for the paper's 'no difference' claims (pure Python).

    python tost_nulls.py --greedy greedy_matched.json --medcase medcase.json --closed closed.json --out tost_nulls.json

Standard errors come from the 95% intervals already reported (width / 3.92); TOST at the 5% level asks whether the
90% interval lies inside the margin fixed in advance (5 points for accuracy and coverage, 0.03 for AUROC).
Non-inferiority asks only whether the lower 90% bound exceeds -margin.
1. Greedy top-1, post-trained minus base, under both matched pairings: equivalent / lower / higher.
2. Case reports: alternatives covered by one stated list, post-trained minus base (claim: covers them as before).
3. Case reports: stated probability minus agreement AUROC within each post-trained model (claim: at least as good).
"""
import argparse
import json


def tost(diff, lo, hi, margin):
    se = (hi - lo) / 3.92
    a, b = diff - 1.645 * se, diff + 1.645 * se
    return {"diff": diff, "ci90": [a, b], "margin": margin, "equivalent": a > -margin and b < margin,
            "noninferior": a > -margin, "ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    for x in ["greedy", "medcase", "closed", "out"]:
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    G, M, C = (json.load(open(p)) for p in (a.greedy, a.medcase, a.closed))
    out = {"greedy_top1": {}, "case_alt_list": {}, "case_stated_vs_agree": {}, "closed_stated_vs_agree": {}}
    for k, v in G["pairs"].items():
        t = v["top1"]
        out["greedy_top1"][k] = tost(t["diff"], *t["cluster_ci"], 0.05)
    for k, v in M["pairs"].items():
        t = v["alt_list"]
        out["case_alt_list"][k] = tost(t["diff"], *t["ci"], 0.05)
    for t, m in M["models"].items():
        if "d_auroc_stated-agree32" in m:
            d, lo, hi = m["d_auroc_stated-agree32"]
            out["case_stated_vs_agree"][t] = tost(d, lo, hi, 0.03)
    for t, w in C["within"].items():
        v = w["auroc_stated-agree32"]
        out["closed_stated_vs_agree"][t] = tost(v["diff"], *v["ci"], 0.03)
    for sec, d in out.items():
        print("==", sec)
        for k, v in d.items():
            print(f"  {k:55s} {v['diff']:+.3f} ci90 [{v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f}] equiv={v['equivalent']} noninf={v['noninferior']}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
