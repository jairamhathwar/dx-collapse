"""One-table summary of the GRPO reward-recipe comparison (E7/E8).

    python summarize_e7.py --runs $THESIS/runs [--dataset ddx|go] [--prefix e7|e7v2]
"""
import argparse
import json
from pathlib import Path

ORDER = ["qwen3-1.7b", "acc-s0", "acc-s1", "dist-s0", "dist-s1", "brier-s0", "overlap-s0", "puri-rlvr-s0", "puri-rlcr-s0"]


def ci(d, i):
    return f"{d['mean'][i]:+.3f} [{d['ci_lo'][i]:+.3f},{d['ci_hi'][i]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dataset", choices=["ddx", "go"], default="ddx")
    ap.add_argument("--prefix", default="e7")
    a = ap.parse_args()
    root = Path(a.runs) / f"{a.prefix}_{a.dataset}"
    rep = json.load(open(root / "report" / "report.json"))["models"]
    tags = [t for t in ORDER if t in rep] + [t for t in rep if t not in ORDER]

    if a.dataset == "ddx":
        cols = ["p@1", "p@4", "p@32", "distinct", "entropy", "TV", "cov@1", "cov@32", "ddxRec", "ECE", "conf", "acc"]
    else:
        cols = ["p@1", "p@32", "rel@32", "distinct", "ECE", "conf", "acc", "unparsed"]
    print(f"{'tag':>13s} " + " ".join(f"{c:>8s}" for c in cols))
    for t in tags:
        m, s = rep[t], json.load(open(root / t / "scores.json"))
        c = s["calibration"]
        if a.dataset == "ddx":
            vals = [m["pass_exact"][0], m["pass_exact"][2], m["pass_exact"][-1], s["distinct_top1"], s["entropy_top1"],
                    s["tv_to_gt_ddx"], s["ddx_cov@1"], s["ddx_cov@32"], s["ddx_list_recall"], c["ece"], c["mean_conf"], c["accuracy"]]
        else:
            vals = [m["pass_exact"][0], m["pass_exact"][-1], m["pass_related"][-1], m["distinct_top1"],
                    c["ece"], c["mean_conf"], c["accuracy"], m["unparsed"]]
        print(f"{t:>13s} " + " ".join(f"{v:8.3f}" if isinstance(v, float) else f"{v:8d}" for v in vals))
    print("\nΔ pass@k vs untrained Qwen3-1.7B (95% CI):")
    for t in tags:
        d = rep[t].get("delta_vs_base_c_exact")
        if d:
            print(f"  {t:13s} k=1 {ci(d, 0)}   k=32 {ci(d, -1)}")


if __name__ == "__main__":
    main()
