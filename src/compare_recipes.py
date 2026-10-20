"""Paired case-level bootstrap between two reward recipes, averaging each recipe over its seeds.

    python compare_recipes.py --runs-dir $THESIS/runs/e7v2_ddx --a gm-s0,gm-s1 --b acc-s0,acc-s1

Per case, a metric is first averaged over the recipe's seeds; the difference (A - B) is then
bootstrapped over cases. Metrics: pass@1, pass@32 (exact), distinct top-1, entropy, TV, coverage@32.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(run_dir, tag):
    rows = {r["case_id"]: r for r in map(json.loads, open(Path(run_dir) / tag / "graded.jsonl"))}
    return {cid: {"pass@1": pass_at_k(r["n"], r["c_exact"], 1), "pass@32": pass_at_k(r["n"], r["c_exact"], 32),
                  "distinct": r["distinct"], "entropy": r["entropy"], "TV": r.get("tv"),
                  "cov@32": r.get("ddx_cov@32")} for cid, r in rows.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    A = [load(a.runs_dir, t) for t in a.a.split(",")]
    B = [load(a.runs_dir, t) for t in a.b.split(",")]
    cases = sorted(set.intersection(*(set(x) for x in A + B)))
    rng = np.random.default_rng(0)
    print(f"{len(cases)} cases;  A = {a.a}  vs  B = {a.b}  (A - B, 95% CI)")
    for m in ["pass@1", "pass@32", "distinct", "entropy", "TV", "cov@32"]:
        if A[0][cases[0]][m] is None:
            continue
        va = np.array([np.mean([x[c][m] for x in A]) for c in cases])
        vb = np.array([np.mean([x[c][m] for x in B]) for c in cases])
        d = va - vb
        boots = d[rng.integers(0, len(d), size=(a.n_boot, len(d)))].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {m:9s} A={va.mean():.4f} B={vb.mean():.4f}  diff={d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")


if __name__ == "__main__":
    main()
