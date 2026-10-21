"""Format-robustness check for base vs post-trained comparisons on DDXPlus.

Base models fail the output format more often, and unparsed samples count as wrong in the main metrics.
This reports, per comparison:
  (a) pass@k on the subset of cases where every sample of BOTH models parses;
  (b) pass@1 among parsed samples only (c_exact / n_parsed), over cases with >=1 parsed sample in both.
Differences are post-trained minus base with 95% case-bootstrap intervals.

    python format_robust.py --runs-dir $THESIS/runs/pair_q3 --base qwen3-8b-base --post qwen3-8b
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

KS = [1, 2, 4, 8, 16, 32]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load(d, tag):
    rows = {r["case_id"]: r for r in map(json.loads, open(Path(d) / tag / "graded.jsonl"))}
    assert all("n_parsed" in r for r in rows.values()), f"{tag}: rescore with current score_ddxplus.py"
    return rows


def ci(x, rng, n_boot=5000):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    b = x[idx].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--post", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    B, P = load(a.runs_dir, a.base), load(a.runs_dir, a.post)
    ids = sorted(set(B) & set(P))
    full = [i for i in ids if B[i]["n_parsed"] == B[i]["n"] and P[i]["n_parsed"] == P[i]["n"]]
    rate = 1 - sum(B[i]["n_parsed"] for i in ids) / sum(B[i]["n"] for i in ids)
    print(f"{a.runs_dir}: {a.post} vs {a.base}; base unparsed rate {rate:.3f}; "
          f"fully parsed cases {len(full)}/{len(ids)}")
    for k in KS:
        b = np.array([pass_at_k(B[i]["n"], B[i]["c_exact"], k) for i in full])
        p = np.array([pass_at_k(P[i]["n"], P[i]["c_exact"], k) for i in full])
        m, lo, hi = ci(p - b, rng)
        print(f"  fully-parsed pass@{k:<2d} base={b.mean():.3f} post={p.mean():.3f} diff={m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    both = [i for i in ids if B[i]["n_parsed"] and P[i]["n_parsed"]]
    b = np.array([B[i]["c_exact"] / B[i]["n_parsed"] for i in both])
    p = np.array([P[i]["c_exact"] / P[i]["n_parsed"] for i in both])
    m, lo, hi = ci(p - b, rng)
    print(f"  parsed-only pass@1 ({len(both)} cases) base={b.mean():.3f} post={p.mean():.3f} diff={m:+.3f} [{lo:+.3f},{hi:+.3f}]")


if __name__ == "__main__":
    main()
