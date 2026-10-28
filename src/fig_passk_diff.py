"""Workshop Figure 2: post-trained minus base pass@k against k, one line per zero-shot pair (DDXPlus, T=0.7).

    python fig_passk_diff.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json --out fig.pdf

Top answers and validity as in every table (stated_vs_sampled.stated_dist; invalid samples count as wrong). Shaded
bands: 95% condition-clustered bootstrap intervals (2,000 resamples). Colors: Okabe-Ito (colorblind-safe); each line
also has its own marker and a direct label.
"""
import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist

PAIRS = [("Qwen3-8B", "pair_q3", "qwen3-8b-base", "qwen3-8b", "#0072B2", "o"),
         ("Qwen2.5-7B", "pair25_t07", "qwen25-7b-base", "qwen25-7b-it", "#E69F00", "s"),
         ("Qwen2.5-32B", "big32_ddx", "qwen25-32b-base", "qwen25-32b-it", "#009E73", "^"),
         ("OLMo 3 7B", "ddx_olmo", "olmo3-base", "olmo3-instruct", "#D55E00", "D"),
         ("OLMo 3.1 32B", "big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct", "#CC79A7", "v")]
KS = [1, 2, 4, 8, 16, 32]


def pk(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curves(path, gold, match):
    out = {}
    for rec in map(json.loads, open(path)):
        cid = rec["case_id"]
        t = [stated_dist(parse_dx(s["text"]), match)[1] for s in rec["samples"]]
        n, c = len(t), sum(x == gold[cid] for x in t)
        out[cid] = [pk(n, c, k) for k in KS]
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "cases", "conditions-json", "out"]:
        ap.add_argument(f"--{x}", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(a.cases))}
    keys = sorted(set(gold.values()))
    idx = {k: i for i, k in enumerate(keys)}
    rng = np.random.default_rng(0)
    W = rng.multinomial(len(keys), [1 / len(keys)] * len(keys), size=2000).astype(float)
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.3, 1.8))
    ax.axhline(0, color="#555555", lw=0.7, zorder=1)
    res = {}
    for nm, d, b, p, col, mk in PAIRS:
        B = curves(f"{a.runs}/{d}/{b}/samples.jsonl", gold, match)
        P = curves(f"{a.runs}/{d}/{p}/samples.jsonl", gold, match)
        cs = [c for c in gold if c in B and c in P]
        S = np.zeros((len(keys), len(KS)))
        N = np.zeros(len(keys))
        for c in cs:
            S[idx[gold[c]]] += np.array(P[c]) - np.array(B[c])
            N[idx[gold[c]]] += 1
        point = 100 * S.sum(0) / N.sum()
        boot = 100 * (W @ S) / (W @ N)[:, None]
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        res[nm] = {"k": KS, "diff": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
        ax.fill_between(KS, lo, hi, color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot(KS, point, color=col, lw=1.4, marker=mk, ms=3.2, zorder=3)
        res[nm]["col"] = col
    # direct labels at k = 32, pushed apart to at least 3.6 points
    order = sorted(res, key=lambda n: res[n]["diff"][-1])
    ys, gap = [], 4.2
    for n in order:
        y = res[n]["diff"][-1]
        ys.append(max(y, ys[-1] + gap) if ys else y)
    for n, y in zip(order, ys):
        ax.annotate(n, (KS[-1], res[n]["diff"][-1]), xytext=(KS[-1] * 1.18, y), textcoords="data", va="center",
                    fontsize=6.3, color="#222222",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.4, shrinkA=0, shrinkB=2))
        res[n].pop("col")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlim(0.85, 32 * 3.2)
    ax.set_xlabel("$k$ (samples per case)")
    ax.set_ylabel("pass@$k$: post $-$ base (points)")
    ax.grid(axis="y", color="#dddddd", lw=0.5, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    json.dump(res, open(a.out.replace(".pdf", ".json"), "w"), indent=1)
    for nm, r in res.items():
        print(nm, [round(x, 1) for x in r["diff"]])


if __name__ == "__main__":
    main()
