"""Paper-style figures (vector PDF) with 95% case-bootstrap bands, built from graded.jsonl and per_condition.json.

    python paper_figures.py --runs $THESIS/runs --out $THESIS/code/paper/figures

Figures:
    fig_passk_pairs.pdf      pass@k, base vs post-trained (Qwen3-8B and Qwen2.5-7B, T=1), gold and gold-or-in-differential
    fig_passk_medical.pdf    pass@k, Qwen2.5-7B-Instruct vs its medical descendants (T=0.7)
    fig_passk_grpo8b.pdf     pass@k, Qwen3-8B-Base vs our GRPO runs (2 seeds pooled per recipe) vs official Qwen3-8B
    fig_percond.pdf          per-condition change in consideration, Qwen2.5-7B Instruct minus base (T=1)
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KS = [1, 2, 4, 8, 16, 32]
N_BOOT = 2000
SEV_COLORS = {1: "#b2182b", 2: "#ef8a62", 3: "#999999", 4: "#67a9cf", 5: "#2166ac"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def case_matrix(paths, field):
    """cases x ks matrix of pass@k, averaged over the given runs (seeds) per case; rows keyed by case_id."""
    per_run = []
    for p in paths:
        rows = {}
        for line in open(p):
            r = json.loads(line)
            rows[r["case_id"]] = [pass_at_k(r["n"], r[field], k) for k in KS]
        per_run.append(rows)
    ids = sorted(set.intersection(*(set(r) for r in per_run)))
    return ids, np.array([[np.mean([run[i][j] for run in per_run]) for j in range(len(KS))] for i in ids])


def curve(ax, runs, rng, field="c_exact"):
    """runs: list of (label, [graded paths], style dict). All runs are restricted to their common cases."""
    mats = {}
    for label, paths, _ in runs:
        mats[label] = case_matrix(paths, field)
    common = sorted(set.intersection(*(set(ids) for ids, _ in mats.values())))
    idx = rng.integers(0, len(common), size=(N_BOOT, len(common)))
    for label, _, style in runs:
        ids, m = mats[label]
        pos = {c: i for i, c in enumerate(ids)}
        m = m[[pos[c] for c in common]]
        mean = m.mean(0)
        boot = m[idx].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        ax.plot(KS, mean, label=label, **style)
        ax.fill_between(KS, lo, hi, color=style.get("color"), alpha=0.18, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.set_xlabel("$k$ (samples)")
    ax.grid(alpha=0.25, linewidth=0.4)
    return len(common)


def g(runs_dir, d, tag):
    return Path(runs_dir) / d / tag / "graded.jsonl"


BASE = {"color": "#2166ac", "linestyle": "--", "marker": "o", "markersize": 2.5}
POST = {"color": "#b2182b", "linestyle": "-", "marker": "s", "markersize": 2.5}


def fig_pairs(R, out, rng):
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 1.85), sharey=False)
    specs = [("Qwen3-8B", "ddx_qwen3_t1", "qwen3-8b-base", "qwen3-8b", "Base", "Post-trained"),
             ("Qwen2.5-7B", "pair25_t1", "qwen25-7b-base", "qwen25-7b-it", "Base", "Instruct")]
    j = 0
    for name, d, b, p, bl, pl in specs:
        for field, crit in (("c_exact", "gold"), ("c_related", "gold or in ref. ddx")):
            ax = axes[j]
            n = curve(ax, [(bl, [g(R, d, b)], BASE), (pl, [g(R, d, p)], POST)], rng, field)
            ax.set_title(f"{name}: {crit}")
            j += 1
    axes[0].set_ylabel("pass@$k$")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_passk_pairs.pdf")
    plt.close(fig)


def fig_medical(R, out, rng):
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    runs = [("Qwen2.5-7B-Instruct", [g(R, "ddx", "qwen25-7b-it")], {"color": "#555555", "linestyle": "--", "marker": "o", "markersize": 2.5}),
            ("AlphaMed (RL)", [g(R, "ddx", "alphamed-7b")], {"color": "#b2182b", "marker": "s", "markersize": 2.5}),
            ("HuatuoGPT-o1 (SFT+RL)", [g(R, "ddx", "huatuo-o1-7b")], {"color": "#ef8a62", "marker": "^", "markersize": 2.5}),
            ("m1 (SFT, 8k tokens)", [g(R, "ddx", "m1-7b")], {"color": "#2166ac", "marker": "D", "markersize": 2.2})]
    curve(ax, runs, rng)
    ax.set_ylabel("pass@$k$ (gold)")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(out / "fig_passk_medical.pdf")
    plt.close(fig)


def fig_grpo8b(R, out, rng):
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    runs = [("Qwen3-8B-Base", [g(R, "e9_ddx", "qwen3-8b-base")], BASE),
            ("+ acc GRPO", [g(R, "e9_ddx", f"b8-acc-s{s}") for s in (0, 1)], {"color": "#999999", "marker": "^", "markersize": 2.5}),
            ("+ gm GRPO (ours)", [g(R, "e9_ddx", f"b8-gm-s{s}") for s in (0, 1)], {"color": "#1b7837", "marker": "D", "markersize": 2.2}),
            ("Qwen3-8B (official)", [g(R, "e9_ddx", "qwen3-8b")], POST)]
    curve(ax, runs, rng)
    ax.set_ylabel("pass@$k$ (gold)")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "fig_passk_grpo8b.pdf")
    plt.close(fig)


def fig_percond(pc_json, out):
    d = json.load(open(pc_json))
    rows = []
    for c, v in d["conditions"].items():
        b, p = v["base"]["considered_top1"], v["instruct"]["considered_top1"]
        if b is None or p is None or v["base"].get("n_considered_cases", 0) < 5:
            continue
        rows.append((p - b, c, v["severity"]))
    rows.sort()
    fig, ax = plt.subplots(figsize=(3.25, 5.4))
    y = np.arange(len(rows))
    ax.barh(y, [100 * r[0] for r in rows], color=[SEV_COLORS[r[2]] for r in rows], height=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1].replace("Larygospasm", "Laryngospasm") for r in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Change in Considered@32 (points)")
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEV_COLORS[s]) for s in (1, 2, 3, 4, 5)]
    ax.legend(handles, ["Severity 1 (most severe)", "2", "3", "4", "5"], frameon=False, loc="upper right", fontsize=6)
    fig.savefig(out / "fig_percond.pdf")
    plt.close(fig)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--percond", default=None, help="per_condition.json for the Qwen2.5 T=1 pair")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    fig_pairs(a.runs, out, rng)
    fig_medical(a.runs, out, rng)
    fig_grpo8b(a.runs, out, rng)
    if a.percond:
        print("conditions plotted:", fig_percond(a.percond, out))
    print("wrote", sorted(p.name for p in out.glob("fig_*.pdf")))


if __name__ == "__main__":
    main()
