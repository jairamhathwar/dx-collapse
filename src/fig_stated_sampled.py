"""Figure: base -> post-trained change in the stated vs sampled readouts (vector PDF).

    python fig_stated_sampled.py --review2 review2_zs.json review2_third.json --aggregate aggregate_zs.json aggregate_third.json \
        --cons-dir CONS --out paper/figures/fig_stated_sampled.pdf

Panel a: total-variation distance to the reference differential (lower is better) of one stated list
(per-case mean over valid lists) and of the sampled top answers (review2.py), plus the mean of the base
model's first five stated lists (aggregate_lists.py, grey diamond, drawn just above the stated arrow so that it
cannot hide the post-trained marker).
Panel b: AUROC for the single-call answer being correct, of its stated probability and of agreement across
31 further samples (consequences.py).
Each arrow runs from the base model (hollow marker) to its post-trained model (filled marker).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATED, SAMPLED = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 of the reference palette
INK, MUTED, AGG = "#0b0b0b", "#52514e", "#52514e"
F, T = "fewshot_t07", "think_t07"  # keys are normalized to the part after /runs/
ROWS = [  # label, review2/aggregate dir key, base tag, post tag, cons file, cons-dir key for tags
    ("Qwen3-8B, $T{=}0.7$", "pair_q3", "qwen3-8b-base", "qwen3-8b", "pair_q3"),
    ("Qwen3-8B, $T{=}1$", "ddx_qwen3_t1", "qwen3-8b-base", "qwen3-8b", "ddx_qwen3_t1"),
    ("Qwen2.5-7B, $T{=}0.7$", "pair25_t07", "qwen25-7b-base", "qwen25-7b-it", "pair25_t07"),
    ("Qwen2.5-7B, $T{=}1$", "pair25_t1", "qwen25-7b-base", "qwen25-7b-it", "pair25_t1"),
    ("Qwen2.5-32B, $T{=}0.7$", "big32_ddx", "qwen25-32b-base", "qwen25-32b-it", "big32_q"),
    ("OLMo-3-7B, $T{=}0.7$", "ddx_olmo", "olmo3-base", "olmo3-instruct", "ddx_olmo"),
    ("OLMo-3-7B, $T{=}1$", "ddx_olmo_t1", "olmo3-base", "olmo3-instruct", "ddx_olmo_t1"),
    ("OLMo-3.1-32B, $T{=}0.7$", "big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct", "big32_o"),
    ("Mistral-7B, few-shot", F, "mistral7b-base", "mistral7b-it", "cons_fewshot_t07"),
    ("Qwen3-8B thinking, $T{=}0.7$", T, "qwen3-8b-base", "qwen3-8b-think", "cons_think_t07"),
]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
})


def arrow(ax, y, x0, x1, color, marker):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=4, shrinkB=4, mutation_scale=8))
    ax.plot([x0], [y], marker=marker, ms=6, mfc="white", mec=color, mew=1.4, ls="none", zorder=3)
    ax.plot([x1], [y], marker=marker, ms=6, mfc=color, mec="white", mew=0.8, ls="none", zorder=3)


def merged(paths, key):
    out = {}
    for p in paths:
        out.update({k.split("/runs/")[-1]: v for k, v in json.load(open(p))[key].items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review2", nargs="+", required=True)
    ap.add_argument("--aggregate", nargs="+", required=True)
    ap.add_argument("--cons-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", help="comma-separated row indices to draw (default: all); e.g. 0,2,4,5,7 for the T=0.7 pairs")
    ap.add_argument("--height", type=float, default=3.6)
    a = ap.parse_args()
    rows = [ROWS[int(i)] for i in a.rows.split(",")] if a.rows else ROWS
    R = merged(a.review2, "models")
    G = merged(a.aggregate, "models")
    C = Path(a.cons_dir)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, a.height), sharey=True, gridspec_kw={"wspace": 0.08})
    off = 0.13
    for i, (label, d, base, post, cf) in enumerate(rows):
        y = len(rows) - 1 - i
        rb, rp = R[f"{d}/{base}"], R[f"{d}/{post}"]
        cons = json.load(open(C / f"{cf}.json"))["models"]
        arrow(axes[0], y + off, rb["tv_list"], rp["tv_list"], STATED, "o")
        axes[0].plot([G[f"{d}/{base}"]["agg@5:tv"]], [y + 2.6 * off], marker="D", ms=4.0, mfc="white", mec=AGG, mew=1.0, ls="none", zorder=4)
        arrow(axes[0], y - off, rb["tv_samples"], rp["tv_samples"], SAMPLED, "s")
        arrow(axes[1], y + off, cons[base]["auroc_stated_p"], cons[post]["auroc_stated_p"], STATED, "o")
        arrow(axes[1], y - off, cons[base]["auroc_consist@32"], cons[post]["auroc_consist@32"], SAMPLED, "s")

    labels = [r[0] for r in rows][::-1]
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].set_xlabel("TV distance to reference differential (lower is better)")
    axes[0].set_title("(a) Distance to the reference differential", loc="left")
    axes[1].set_xlabel("AUROC, single-call answer correct (higher is better)")
    axes[1].set_title("(b) Error signal", loc="left")
    axes[0].set_xlim(0.64, 0.87)
    axes[1].set_xlim(0.50, 0.89)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e5e1", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        groups = [2, 4, 5, 7, 8, 9] if not a.rows else [i for i in range(1, len(rows))]
        for b in groups:  # family/size groups (all rows) or one band per pair (subset), counted from the top
            ax.axhline(len(rows) - b - 0.5, color="#d0cfca", lw=0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=STATED, marker="o", ms=5, lw=1.4, label="Stated: one list (a), its stated probability (b)"),
        Line2D([], [], color=SAMPLED, marker="s", ms=5, lw=1.4, label="Sampled: top answers (a), agreement of 31 samples (b)"),
        Line2D([], [], color=MUTED, marker="o", ms=5, mfc="white", lw=0, label="base model"),
        Line2D([], [], color=MUTED, marker="o", ms=5, lw=0, label="post-trained model"),
        Line2D([], [], color=AGG, marker="D", ms=4.5, mfc="white", lw=0, label="base model, mean of five lists (a)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.10 if not a.rows else -0.25))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(str(a.out).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
