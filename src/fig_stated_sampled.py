"""Figure: base -> post-trained change in the stated vs sampled readouts (vector PDF).

    python fig_stated_sampled.py --analysis $THESIS/runs/analysis --out $THESIS/code/paper/figures/fig_stated_sampled.pdf

Panel a: total-variation distance to the reference differential (lower is better) of one stated list
and of the sampled top-1 answers (from stated_vs_sampled.py, runs/analysis/svs/*.json).
Panel b: AUROC for the single-call answer being correct, of its stated probability and of agreement
across 31 further samples (from consequences.py, runs/analysis/cons/*.json).
Each arrow runs from the base model (hollow marker) to its post-trained model (filled marker).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATED, SAMPLED = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 of the reference palette
INK, MUTED = "#0b0b0b", "#52514e"
ROWS = [  # label, svs/cons file, base tag, post tag
    ("Qwen3-8B, $T{=}0.7$", "pair_q3", "qwen3-8b-base", "qwen3-8b"),
    ("Qwen3-8B, $T{=}1$", "ddx_qwen3_t1", "qwen3-8b-base", "qwen3-8b"),
    ("Qwen2.5-7B, $T{=}0.7$", "pair25_t07", "qwen25-7b-base", "qwen25-7b-it"),
    ("Qwen2.5-7B, $T{=}1$", "pair25_t1", "qwen25-7b-base", "qwen25-7b-it"),
    ("OLMo-3-7B, $T{=}0.7$", "ddx_olmo", "olmo3-base", "olmo3-instruct"),
    ("OLMo-3-7B, $T{=}1$", "ddx_olmo_t1", "olmo3-base", "olmo3-instruct"),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A = Path(a.analysis)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    off = 0.17
    for i, (label, d, base, post) in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        svs = json.load(open(A / "svs" / f"{d}.json"))["models"]
        cons = json.load(open(A / "cons" / f"{d}.json"))["models"]
        arrow(axes[0], y + off, svs[base]["tv_list"], svs[post]["tv_list"], STATED, "o")
        arrow(axes[0], y - off, svs[base]["tv_sampled"], svs[post]["tv_sampled"], SAMPLED, "s")
        arrow(axes[1], y + off, cons[base]["auroc_stated_p"], cons[post]["auroc_stated_p"], STATED, "o")
        arrow(axes[1], y - off, cons[base]["auroc_consist@32"], cons[post]["auroc_consist@32"], SAMPLED, "s")

    labels = [r[0] for r in ROWS][::-1]
    axes[0].set_yticks(range(len(ROWS)), labels)
    axes[0].set_xlabel("TV distance to reference differential (lower is better)")
    axes[0].set_title("(a) Distance to the reference differential", loc="left")
    axes[1].set_xlabel("AUROC, single-call answer correct (higher is better)")
    axes[1].set_title("(b) Error signal", loc="left")
    axes[0].set_xlim(0.68, 0.87)
    axes[1].set_xlim(0.50, 0.89)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e5e1", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        for y in [1.5, 3.5]:
            ax.axhline(y, color="#d0cfca", lw=0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=STATED, marker="o", ms=5, lw=1.4, label="Stated: one list (a), its stated probability (b)"),
        Line2D([], [], color=SAMPLED, marker="s", ms=5, lw=1.4, label="Sampled: top-1 answers (a), agreement of 31 samples (b)"),
        Line2D([], [], color=MUTED, marker="o", ms=5, mfc="white", lw=0, label="base model"),
        Line2D([], [], color=MUTED, marker="o", ms=5, lw=0, label="post-trained model"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.24))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(str(a.out).replace(".pdf", ".png"), dpi=200)


if __name__ == "__main__":
    main()
