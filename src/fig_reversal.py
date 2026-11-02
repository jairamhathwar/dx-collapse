"""Readout reversal figure: recall of the true pathology against the number of calls, sampled answers vs stated lists.

    python fig_reversal.py r25.json fig_reversal.pdf

One panel per pair (DDXPlus, T=0.7; zero-shot unless marked). Orange squares: pass@k of the sampled top answers.
Blue circles: the union of k calls' stated lists contains the true pathology. Hollow, dashed: base model; filled,
solid: post-trained model. Colors as in Figure 1 (stated blue, sampled orange).
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATED, SAMPLED, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"
PAIRS = [("Qwen3-8B", "qwen3-8b-base", "qwen3-8b"), ("Qwen2.5-7B", "qwen25-7b-base", "qwen25-7b-it"),
         ("Qwen2.5-32B", "qwen25-32b-base", "qwen25-32b-it"), ("Qwen2.5-72B", "qwen25-72b-base", "qwen25-72b-it"),
         ("OLMo 3 7B", "olmo3-base", "olmo3-instruct"), ("OLMo 3.1 32B", "olmo31-32b-base", "olmo31-32b-instruct"),
         ("Mistral-7B*", "mistral7b-base", "mistral7b-it"), ("MedGemma-4B*", "medgemma-4b-pt", "medgemma-4b-it")]


def main():
    R = json.load(open(sys.argv[1]))
    ks = R["ks"]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.5})
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.3), sharex=True, sharey=True)
    for ax, (title, b, p) in zip(axes.flat, PAIRS):
        key = lambda t: next(k for k in R["curves"] if k.endswith("/" + t))
        for t, ls, fill in [(b, "--", False), (p, "-", True)]:
            cv = R["curves"][key(t)]
            for vals, col, mk in [(cv["pass"], SAMPLED, "s"), (cv["list"], STATED, "o")]:
                ax.plot(ks, vals, ls=ls, color=col, lw=1.2, marker=mk, ms=3.6,
                        markerfacecolor=col if fill else "white", markeredgecolor=col, markeredgewidth=0.9, zorder=3 if fill else 2)
        ax.set_title(title, color=INK, pad=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(0.15, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.tick_params(length=2, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("calls $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("true pathology named")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=STATED, marker="o", lw=1.2, label="union of $k$ stated lists"),
         Line2D([], [], color=SAMPLED, marker="s", lw=1.2, label="$k$ sampled answers (pass@$k$)"),
         Line2D([], [], color=MUTED, ls="--", marker="o", markerfacecolor="white", lw=1.0, label="base model"),
         Line2D([], [], color=MUTED, ls="-", marker="o", lw=1.0, label="post-trained model")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), pad=0.3, w_pad=0.4, h_pad=0.6)
    fig.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
