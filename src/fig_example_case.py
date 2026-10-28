"""Workshop figure: one DDXPlus patient read both ways (Qwen3-8B pair, zero-shot, T=0.7).

    python fig_example_case.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --case ddx_test_120286 --out fig_example_case.pdf

Panels: the reference differential; the top answers of the base model's 32 samples; the top answers of the
post-trained model's 32 samples; the post-trained model's first valid stated list (renormalized). Top answers and
lists as in every table (stated_vs_sampled.stated_dist). Also prints, over all 980 patients, how often the
post-trained model's 32 samples never name the true pathology while its first list does.
"""
import argparse
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist

STATED, SAMPLED, REF, INK, MUTED = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e"
SHORT = {"Bronchospasm / acute asthma exacerbation": "Asthma exacerbation", "Acute rhinosinusitis": "Acute rhinosinusitis",
         "URTI": "URTI", "Allergic sinusitis": "Allergic sinusitis"}


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "cases", "conditions-json", "case", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--pair", default="pair_q3:qwen3-8b-base>qwen3-8b")
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    d, rest = a.pair.split(":")
    b, p = rest.split(">")

    def load(t):
        return {rec["case_id"]: [stated_dist(parse_dx(s["text"]), match) for s in rec["samples"]]
                for rec in map(json.loads, open(f"{a.runs}/{d}/{t}/samples.jsonl"))}
    B, P = load(b), load(p)
    n = hit = 0
    for cid, c in cases.items():
        pt = [t for _, t in P.get(cid, []) if t]
        pl = next((x for x, _ in P.get(cid, []) if x), None)
        if not pt or pl is None:
            continue
        n += 1
        hit += c["gold"] not in pt and c["gold"] in pl
    c = cases[a.case]
    z = sum(v for _, v in c["gt_differential"])
    ref = {k: v / z for k, v in c["gt_differential"]}
    bt = Counter(t for _, t in B[a.case] if t)
    pt = Counter(t for _, t in P[a.case] if t)
    pl = next(x for x, _ in P[a.case] if x)
    nb, np_ = len(B[a.case]), len(P[a.case])
    rows = list(ref) + [k for k in sorted(set(bt) | set(pt) | set(pl), key=lambda k: -(bt[k] + pt[k] + 32 * pl.get(k, 0))) if k not in ref]
    res = {"case": a.case, "gold": c["gold"], "reference": ref, "base_samples": dict(bt), "base_n": nb, "post_samples": dict(pt),
           "post_n": np_, "post_list": pl, "share_post_samples_miss_list_has": hit / n, "n_cases": n}
    print(json.dumps(res, indent=1))

    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.titlesize": 7, "axes.linewidth": 0.5})
    fig, axes = plt.subplots(1, 4, figsize=(3.3, 1.55), sharey=True, gridspec_kw={"wspace": 0.12})
    panels = [("Reference", [ref.get(k, 0) for k in rows], REF, "{:.2f}"),
              (f"Base model:\n{nb} samples", [bt[k] / nb for k in rows], SAMPLED, "count"),
              (f"Post-trained:\n{np_} samples", [pt[k] / np_ for k in rows], SAMPLED, "count"),
              ("Post-trained:\none list", [pl.get(k, 0) for k in rows], STATED, "{:.2f}")]
    y = range(len(rows))
    for ax, (title, vals, col, fmt) in zip(axes, panels):
        ax.barh(y, vals, color=col, height=0.62, zorder=2)
        for i, v in enumerate(vals):
            if v > 0:
                lab = str(round(v * (nb if "Base" in title else np_))) if fmt == "count" else fmt.format(v).lstrip("0")
                ax.text(v + 0.04, i, lab, va="center", ha="left", fontsize=6, color=MUTED)
        ax.set_xlim(0, 1.45)
        ax.set_xticks([])
        ax.set_title(title, fontsize=6.3, color=INK, pad=3)
        for s in ["top", "right", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#bdbcb7")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([SHORT.get(k, k) + (" (true)" if k == c["gold"] else "") for k in rows], fontsize=6.3)
    for t, k in zip(axes[0].get_yticklabels(), rows):
        if k == c["gold"]:
            t.set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    json.dump(res, open(a.out.replace(".pdf", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
