"""Compare scored runs: pass@k table, curve crossings, paired bootstrap CIs, and a figure.

    python compare_runs.py --runs-dir $THESIS/runs/go --base qwen25-7b-it --out-dir $THESIS/runs/go/report
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

KS = [1, 2, 4, 8, 16, 32]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_graded(path):
    return {r["case_id"]: r for r in map(json.loads, open(path))}


def curve(rows, key, ks=KS):
    return np.array([[pass_at_k(r["n"], r[key], k) for k in ks] for r in rows])  # cases x ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--title", default="Phenopacket rare-disease diagnosis")
    ap.add_argument("--related-label", default="Exact or related group")
    a = ap.parse_args()

    runs = {p.parent.name: load_graded(p) for p in sorted(Path(a.runs_dir).glob("*/graded.jsonl"))}
    common = sorted(set.intersection(*(set(g) for g in runs.values())))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {"n_cases": len(common), "ks": KS, "models": {}}

    curves = {}
    for tag, g in runs.items():
        rows = [g[c] for c in common]
        scores = json.load(open(Path(a.runs_dir) / tag / "scores.json"))
        curves[tag] = {k: curve(rows, k) for k in ("c_exact", "c_related")}
        report["models"][tag] = {
            "pass_exact": curves[tag]["c_exact"].mean(0).round(4).tolist(),
            "pass_related": curves[tag]["c_related"].mean(0).round(4).tolist(),
            "distinct_top1": float(np.mean([r["distinct"] for r in rows])),
            "entropy_top1": float(np.mean([r["entropy"] for r in rows])),
            "diseases_ever_solved": len({r["gold"] for r in rows if r["c_exact"] > 0}),
            "list_recall": float(np.mean([r["list_hits"] / r["n"] for r in rows])),
            "calibration": scores["calibration"],
            "unparsed": scores["unparsed_samples"],
        }

    # paired bootstrap of (model - base) at each k, over cases
    base = curves[a.base]
    for tag in runs:
        if tag == a.base:
            continue
        for key in ("c_exact", "c_related"):
            diff = curves[tag][key] - base[key]  # cases x ks
            idx = rng.integers(0, len(common), size=(a.n_boot, len(common)))
            boots = diff[idx].mean(1)  # n_boot x ks
            lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
            report["models"][tag][f"delta_vs_base_{key}"] = {
                "mean": diff.mean(0).round(4).tolist(), "ci_lo": lo.round(4).tolist(), "ci_hi": hi.round(4).tolist()}

    json.dump(report, open(out / "report.json", "w"), indent=2)

    # text table
    lines = [f"n_cases={len(common)}   pass@k (exact) for k={KS}"]
    for tag, m in report["models"].items():
        lines.append(f"{tag:16s} exact   " + " ".join(f"{v:.3f}" for v in m["pass_exact"]))
        lines.append(f"{'':16s} related " + " ".join(f"{v:.3f}" for v in m["pass_related"]))
        lines.append(f"{'':16s} distinct={m['distinct_top1']:.2f} entropy={m['entropy_top1']:.2f} "
                     f"solved={m['diseases_ever_solved']} list_recall={m['list_recall']:.3f} "
                     f"ECE={m['calibration']['ece']} conf={m['calibration']['mean_conf']} unparsed={m['unparsed']}")
        if "delta_vs_base_c_exact" in m:
            d = m["delta_vs_base_c_exact"]
            lines.append(f"{'':16s} Δexact  " + " ".join(
                f"{mu:+.3f}[{lo:+.3f},{hi:+.3f}]" for mu, lo, hi in zip(d["mean"], d["ci_lo"], d["ci_hi"])))
    (out / "table.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
        for ax, key, title in zip(axes, ("pass_exact", "pass_related"), ("Exact diagnosis", a.related_label)):
            for tag, m in report["models"].items():
                ax.plot(KS, m[key], marker="o", lw=2.5 if tag == a.base else 1.5,
                        ls="--" if tag == a.base else "-", label=tag)
            ax.set_xscale("log", base=2)
            ax.set_xticks(KS, [str(k) for k in KS])
            ax.set_xlabel("k (samples)")
            ax.set_ylabel("pass@k")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{a.title}, {len(common)} cases")
        fig.tight_layout()
        fig.savefig(out / "pass_at_k.png", dpi=150)
        print("figure ->", out / "pass_at_k.png")
    except ImportError:
        print("matplotlib not installed; skipped figure")


if __name__ == "__main__":
    main()
