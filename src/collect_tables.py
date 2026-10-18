"""Collect scored comparisons into one JSON (for paper tables), so numbers come from files, not chat.

    python collect_tables.py --runs $THESIS/runs --out $THESIS/code/results/paper_numbers.json

Each entry: run dir (with report/report.json and <model>/scores.json), the base/reference model,
and a label. Missing runs are skipped with a note.
"""
import argparse
import json
from pathlib import Path

COMPARISONS = [
    # label, dir, dataset, models (first = reference)
    ("qwen3_t07", "pair_q3", "ddx", ["qwen3-8b-base", "qwen3-8b"]),
    ("qwen3_t10", "ddx_qwen3_t1", "ddx", ["qwen3-8b-base", "qwen3-8b"]),
    ("qwen25_t10", "peek_t1_25", "ddx", ["qwen25-7b-base", "qwen25-7b-it"]),
    ("qwen25_med_t07", "ddx_peek", "ddx", ["qwen25-7b-it", "alphamed-7b"]),
    ("pheno_qwen25", "go", "go", ["qwen25-7b-it", "alphamed-7b", "huatuo-o1-7b", "m1-7b"]),
]


def model_row(root, tag, rep, dataset):
    m = rep["models"].get(tag)
    sp = root / tag / "scores.json"
    if m is None or not sp.exists():
        return None
    s = json.load(open(sp))
    c = s["calibration"]
    row = {
        "n_cases": rep["n_cases"], "pass_at": dict(zip(map(str, rep["ks"]), m["pass_exact"])),
        "pass_related_at": dict(zip(map(str, rep["ks"]), m["pass_related"])),
        "distinct_top1": m["distinct_top1"], "entropy_top1": m["entropy_top1"],
        "ece": c["ece"], "mean_conf": c["mean_conf"], "accuracy": c["accuracy"],
        "unparsed_samples": s["unparsed_samples"],
    }
    if dataset == "ddx":
        row.update({k: s[k] for k in ("tv_to_gt_ddx", "gt_entropy", "ddx_cov@1", "ddx_cov@32", "ddx_list_recall")})
    for key in ("delta_vs_base_c_exact",):
        if key in m:
            row["delta_exact_vs_report_base"] = m[key]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for label, d, dataset, models in COMPARISONS:
        root = Path(a.runs) / d
        rp = root / "report" / "report.json"
        if not rp.exists():
            out[label] = {"missing": str(rp)}
            continue
        rep = json.load(open(rp))
        out[label] = {"dir": d, "dataset": dataset, "reference": models[0],
                      "models": {t: r for t in models if (r := model_row(root, t, rep, dataset))}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for label, v in out.items():
        if "missing" in v:
            print(f"{label}: MISSING {v['missing']}")
            continue
        for t, r in v["models"].items():
            extra = f" TV={r['tv_to_gt_ddx']:.3f} cov@32={r['ddx_cov@32']:.3f}" if "tv_to_gt_ddx" in r else ""
            print(f"{label:15s} {t:15s} n={r['n_cases']:4d} p@1={r['pass_at']['1']:.3f} p@32={r['pass_at']['32']:.3f} "
                  f"distinct={r['distinct_top1']:.2f} H={r['entropy_top1']:.2f} conf={r['mean_conf']:.3f} "
                  f"acc={r['accuracy']:.3f} ECE={r['ece']:.3f}{extra}")


if __name__ == "__main__":
    main()
