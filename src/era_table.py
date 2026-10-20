"""Phenopacket pass@k by publication era of the source case report (memorization check).

    python era_table.py --runs-dir $THESIS/runs/go [--models a,b,c] [--out era.json]

Reads <model>/scores.json (score_samples.py writes a by_pub_era block) and prints, per era,
the number of cases and pass@1 / pass@32 (exact) and pass@32 (exact or related).
"""
import argparse
import json
from pathlib import Path

ERAS = ["<=2022", "2023-24", "2025+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--models")
    ap.add_argument("--out")
    a = ap.parse_args()
    root = Path(a.runs_dir)
    tags = a.models.split(",") if a.models else sorted(p.parent.name for p in root.glob("*/scores.json"))
    out = {}
    print(f"{'model':16s} " + "  ".join(f"{e:>24s}" for e in ERAS))
    print(f"{'':16s} " + "  ".join(f"{'n  p@1   p@32  rel@32':>24s}" for _ in ERAS))
    for t in tags:
        sp = root / t / "scores.json"
        if not sp.exists():
            continue
        by = json.load(open(sp))["by_pub_era"]
        out[t] = {e: {k: by[e].get(k) for k in ("n_cases", "pass@1", "pass@32", "pass_related@32")} for e in ERAS if e in by}
        cells = []
        for e in ERAS:
            v = by.get(e)
            cells.append(f"{v['n_cases']:4d} {v['pass@1']:.3f} {v['pass@32']:.3f} {v['pass_related@32']:.3f}" if v else f"{'-':>24s}")
        print(f"{t:16s} " + "  ".join(f"{c:>24s}" for c in cells))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
