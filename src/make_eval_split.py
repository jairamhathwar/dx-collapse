"""Build disease-balanced evaluation subsets of Phenopacket cases.

Drops cases whose gold label does not grade 'exact' against itself (ambiguous group-level
labels such as "Citrullinemia"), then takes up to --per-disease cases per disease so that
common-in-the-literature diseases do not dominate. Writes case-id lists:
    eval_full.txt   all selected cases
    eval_go.txt     a smaller random subset for the go/no-go run
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from disease_normalizer import DiseaseNormalizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-disease", type=int, default=2)
    ap.add_argument("--go-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    norm = DiseaseNormalizer.load(a.normalizer)
    rng = random.Random(a.seed)
    by_dz, dropped = defaultdict(list), 0
    for c in map(json.loads, open(a.cases)):
        if norm.grade(c["disease_label"], c["disease_id"])["tier"] != "exact":
            dropped += 1
            continue
        by_dz[c["disease_id"]].append(c)

    chosen = []
    for dz in sorted(by_dz):
        cs = by_dz[dz]
        rng.shuffle(cs)
        # prefer recently published cases (less likely to be memorized) when available
        cs.sort(key=lambda c: (c.get("pub_year") or 0) < 2025)  # stable: 2025+ first
        chosen += cs[:a.per_disease]
    rng.shuffle(chosen)
    go = chosen[:a.go_size]

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_full.txt").write_text("\n".join(c["case_id"] for c in chosen) + "\n")
    (out / "eval_go.txt").write_text("\n".join(c["case_id"] for c in go) + "\n")

    def desc(cs):
        recent = sum((c.get("pub_year") or 0) >= 2025 for c in cs)
        return f"{len(cs)} cases, {len({c['disease_id'] for c in cs})} diseases, {recent} published 2025+"

    print(f"dropped {dropped} ambiguous-gold cases")
    print("eval_full:", desc(chosen))
    print("eval_go:  ", desc(go))


if __name__ == "__main__":
    main()
