"""Stated vs sampled readouts on real rare-disease cases (Phenopacket Store): aggregation and uncertainty.

    python pheno_consequences.py --runs-dir RUNS/go_full_pair25 --tags qwen25-7b-base,qwen25-7b-it \
        --cases cases.jsonl --normalizer normalizer.pkl --out out.json

There is no reference differential here, so this mirrors consequences.py only. Valid samples only;
answers are graded against the case's OMIM diagnosis via Mondo (exact / related / wrong), and top-1
answers are grouped by Mondo id (or normalized text if unmapped). Two targets: exact, and exact-or-related.
Per model:
  single              the first valid sample's top-1 is correct
  vote@5, vote@32     the most frequent top-1 among the first 5 / all valid samples is correct
  top3_list1          the first valid sample's top 3 contains a correct answer
  top3_vote32         the 3 most frequent top-1 answers over all valid samples contain a correct answer
  AUROC for the single-call answer being correct, of
      stated_p        its stated probability, renormalized over the list (uniform if missing)
      agree5, agree32 the share of the next 4 / all other valid samples with the same top-1
Differences come with paired case-bootstrap 95% intervals.
"""
import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from consequences import auroc
from disease_normalizer import DiseaseNormalizer, norm_text
from parsing import parse_dx

TARGETS = {"exact": ("exact",), "related": ("exact", "related")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--normalizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    norm = DiseaseNormalizer.load(a.normalizer)
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}

    @lru_cache(maxsize=None)
    def grade(name, gold):
        g = norm.grade(name, gold)
        return g["tier"], (g["mondo"] or "txt:" + norm_text(name))

    rng = np.random.default_rng(0)
    report = {}
    for tag in a.tags.split(","):
        rows = []
        for rec in map(json.loads, open(Path(a.runs_dir) / tag / "samples.jsonl")):
            gold = cases[rec["case_id"]]["disease_id"]
            samples = []  # (top key, top tier, stated p of top, [tiers of list items])
            for s in rec["samples"]:
                dx = parse_dx(s["text"])
                if not dx:
                    continue
                graded = [grade(d["diagnosis"], gold) for d in dx]
                ps = [d["probability"] for d in dx]
                if any(p is None or not (0 <= p <= 1) for p in ps) or sum(ps) <= 0:
                    ps = [1.0] * len(dx)
                samples.append((graded[0][1], graded[0][0], ps[0] / sum(ps), [t for t, _ in graded]))
            if not samples:
                rows.append(None)
                continue
            key_tier = {}
            for k, t, _, _ in samples:
                key_tier.setdefault(k, t)
            keys = [k for k, _, _, _ in samples]
            c32, c5 = Counter(keys), Counter(keys[:5])
            A = keys[0]
            top3v = [k for k, _ in c32.most_common(3)]
            row = {"valid": len(samples), "stated_p": samples[0][2],
                   "agree5": float(np.mean([k == A for k in keys[1:5]])) if len(keys) > 1 else None,
                   "agree32": float(np.mean([k == A for k in keys[1:]])) if len(keys) > 1 else None}
            for name, ok in TARGETS.items():
                row[f"single_{name}"] = float(samples[0][1] in ok)
                row[f"vote5_{name}"] = float(key_tier[c5.most_common(1)[0][0]] in ok)
                row[f"vote32_{name}"] = float(key_tier[c32.most_common(1)[0][0]] in ok)
                row[f"top3list_{name}"] = float(any(t in ok for t in samples[0][3][:3]))
                row[f"top3vote_{name}"] = float(any(key_tier[k] in ok for k in top3v))
            rows.append(row)

        valid_rows = [r for r in rows if r]
        out = {"n_cases": len(rows), "n_valid_cases": len(valid_rows),
               "mean_valid_samples": float(np.mean([r["valid"] for r in valid_rows]))}
        full = [r if r else {k: 0.0 for k in valid_rows[0] if k.startswith(("single", "vote", "top3"))} for r in rows]
        for name in TARGETS:
            for m in ["single", "vote5", "vote32", "top3list", "top3vote"]:
                out[f"{m}_{name}"] = float(np.mean([r[f"{m}_{name}"] for r in full]))
            for a_, b_ in [("vote32", "single"), ("top3list", "top3vote")]:
                d = np.array([r[f"{a_}_{name}"] - r[f"{b_}_{name}"] for r in full])
                bo = d[rng.integers(0, len(d), size=(a.n_boot, len(d)))].mean(1)
                out[f"d_{a_}-{b_}_{name}"] = [float(d.mean()), *map(float, np.percentile(bo, [2.5, 97.5]))]
            sub = [r for r in valid_rows if r["agree32"] is not None]
            y = np.array([r[f"single_{name}"] for r in sub])
            S = {s: np.array([r[s] for r in sub]) for s in ["stated_p", "agree5", "agree32"]}
            out[f"n_pos_{name}"] = int(y.sum())
            for s in S:
                out[f"auroc_{s}_{name}"] = auroc(S[s], y)
            for other in ["agree5", "agree32"]:
                diffs = []
                for _ in range(a.n_boot):
                    i = rng.integers(0, len(sub), size=len(sub))
                    diffs.append(auroc(S["stated_p"][i], y[i]) - auroc(S[other][i], y[i]))
                lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
                out[f"d_auroc_stated-{other}_{name}"] = [out[f"auroc_stated_p_{name}"] - out[f"auroc_{other}_{name}"], float(lo), float(hi)]
        report[tag] = out
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else ([round(x, 4) for x in v] if isinstance(v, list) else v))
                               for k, v in out.items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
