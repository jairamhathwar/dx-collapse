"""Case reports: distinct sampled diagnoses after merging paraphrases the judges matched to the same target.

    python r25_medcase_merge.py --cases cases_test.jsonl --runs RUNS/medcase_t07 --judge judge.jsonl \
        --ddx-judge judge_ddx.jsonl --pairs "base>post ..." [--normalizer normalizer.pkl] --out merge.json

Each valid sample's top answer is keyed as the final diagnosis if the primary judge accepted it, as the authors'
alternative it was matched to otherwise (judge_ddx item >= 2), and else by its Mondo concept (if --normalizer is given
and the name resolves) or its normalized text. Distinct = distinct keys among valid samples. Matched paraphrases are
merged and unmatched ones are not, so this is an upper bound on the number of distinct diagnoses; it is not used for
agreement, which the asymmetric merging would bias. Post-trained minus base with a paired case bootstrap (10,000).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from disease_normalizer import norm_text
from medcase_analysis import load


def main():
    ap = argparse.ArgumentParser()
    for x in ["cases", "runs", "judge", "ddx-judge", "pairs", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--normalizer")
    a = ap.parse_args()
    J = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge))}
    K = {(r["case_id"], r["key"]): r["item"] for r in map(json.loads, open(a.ddx_judge))}
    keyfn = None
    if a.normalizer:
        from disease_normalizer import DiseaseNormalizer
        dn, kc = DiseaseNormalizer.load(a.normalizer), {}

        def keyfn(name):
            t = norm_text(name)
            if t not in kc:
                ids, _ = dn.resolve(name)
                kc[t] = "mondo:" + sorted(ids)[0] if ids else "txt:" + t
            return kc[t]
    rng = np.random.default_rng(0)
    rows, report = {}, {"models": {}, "pairs": {}}
    for pair in a.pairs.split():
        b, p = pair.split(">")
        for t in (b, p):
            if t in rows:
                continue
            R = load(Path(a.runs) / t, J, K, keyfn)
            per = {}
            for c, samples in R.items():
                v = [s for s in samples if s]
                if not v:
                    continue
                keys = ["gold" if s["ok"][0] else (f"alt:{s['items'][0]}" if s["items"] and s["items"][0] >= 2 else s["key"]) for s in v]
                per[c] = {"distinct_raw": float(len({s["key"] for s in v})), "distinct_merged": float(len(set(keys)))}
            rows[t] = per
            report["models"][t] = {k: float(np.mean([r[k] for r in per.values()])) for k in ["distinct_raw", "distinct_merged"]}
            print(t, json.dumps({k: round(v, 2) for k, v in report["models"][t].items()}), flush=True)
        cs = [c for c in rows[p] if c in rows[b]]
        res = {}
        for k in ["distinct_raw", "distinct_merged"]:
            d = np.array([rows[p][c][k] - rows[b][c][k] for c in cs])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
            res[k] = {"diff": float(d.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": len(cs)}
        report["pairs"][pair] = res
        print(pair, json.dumps({k: [round(v["diff"], 2), [round(x, 2) for x in v["ci"]]] for k, v in res.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
