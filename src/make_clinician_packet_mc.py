"""Clinician rating packet, part B (published case reports): 15 cases.

    python make_clinician_packet_mc.py --cases cases_test.jsonl --core medcasereasoning_core.pqt \
        --authors-ddx authors_ddx_clean.jsonl --judge judge.jsonl --ddx-judge judge_ddx.jsonl --out-dir DIR [--seed 0]

Per case (at least two alternatives kept): the presentation shown to the models, the report's final diagnosis, the
report's diagnostic-reasoning notes, and
  (1) each alternative extracted from the notes (after dropping category words): did the authors consider it as an
      alternative diagnosis? (checks the extraction)
  (2) four model answers paired with a target (the final diagnosis, or an authors' alternative the answer was matched
      to): do they name the same disease? Two pairs the judge accepted and two it rejected where possible, in random
      order; the judge's verdict is hidden (checks the judge).
Writes part_B_cases.csv (for the clinician) and key_B_cases.csv (judge verdicts; not for the clinician).
"""
import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    for x in ["cases", "core", "authors-ddx", "judge", "ddx-judge", "out-dir"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    core = pd.read_parquet(a.core).set_index("pmcid")
    dd = {r["case_id"]: r for r in map(json.loads, open(a.authors_ddx))}
    J, K = {}, {}
    for r in map(json.loads, open(a.judge)):
        J.setdefault(r["case_id"], []).append(r)
    for r in map(json.loads, open(a.ddx_judge)):
        K.setdefault(r["case_id"], []).append(r)
    rng = random.Random(a.seed)
    pool = [cid for cid in cases if cid in dd and dd[cid].get("n_kept", 0) >= 2 and cid in core.index
            and any(r["match"] for r in J.get(cid, [])) and any(not r["match"] for r in J.get(cid, []))]
    rng.shuffle(pool)
    rows, keys = [], []
    for i, cid in enumerate(pool[:15], 1):
        c, d = cases[cid], dd[cid]
        alts = [d["ddx"][j - 1] for j in d["keep"]]
        acc = [r for r in J[cid] if r["match"]]
        rej = [r for r in J[cid] if not r["match"]]
        rng.shuffle(acc)
        rng.shuffle(rej)
        pairs = [(r["cand"], c["gold"], True) for r in acc[:1]] + [(r["cand"], c["gold"], False) for r in rej[:2]]
        km = [r for r in K.get(cid, []) if r["item"] and r["item"] >= 2]
        rng.shuffle(km)
        pairs += [(r["cand"], d["ddx"][r["item"] - 1], True) for r in km[:1]]
        if len(pairs) < 4:
            pairs += [(r["cand"], c["gold"], True) for r in acc[1:1 + 4 - len(pairs)]]
        rng.shuffle(pairs)
        row = {"case": f"B{i:02d}", "presentation shown to the models": c["vignette"],
               "final diagnosis (report)": c["gold"],
               "authors' diagnostic reasoning (from the report)": core.loc[cid, "diagnostic_reasoning"]}
        key = {"case": f"B{i:02d}", "case_id": cid}
        for j, alt in enumerate(alts[:6], 1):
            row[f"alternative {j}"] = alt
            row[f"Q4 alternative {j}: authors considered it? (Y/N/unsure)"] = ""
        for j, (cand, target, verdict) in enumerate(pairs[:4], 1):
            row[f"pair {j}"] = f"Model answer: {cand}  |  Target: {target}"
            row[f"Q5 pair {j}: same disease? (Y/N/partly)"] = ""
            key[f"pair {j} judge verdict"] = "same" if verdict else "different"
        row["comments"] = ""
        rows.append(row)
        keys.append(key)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [("part_B_cases.csv", rows), ("key_B_cases.csv", keys)]:
        cols = list(dict.fromkeys(k for r in data for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print("eligible", len(pool), "chosen", len(rows))


if __name__ == "__main__":
    main()
