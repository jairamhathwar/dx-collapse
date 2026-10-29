"""Clinician rating packet, part A (DDXPlus): 15 patients, four blinded readouts each.

    python make_clinician_packet_ddx.py --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --out-dir DIR [--pair pair_q3:qwen3-8b-base>qwen3-8b] [--seed 0]

Per patient: the vignette, the reference differential (its five most probable conditions) and four readouts in a
random order labelled A-D: one stated list of the base model and of the post-trained model (first valid sample, as
in the paper) and the distinct top answers of each model's 32 samples, with counts. Patients are stratified (5 each):
the post-trained samples never name the true pathology while its first list does; both readouts name it; any other
case. One patient per pathology. Writes part_A_ddxplus.csv (for the clinician) and key_A_ddxplus.csv (which readout
is which; not for the clinician).
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "cases", "conditions-json", "out-dir"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--pair", default="pair_q3:qwen3-8b-base>qwen3-8b")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    d, rest = a.pair.split(":")
    b, p = rest.split(">")

    def load(t):
        return {r["case_id"]: [stated_dist(parse_dx(s["text"]), match) for s in r["samples"]]
                for r in map(json.loads, open(f"{a.runs}/{d}/{t}/samples.jsonl"))}
    B, P = load(b), load(p)
    strata = {"miss_list_has": [], "both_have": [], "other": []}
    for cid, c in cases.items():
        pv, bv = [x for x in P.get(cid, []) if x[0]], [x for x in B.get(cid, []) if x[0]]
        if not pv or not bv:
            continue
        tops, first = {t for _, t in pv}, pv[0][0]
        g = c["gold"]
        key = "miss_list_has" if g not in tops and g in first else "both_have" if g in tops and g in first else "other"
        strata[key].append(cid)
    rng = random.Random(a.seed)
    chosen, used = [], set()
    for key in ["miss_list_has", "both_have", "other"]:
        pool = strata[key][:]
        rng.shuffle(pool)
        n = 0
        for cid in pool:
            if cases[cid]["gold"] in used:
                continue
            chosen.append((key, cid))
            used.add(cases[cid]["gold"])
            n += 1
            if n == 5:
                break
    rng.shuffle(chosen)

    def fmt_list(dist):
        return "; ".join(f"{k} ({v:.2f})" for k, v in sorted(dist.items(), key=lambda x: -x[1]))

    def fmt_samples(v):
        cnt = Counter(t for dd, t in v if dd and t)
        inv = sum(1 for dd, t in v if not (dd and t))
        s = "; ".join(f"{k} ({n})" for k, n in cnt.most_common())
        return s + (f"; [no usable answer] ({inv})" if inv else "")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, keys = [], []
    for i, (stratum, cid) in enumerate(chosen, 1):
        c = cases[cid]
        z = sum(pp for _, pp in c["gt_differential"])
        ref = sorted(((k, pp / z) for k, pp in c["gt_differential"]), key=lambda x: -x[1])[:5]
        readouts = [("base list", "One answer's ranked list (probabilities)", fmt_list(next(x[0] for x in B[cid] if x[0]))),
                    ("post-trained list", "One answer's ranked list (probabilities)", fmt_list(next(x[0] for x in P[cid] if x[0]))),
                    ("base samples", "Top answers of 32 repeated calls (counts)", fmt_samples(B[cid])),
                    ("post-trained samples", "Top answers of 32 repeated calls (counts)", fmt_samples(P[cid]))]
        rng.shuffle(readouts)
        row = {"patient": f"A{i:02d}", "vignette": c["vignette"],
               "reference differential (rule-based engine, top 5)": "; ".join(f"{k} ({pp:.2f})" for k, pp in ref),
               "Q1 reference plausible? (1-5)": ""}
        key = {"patient": f"A{i:02d}", "case_id": cid, "true_pathology": c["gold"], "stratum": stratum}
        for lab, (what, kind, text) in zip("ABCD", readouts):
            row[f"readout {lab}"] = f"[{kind}] {text}"
            row[f"Q2 readout {lab} adequate? (1-5)"] = ""
            row[f"Q3 readout {lab} omits a must-not-miss condition? (Y/N; which)"] = ""
            key[f"readout {lab}"] = what
        row["comments"] = ""
        rows.append(row)
        keys.append(key)
    for name, data in [("part_A_ddxplus.csv", rows), ("key_A_ddxplus.csv", keys)]:
        cols = list(dict.fromkeys(k for r in data for k in r))
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    print({k: len(v) for k, v in strata.items()}, "chosen", len(chosen))


if __name__ == "__main__":
    main()
