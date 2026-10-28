"""Draw a blind audit sample of case-report judge verdicts from the names string matching cannot label.

    python make_judge_audit.py --cases cases_test.jsonl --runs RUNS --judge judge_mistral24b.jsonl \
        --judge2 judge2_phi4.jsonl --pairs "qwen3-8b-base>qwen3-8b ..." --n 200 --out-dir AUDIT

Population: sampled top answers (first item of a valid list) that are neither silver positives nor silver negatives
(medcase_silver.py), i.e. that share some content word with the final diagnosis without matching it. Half are drawn
from base models' answers and half from post-trained models', each with probability proportional to how often the
(case, name) pair occurs as a top answer, without replacement. Writes audit.csv (for the grader: final diagnosis and
candidate only, in random order) and audit_key.csv (judges' verdicts and which models produced the name).
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict

from disease_normalizer import norm_text
from medcase_analysis import sample_files
from medcase_silver import silver
from parsing import parse_dx


def main():
    ap = argparse.ArgumentParser()
    for x in ["cases", "runs", "judge", "judge2", "pairs", "out-dir"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(a.cases))}
    J1 = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge))}
    J2 = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge2))}
    arm = {}
    for p in a.pairs.split():
        b, q = p.split(">")
        arm[b], arm[q] = "base", "post"
    count = {"base": Counter(), "post": Counter()}
    shown, models = {}, defaultdict(set)
    for t, side in arm.items():
        for rec in (json.loads(l) for f in sample_files(f"{a.runs}/{t}") for l in open(f)):
            cid = rec["case_id"]
            for s in rec["samples"]:
                dx = parse_dx(s["text"])
                if not dx:
                    continue
                cand = dx[0]["diagnosis"]
                key = (cid, norm_text(cand))
                if key not in J1 or silver(cand, gold[cid]) is not None:
                    continue
                count[side][key] += 1
                shown.setdefault(key, cand[:200])
                models[key].add(t)
    rng = random.Random(a.seed)
    picked = []
    for side in ["base", "post"]:
        pool = dict(count[side])
        for _ in range(a.n // 2):
            keys = [k for k in pool if k not in picked]
            k = rng.choices(keys, weights=[pool[x] for x in keys])[0]
            picked.append(k)
    rng.shuffle(picked)
    import os
    os.makedirs(a.out_dir, exist_ok=True)
    with open(f"{a.out_dir}/audit.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "final_diagnosis", "candidate", "verdict (same / different / unsure)", "note"])
        for i, k in enumerate(picked, 1):
            w.writerow([i, gold[k[0]], shown[k], "", ""])
    with open(f"{a.out_dir}/audit_key.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "case_id", "key", "mistral_small", "phi4", "base_count", "post_count", "models"])
        for i, k in enumerate(picked, 1):
            w.writerow([i, k[0], k[1], int(J1[k]), int(J2.get(k, False)), count["base"][k], count["post"][k], " ".join(sorted(models[k]))])
    tot = {s: sum(c.values()) for s, c in count.items()}
    print(json.dumps({"middle_zone_top_answers": tot, "distinct_pairs": {s: len(c) for s, c in count.items()}, "picked": len(picked)}))


if __name__ == "__main__":
    main()
