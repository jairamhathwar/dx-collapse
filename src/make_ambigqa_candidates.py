"""Candidate lists for the closed AmbigQA prompt (sample_diagnoses.py --task ambigqa --candidates).

    python make_ambigqa_candidates.py CASES.jsonl OUT.jsonl SAMPLES.jsonl [SAMPLES.jsonl ...] [--n 10] [--seed 0]

Per question: one alias for each acceptable answer (at most 3), then the wrong answers that models gave most often in
the SAMPLES files (top answers, merged by qa_norm's lenient rules), up to n candidates in total, shuffled. Questions
with fewer than 4 candidates are dropped. Plausible distractors come from models' own errors, so the list is not
trivially separable; it is the QA counterpart of DDXPlus's 49-condition list.
"""
import argparse
import json
import random
from collections import Counter

from parsing import parse_dx
from qa_norm import Grader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases")
    ap.add_argument("out")
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-from", help="SAMPLES.jsonl of one model: its 3 most frequent wrong answers per question "
                    "are kept off the list (so the list cannot simply confirm that model's own error)")
    ap.add_argument("--random-distractors", action="store_true",
                    help="distractors are acceptable answers of other, randomly chosen questions instead of model errors")
    ap.add_argument("--case-ids", help="only these questions (one id per line)")
    a = ap.parse_args()
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    all_cases = list(cases.values())
    if a.case_ids:
        keep = set(open(a.case_ids).read().split())
        cases = {k: v for k, v in cases.items() if k in keep}
    banned = {cid: set() for cid in cases}
    if a.exclude_from:
        for rec in map(json.loads, open(a.exclude_from)):
            c = cases.get(rec["case_id"])
            if c is None:
                continue
            g = Grader(c)
            raw = [dx[0]["diagnosis"] for dx in (parse_dx(s["text"]) for s in rec["samples"]) if dx]
            own = Counter(k for k in g.keys(raw) if k and not g.correct(k))
            banned[rec["case_id"]] = {k for k, _ in own.most_common(3)}
    wrong = {cid: Counter() for cid in cases}
    for path in a.samples:
        for rec in map(json.loads, open(path)):
            c = cases.get(rec["case_id"])
            if c is None:
                continue
            g = Grader(c)
            raw = [dx[0]["diagnosis"] for dx in (parse_dx(s["text"]) for s in rec["samples"]) if dx]
            for ans, key in zip(raw, g.keys(raw)):
                if key and not g.correct(key):
                    wrong[rec["case_id"]][(key, ans)] += 1
    rng, n_out = random.Random(a.seed), 0
    with open(a.out, "w") as fh:
        for cid, c in cases.items():
            cands = [al[0] for al in c["answers"][:3]]
            by_key = Counter()
            first = {}
            for (key, ans), v in wrong[cid].items():
                by_key[key] += v
                first.setdefault(key, ans)
            if a.random_distractors:
                g = Grader(c)
                pool = rng.sample(all_cases, 4 * a.n)
                for o in pool:
                    x = o["answers"][0][0]
                    if len(cands) < a.n and o["case_id"] != cid and len(x) <= 60 and not g.correct(g.key(x)) and x not in cands:
                        cands.append(x)
            else:
                for key, _ in by_key.most_common():
                    if len(cands) >= a.n:
                        break
                    if len(first[key]) <= 60 and key not in banned[cid]:
                        cands.append(first[key])
            if len(cands) < 4:
                continue
            rng.shuffle(cands)
            fh.write(json.dumps({"case_id": cid, "candidates": cands}) + "\n")
            n_out += 1
    print(f"{n_out} of {len(cases)} questions with candidate lists")


if __name__ == "__main__":
    main()
