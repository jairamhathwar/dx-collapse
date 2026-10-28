"""Candidate lists for the closed-set case-report prompt.

    python make_medcase_closed.py cases_test.jsonl authors_ddx_clean.jsonl candidates.jsonl [--size 10 --seed 0]

Per case: the final diagnosis, the authors' alternatives kept by clean_authors_ddx.py (at most size - 1), and
distractors drawn at random from other cases' final diagnoses (sharing no content word with any listed candidate)
up to `size` candidates, in random order. Each line also records the role of every candidate.
"""
import argparse
import json
import random

from medcase_silver import content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases"); ap.add_argument("ddx"); ap.add_argument("out")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cases = [json.loads(l) for l in open(a.cases)]
    ddx = {r["case_id"]: r for r in map(json.loads, open(a.ddx))}
    golds = [c["gold"] for c in cases]
    rng = random.Random(a.seed)
    with open(a.out, "w") as fh:
        for c in cases:
            r = ddx.get(c["case_id"])
            alts = [r["ddx"][j - 1] for j in r["keep"]][:a.size - 1] if r else []
            items = [(c["gold"], "gold")] + [(x, "alt") for x in alts]
            words = set().union(*(content(x) for x, _ in items))
            while len(items) < a.size:
                g = rng.choice(golds)
                if g != c["gold"] and not (content(g) & words):
                    items.append((g, "distractor"))
                    words |= content(g)
            rng.shuffle(items)
            fh.write(json.dumps({"case_id": c["case_id"], "candidates": [x for x, _ in items],
                                 "roles": [r_ for _, r_ in items]}) + "\n")


if __name__ == "__main__":
    main()
