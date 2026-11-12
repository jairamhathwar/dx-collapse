"""Self-targets for the self_witness GRPO reward: the starting model's own mean stated list per training case.

    python make_self_targets.py SAMPLES.jsonl conditions.json out.json

SAMPLES.jsonl: sample_diagnoses.py output of the starting model on the training cases (a few samples per case).
Each valid sample's stated list is renormalized over the conditions it names (stated_vs_sampled.stated_dist),
and the case's target is the mean over valid samples. Cases with no valid sample get no target (reward 0).
"""
import json
import sys
from collections import Counter

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist


def main():
    match = Matcher(json.load(open(sys.argv[2])))
    out, n_none = {}, 0
    for rec in map(json.loads, open(sys.argv[1])):
        dists = [d for d, _ in (stated_dist(parse_dx(s["text"]), match) for s in rec["samples"]) if d]
        if not dists:
            n_none += 1
            continue
        m = Counter()
        for d in dists:
            for c, v in d.items():
                m[c] += v / len(dists)
        out[rec["case_id"]] = dict(m)
    json.dump(out, open(sys.argv[3], "w"))
    print(f"targets for {len(out)} cases; {n_none} without a valid list")


if __name__ == "__main__":
    main()
