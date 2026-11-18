"""Behavioural agreement between models on short-answer QA: do two models give the same first answer?

    python agree_an.py CASES.jsonl out.json name=SAMPLES.jsonl [...] --pairs a:b [a:b ...]

Each SAMPLES file holds one greedy answer list per question (sample_diagnoses.py --task ambigqa --k 1 --temperature 0).
Per model: accuracy of the first answer (lenient grading, qa_norm) and the share of questions with a parsable answer.
Per pair: share of questions on which the two first answers match (same acceptable answer, or the same text under
qa_norm's lenient rules); questions where either model has no answer count as disagreement.
"""
import json
import sys

from parsing import parse_dx
from qa_norm import Grader


def main():
    i = sys.argv.index("--pairs")
    cases = {c["case_id"]: c for c in map(json.loads, open(sys.argv[1]))}
    files = dict(x.split("=", 1) for x in sys.argv[3:i])
    first = {}
    report = {"models": {}, "pairs": {}}
    for name, path in files.items():
        first[name] = {}
        for rec in map(json.loads, open(path)):
            dx = parse_dx(rec["samples"][0]["text"]) if rec["samples"] else []
            first[name][rec["case_id"]] = dx[0]["diagnosis"] if dx else None
        g = {cid: Grader(cases[cid]) for cid in first[name]}
        ok = sum(a is not None and g[cid].correct(g[cid].key(a)) for cid, a in first[name].items())
        report["models"][name] = {"n": len(first[name]), "acc": ok / max(1, len(first[name])),
                                  "valid": sum(a is not None for a in first[name].values()) / max(1, len(first[name]))}
        print(name, json.dumps({k: round(v, 3) for k, v in report["models"][name].items()}), flush=True)
    for p in sys.argv[i + 1:]:
        a, b = p.split(":")
        ids = [c for c in first[a] if c in first[b]]
        same = 0
        for cid in ids:
            x, y = first[a][cid], first[b][cid]
            if x is None or y is None:
                continue
            ks = Grader(cases[cid]).keys([x, y])
            same += ks[0] == ks[1]
        report["pairs"][p] = same / max(1, len(ids))
        print(f"agree({a}, {b}) = {report['pairs'][p]:.3f}", flush=True)
    json.dump(report, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
