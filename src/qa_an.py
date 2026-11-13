"""Stated and sampled readouts on short-answer QA (AmbigQA), mirroring minp_an.py for DDXPlus.

    python qa_an.py CASES.jsonl out.json label:SAMPLES.jsonl [label:SAMPLES.jsonl ...]

Per run (k samples per question; answers keyed by qa_norm.Grader; a sample is valid if its list parses and its first
answer is non-empty; invalid samples count as wrong):
  valid, distinct, pass1, pass32   as in minp_an.py, over the answer keys
  list1          the first valid sample's stated list contains an acceptable answer
  vote_gain      majority answer over valid samples minus the single-call answer (first valid sample)
  auroc_agree    AUROC of agreement (share of the other valid samples with the same key) for the single-call answer
                 being correct; auroc_stated: the same for its stated probability (missing probability = 0)
  silent         share of wrong single-call answers that every other valid sample repeats
Reported for all questions and for the single- and multi-answer subsets. No intervals.
"""
import json
import sys
from collections import Counter

from minp_an import auroc
from parsing import parse_dx
from qa_norm import Grader


def score(cases, path):
    acc = {s: {"m": Counter(), "n": 0, "ns": 0, "nw": 0, "ca": [], "cs": [], "ok": []} for s in ("all", "single", "multi")}
    for rec in map(json.loads, open(path)):
        c = cases.get(rec["case_id"])
        if c is None:
            continue
        g = Grader(c)
        parsed = []
        for s in rec["samples"]:
            dx = parse_dx(s["text"])
            keys = [g.key(d["diagnosis"]) for d in dx]
            top = keys[0] if keys else None
            p = dx[0]["probability"] if dx else None
            parsed.append((top, keys, p if p is not None and 0 <= p <= 1 else 0.0))
        k = len(parsed)
        tops = [t for t, _, _ in parsed if t]
        nc = sum(g.correct(t) for t in tops)
        first = next(((t, ks, p) for t, ks, p in parsed if t), None)
        for sub in ("all", c["type"]):
            A = acc[sub]
            m = A["m"]
            A["n"] += 1
            m["valid"] += len(tops) / k
            m["distinct"] += len(set(tops))
            m["pass1"] += nc / k
            m["pass32"] += float(nc > 0)
            if first is None:
                continue
            a, ks, p = first
            A["ns"] += 1
            m["list1"] += float(any(g.correct(x) for x in ks))
            others = list(tops)
            others.remove(a)
            m["vote_gain"] += float(g.correct(Counter(tops).most_common(1)[0][0])) - float(g.correct(a))
            A["ca"].append(sum(t == a for t in others) / len(others) if others else 1.0)
            A["cs"].append(p)
            A["ok"].append(g.correct(a))
            if not g.correct(a):
                A["nw"] += 1
                m["silent"] += float(bool(others) and all(t == a for t in others))
    out = {}
    for sub, A in acc.items():
        if not A["n"]:
            continue
        m = A["m"]
        r = {x: m[x] / A["n"] for x in ("valid", "distinct", "pass1", "pass32")}
        r.update({x: m[x] / A["ns"] for x in ("list1", "vote_gain")} if A["ns"] else {})
        r["silent"] = m["silent"] / A["nw"] if A["nw"] else None
        r["auroc_agree"], r["auroc_stated"] = auroc(A["ca"], A["ok"]), auroc(A["cs"], A["ok"])
        r["n"] = A["n"]
        out[sub] = r
    return out


def main():
    cases = {c["case_id"]: c for c in map(json.loads, open(sys.argv[1]))}
    report = {}
    for spec in sys.argv[3:]:
        label, path = spec.split(":", 1)
        report[label] = score(cases, path)
        for sub, r in report[label].items():
            print(label, sub, json.dumps({x: round(v, 3) if isinstance(v, float) else v for x, v in r.items()}), flush=True)
    json.dump(report, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
