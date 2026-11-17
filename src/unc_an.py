"""Error-detection baselines on DDXPlus samples: which signal flags a wrong single-call answer?

    python unc_an.py RUNS DATA_DIR out.json label:dir/tag [label:dir/tag ...]

Single-call answer A = top answer of the first valid sample (as in minp_an.py). Signals (higher = more confident):
  agree        share of the other valid samples whose top answer is A (self-consistency)
  neg_entropy  minus the entropy of the sampled top-answer distribution (semantic entropy on a closed answer set,
               where meaning classes are the 49 conditions)
  stated_p     probability the first sample states for A (verbalized confidence)
  stated_negH  minus the entropy of that sample's stated list
  stated_mean  mean stated probability of A over the samples that list it (verbalized confidence, averaged)
  combo        mean of the ranks of agree and stated_p (a label-free combination); combo_mean: of agree and stated_mean
Per signal: AUROC for A being correct, and caught@20: share of wrong answers among the 20% least confident cases
(ties broken by case order). Also acc (accuracy of A) and silent (wrong A repeated by every other valid sample).
No intervals.
"""
import glob
import json
import math
import os
import sys
from collections import Counter

from ddx_names import Matcher
from minp_an import auroc
from parsing import parse_dx
from stated_vs_sampled import stated_dist

SIGNALS = ["agree", "neg_entropy", "stated_p", "stated_negH", "stated_mean", "combo"]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r, i = [0.0] * len(xs), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def caught(score, ok, frac=0.2):
    n = len(score)
    low = sorted(range(n), key=lambda i: score[i])[:max(1, int(round(frac * n)))]
    wrong = sum(not o for o in ok)
    return sum(not ok[i] for i in low) / wrong if wrong else None


def main():
    R, D, out = sys.argv[1], sys.argv[2], sys.argv[3]
    raw = json.load(open(f"{D}/release_conditions.json"))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(f"{D}/cases_test.jsonl"))}
    report = {}
    for spec in sys.argv[4:]:
        label, path = spec.split(":", 1)
        files = [f"{R}/{path}/samples.jsonl"] if os.path.exists(f"{R}/{path}/samples.jsonl") else sorted(glob.glob(f"{R}/{path}/samples.s*.jsonl"))
        sig = {s: [] for s in SIGNALS[:-1]}
        ok, n_silent = [], 0
        for rec in (json.loads(l) for f in files for l in open(f)):
            g = gold[rec["case_id"]]
            parsed = [stated_dist(parse_dx(s["text"]), match) for s in rec["samples"]]
            valid = [(d, t) for d, t in parsed if t]
            if not valid:
                continue
            d0, a = valid[0]
            tops = [t for _, t in valid]
            others = tops[1:]
            agree = sum(t == a for t in others) / len(others) if others else 1.0
            cnt = Counter(tops)
            ent = -sum(v / len(tops) * math.log(v / len(tops)) for v in cnt.values())
            listed = [d[a] for d, _ in valid if a in d]
            sig["agree"].append(agree)
            sig["neg_entropy"].append(-ent)
            sig["stated_p"].append(d0.get(a, 0.0))
            sig["stated_negH"].append(sum(v * math.log(v) for v in d0.values() if v > 0))
            sig["stated_mean"].append(sum(listed) / len(listed) if listed else 0.0)
            ok.append(a == g)
            n_silent += a != g and bool(others) and all(t == a for t in others)
        if not ok:
            print(label, "no samples found; skipped", flush=True)
            continue
        ra, rs = ranks(sig["agree"]), ranks(sig["stated_p"])
        sig["combo"] = [(x + y) / 2 for x, y in zip(ra, rs)]
        rm = ranks(sig["stated_mean"])
        sig["combo_mean"] = [(x + y) / 2 for x, y in zip(ra, rm)]
        r = {"n": len(ok), "acc": sum(ok) / len(ok), "silent": n_silent / max(1, sum(not o for o in ok))}
        for s in SIGNALS + ["combo_mean"]:
            r[f"auroc_{s}"] = auroc(sig[s], ok)
            r[f"caught20_{s}"] = caught(sig[s], ok)
        report[label] = r
        print(label, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
