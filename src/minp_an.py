"""Decoding baseline on DDXPlus: does min-p sampling at a higher temperature restore the sampled differential?

    python minp_an.py RUNS DATA_DIR out.json label:dir/tag [label:dir/tag ...]

Per run (980 cases x 32 samples; top answers and lists as in every table, stated_vs_sampled.stated_dist; invalid
samples count as wrong):
  valid          share of samples whose top answer is one of the 49 conditions
  distinct       distinct valid top answers per case
  pass1, pass32  unbiased pass@1 and pass@32 of the top answers
  list1          the first valid sample's stated list contains the true pathology
  list_minus_pass32  list1 - pass32 (the readout reversal)
  vote_gain      majority top answer over valid samples minus the single-call answer (first valid sample)
  auroc_agree    AUROC of agreement (share of the other valid samples with the same top answer) for the single-call
                 answer being correct; auroc_stated: the same for its stated probability (average ranks for ties)
  silent         share of wrong single-call answers that every other valid sample repeats
No intervals: a smoke-scale check, not a headline analysis.
"""
import glob
import json
import os
import sys
from collections import Counter

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist


def auroc(score, label):
    pairs = sorted(zip(score, label))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(1 for _, l in pairs if l)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    return (sum(r for r, (_, l) in zip(ranks, pairs) if l) - pos * (pos + 1) / 2) / (pos * neg)


def main():
    R, D, out = sys.argv[1], sys.argv[2], sys.argv[3]
    raw = json.load(open(f"{D}/release_conditions.json"))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(f"{D}/cases_test.jsonl"))}
    report = {}
    for spec in sys.argv[4:]:
        label, path = spec.split(":", 1)
        m = Counter()
        n_cases = n_single = n_wrong = 0
        conf_a, conf_s, corr = [], [], []
        files = [f"{R}/{path}/samples.jsonl"] if os.path.exists(f"{R}/{path}/samples.jsonl") else sorted(glob.glob(f"{R}/{path}/samples.s*.jsonl"))
        for rec in (json.loads(l) for f in files for l in open(f)):
            g = gold[rec["case_id"]]
            parsed = []
            for s in rec["samples"]:
                dist, top = stated_dist(parse_dx(s["text"]), match)
                parsed.append((dist, top))
            k = len(parsed)
            tops = [t for _, t in parsed if t]
            c = sum(t == g for t in tops)
            n_cases += 1
            m["valid"] += len(tops) / k
            m["distinct"] += len(set(tops))
            m["pass1"] += c / k
            m["pass32"] += float(c > 0)
            first = next(((d, t) for d, t in parsed if t), None)
            if first is None:
                continue
            d0, a = first
            n_single += 1
            m["list1"] += float(g in d0)
            others = list(tops)
            others.remove(a)
            m["vote_gain"] += float(Counter(tops).most_common(1)[0][0] == g) - float(a == g)
            agree = sum(t == a for t in others) / len(others) if others else 1.0
            conf_a.append(agree)
            conf_s.append(d0.get(a, 0.0))
            corr.append(a == g)
            if a != g:
                n_wrong += 1
                m["silent"] += float(bool(others) and all(t == a for t in others))
        r = {x: m[x] / n_cases for x in ("valid", "distinct", "pass1", "pass32")}
        r.update({x: m[x] / n_single for x in ("list1", "vote_gain")})
        r["list_minus_pass32"] = r["list1"] - r["pass32"]
        r["silent"] = m["silent"] / n_wrong if n_wrong else None
        r["auroc_agree"], r["auroc_stated"] = auroc(conf_a, corr), auroc(conf_s, corr)
        report[label] = r
        print(label, json.dumps({x: round(v, 3) if isinstance(v, float) else v for x, v in r.items()}), flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
