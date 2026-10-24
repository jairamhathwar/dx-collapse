"""Consequences of the stated/sampled gap on DDXPlus: aggregation (self-consistency vs stated lists)
and uncertainty (sampling-based vs stated signals).

    python consequences.py --runs-dir RUNS/pair_q3 --tags qwen3-8b-base,qwen3-8b \
        --cases cases_test.jsonl --conditions-json release_conditions.json [--out out.json]

Valid samples only (a listed name maps to one of the 49 conditions). Per case:
  Answers and accuracy (exact = true pathology):
    single         top-1 of the first valid sample (one call)
    vote@5/@32     majority top-1 over the first 5 / all valid samples (self-consistency)
    stated@5/@32   argmax of the averaged stated lists over the first 5 / all valid samples
    top3: list1    true pathology among the top 3 of the first sample's stated list
          vote@32  ... among the 3 most frequent sampled top-1 answers
          stated@32 ... among the top 3 of the averaged stated lists
  Uncertainty about the single-call answer A (top-1 of the first valid sample), AUROC for A correct:
    stated_p       stated probability of A in the same list (no extra calls)
    stated_negH    minus the entropy of that list
    consist@5      share of the next 4 valid samples whose top-1 is A (4 extra calls)
    consist@32     share of the other valid samples whose top-1 is A (up to 31 extra calls)
    samp_negH@32   minus the entropy of the sampled top-1 distribution
  Ambiguity: Spearman correlation between each uncertainty signal (negated confidence) and the
  entropy of the case's reference differential (does the model's uncertainty track true ambiguity?).
Cases with no valid sample count as wrong for accuracy and are dropped from AUROC/Spearman.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx, sort_by_prob
from stated_vs_sampled import entropy, stated_dist

ACC = ["single", "vote@5", "vote@32", "stated@5", "stated@32", "top3_list1", "top3_vote@32", "top3_stated@32"]
SIG = ["stated_p", "stated_negH", "consist@5", "consist@32", "samp_negH@32"]


def rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    xs = x[order]  # average ties
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def auroc(score, y):
    y = y.astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rank(score)
    return float((r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def argmax_dist(d):
    return max(d.items(), key=lambda x: x[1])[0] if d else None


def mean_dist(ds):
    m = Counter()
    for d in ds:
        for c, v in d.items():
            m[c] += v / len(ds)
    return m


def vote(tops, lists):
    """majority top-1; ties broken by summed stated probability"""
    cnt = Counter(tops)
    best = max(cnt.values())
    tied = [c for c, v in cnt.items() if v == best]
    if len(tied) == 1:
        return tied[0]
    m = mean_dist(lists)
    return max(tied, key=lambda c: m.get(c, 0))


def case_row(tops, lists, gold, ref_h):
    n = len(tops)
    row = {"ref_h": ref_h, "valid": n}
    if n == 0:
        row.update({k: 0.0 for k in ACC})
        row.update({k: None for k in SIG})
        return row
    m5, m32 = mean_dist(lists[:5]), mean_dist(lists)
    freq = [c for c, _ in Counter(tops).most_common(3)]
    l1 = sorted(lists[0].items(), key=lambda x: -x[1])[:3]
    row.update({
        "single": float(tops[0] == gold),
        "vote@5": float(vote(tops[:5], lists[:5]) == gold),
        "vote@32": float(vote(tops, lists) == gold),
        "stated@5": float(argmax_dist(m5) == gold),
        "stated@32": float(argmax_dist(m32) == gold),
        "top3_list1": float(gold in {c for c, _ in l1}),
        "top3_vote@32": float(gold in freq),
        "top3_stated@32": float(gold in {c for c, _ in sorted(m32.items(), key=lambda x: -x[1])[:3]}),
    })
    A = tops[0]
    others = tops[1:]
    emp = Counter(tops)
    emp = {c: v / n for c, v in emp.items()}
    row.update({
        "A_correct": float(A == gold),
        "stated_p": lists[0].get(A, 0.0),
        "stated_negH": -entropy(lists[0]),
        "consist@5": float(np.mean([t == A for t in others[:4]])) if len(others) >= 1 else None,
        "consist@32": float(np.mean([t == A for t in others])) if others else None,
        "samp_negH@32": -entropy(emp),
    })
    return row


def summarize(rows, rng, n_boot=1000):
    out = {"n_cases": len(rows)}
    for k in ACC:
        out[k] = float(np.mean([r[k] for r in rows]))
    sub = [r for r in rows if r.get("A_correct") is not None and all(r[s] is not None for s in SIG)]
    y = np.array([r["A_correct"] for r in sub])
    H = np.array([r["ref_h"] for r in sub])
    out["n_uncert"] = len(sub)
    out["single_acc_uncert_subset"] = float(y.mean()) if len(y) else None
    S = {s: np.array([r[s] for r in sub], float) for s in SIG}
    for s in SIG:
        out[f"auroc_{s}"] = auroc(S[s], y)
        out[f"rho_{s}"] = spearman(-S[s], H)  # uncertainty vs reference entropy
    # paired bootstrap: stated_p vs consist@5 and consist@32
    idx_all = np.arange(len(sub))
    for other in ["consist@5", "consist@32"]:
        da, dr = [], []
        for _ in range(n_boot):
            i = rng.choice(idx_all, len(idx_all))
            da.append(auroc(S["stated_p"][i], y[i]) - auroc(S[other][i], y[i]))
            dr.append(spearman(-S["stated_p"][i], H[i]) - spearman(-S[other][i], H[i]))
        out[f"d_auroc_stated_p-{other}"] = [out["auroc_stated_p"] - out[f"auroc_{other}"], *map(float, np.nanpercentile(da, [2.5, 97.5]))]
        out[f"d_rho_stated_p-{other}"] = [out["rho_stated_p"] - out[f"rho_{other}"], *map(float, np.nanpercentile(dr, [2.5, 97.5]))]
    # aggregation differences
    for a_, b_ in [("vote@32", "single"), ("stated@32", "vote@32"), ("stated@5", "vote@5"), ("top3_list1", "top3_vote@32")]:
        d = np.array([r[a_] - r[b_] for r in rows])
        boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
        out[f"d_{a_}-{b_}"] = [float(d.mean()), *map(float, np.percentile(boots, [2.5, 97.5]))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--out")
    ap.add_argument("--top-by-prob", action="store_true", help="top-1 = highest stated probability (full-distribution prompt)")
    a = ap.parse_args()

    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    rng = np.random.default_rng(0)
    report = {"runs_dir": a.runs_dir, "models": {}}
    for t in a.tags.split(","):
        rows = []
        for rec in map(json.loads, open(Path(a.runs_dir) / t / "samples.jsonl")):
            case = cases[rec["case_id"]]
            z = sum(p for _, p in case["gt_differential"])
            ref = {c: p / z for c, p in case["gt_differential"]}
            tops, lists = [], []
            for s in rec["samples"]:
                d, top = stated_dist((sort_by_prob if a.top_by_prob else list)(parse_dx(s["text"])), match)
                if d is not None and top is not None:
                    tops.append(top)
                    lists.append(d)
            rows.append(case_row(tops, lists, case["gold"], entropy(ref)))
        report["models"][t] = summarize(rows, rng)

    print(f"{a.runs_dir}")
    print("  accuracy:      " + "".join(f"{k:>15s}" for k in ACC))
    for t, v in report["models"].items():
        print(f"  {t:14s}" + "".join(f"{v[k]:15.3f}" for k in ACC))
    print("  AUROC (single-call answer correct) | Spearman(uncertainty, reference entropy)")
    print("                " + "".join(f"{s:>14s}" for s in SIG) + "   |" + "".join(f"{s:>14s}" for s in SIG) + "   n  acc")
    for t, v in report["models"].items():
        print(f"  {t:14s}" + "".join(f"{v['auroc_' + s]:14.3f}" for s in SIG) + "   |"
              + "".join(f"{v['rho_' + s]:14.3f}" for s in SIG) + f"  {v['n_uncert']}  {v['single_acc_uncert_subset']:.3f}")
    for t, v in report["models"].items():
        print(f"  {t}: " + "  ".join(f"{k[2:]}={x[0]:+.3f}[{x[1]:+.3f},{x[2]:+.3f}]" for k, x in v.items() if k.startswith("d_")))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
