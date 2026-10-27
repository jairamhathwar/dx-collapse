"""Greedy single-call answers and the must-not-miss prompt (DDXPlus, CPU only).

    python r7_analysis.py --mode greedy --runs RUNS --cases cases_test.jsonl --conditions-json release_conditions.json \
        --pairs "greedy:qwen3-8b-base>qwen3-8b ..." --out greedy.json
    python r7_analysis.py --mode mnm --runs RUNS ... --pairs "MNMDIR/qwen3-8b>DEFAULTDIR/qwen3-8b ..." --out mnm.json

greedy: per model, from one greedy sample per case: top-1 accuracy, recall@3/5 of the list, list TV, conditional TV,
severe named (reference >= 0.1). Pairs: post minus base, condition-clustered bootstrap.
mnm: each sample's JSON list is split by its "must_not_miss" flag into a most-likely part and a must-not-miss part.
Per model: list lengths; recall@5 of the most-likely part; severe named (reference >= 0.1) by the most-likely part and
by the whole list; unwarranted severe conditions named (severe, reference < 0.1) per list; per severe condition, the
share of lists naming it wherever the reference gives it any weight. Pairs compare the must-not-miss run with another run
of the same or another model (whole list vs that run's list), condition-clustered bootstrap.
"""
import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ddx_names import Matcher
from review2 import boot_p
from stated_vs_sampled import stated_dist, tv
from parsing import parse_dx

_D = json.JSONDecoder()


def flagged_array(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    for i in reversed([m.start() for m in re.finditer(r"\[", text)]):
        try:
            arr, _ = _D.raw_decode(text, i)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(arr, list) and arr and all(isinstance(x, dict) and isinstance(x.get("diagnosis"), str) for x in arr):
            return arr
    return None


def to_dist(items, match):
    """renormalized stated distribution over matched names, in list order"""
    pairs = []
    for x in items:
        n = match(x["diagnosis"])
        if n is None:
            continue
        try:
            p = float(x.get("probability"))
        except (TypeError, ValueError):
            p = None
        pairs.append((n, p))
    if not pairs:
        return None
    ps = [p for _, p in pairs]
    if any(p is None or not (0 <= p <= 1) for p in ps) or sum(ps) <= 0:
        ps = [1.0] * len(pairs)
    d = {}
    for (n, _), p in zip(pairs, ps):
        d[n] = d.get(n, 0) + p
    z = sum(d.values())
    return {k: v / z for k, v in d.items()}


def case_metrics(lists, ref, gold, severity, tau, full=None):
    """lists: stated dists (most-likely part); full: optional names of the whole list per sample"""
    sev_tau = [k for k, p in ref.items() if p >= tau and severity.get(k, 9) <= 2]
    r = {}
    order = [sorted(d, key=lambda k: -d[k]) for d in lists]
    r["top1"] = float(np.mean([o[0] == gold for o in order]))
    r["rec3"] = float(np.mean([gold in o[:3] for o in order]))
    r["rec5"] = float(np.mean([gold in o[:5] for o in order]))
    r["tv_list"] = float(np.mean([tv(d, ref) for d in lists]))
    ctv = []
    for d in lists:
        m = sum(ref.get(k, 0) for k in d)
        if m > 0:
            ctv.append(tv(d, {k: ref[k] / m for k in d if k in ref}))
    r["cond_tv"] = float(np.mean(ctv)) if ctv else None
    r["sev_named"] = float(np.mean([np.mean([k in d for k in sev_tau]) for d in lists])) if sev_tau else None
    if full is not None:
        r["sev_named_full"] = float(np.mean([np.mean([k in f for k in sev_tau]) for f in full])) if sev_tau else None
        r["unwarranted_full"] = float(np.mean([sum(1 for k in f if severity.get(k, 9) <= 2 and ref.get(k, 0) < tau) for f in full]))
        r["len_full"] = float(np.mean([len(f) for f in full]))
    r["unwarranted"] = float(np.mean([sum(1 for k in d if severity.get(k, 9) <= 2 and ref.get(k, 0) < tau) for d in lists]))
    r["len"] = float(np.mean([len(d) for d in lists]))
    r["named_sets"] = [set(f) for f in (full if full is not None else lists)]
    return r


def load(path, cases, refs, severity, match, tau, mnm):
    rows = {}
    fh = gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)
    for rec in map(json.loads, fh):
        cid = rec["case_id"]
        lists, full = [], []
        for s in rec["samples"]:
            if mnm:
                arr = flagged_array(s["text"])
                if not arr:
                    continue
                likely = [x for x in arr if not x.get("must_not_miss")]
                d = to_dist(likely, match)
                f = {match(x["diagnosis"]) for x in arr} - {None}
                if d is None:
                    continue
                lists.append(d)
                full.append(f)
            else:
                d, _ = stated_dist(parse_dx(s["text"]), match)
                if d is not None:
                    lists.append(d)
                    full.append(set(d))
        if lists:
            r = case_metrics(lists, refs[cid], cases[cid]["gold"], severity, tau, full)
            r["gold"] = cases[cid]["gold"]
            r["n_valid"] = len(lists)
            r["n"] = len(rec["samples"])
            rows[cid] = r
        else:
            rows[cid] = {"gold": cases[cid]["gold"], "n_valid": 0, "n": len(rec["samples"])}
    return rows


def clustered_diff(A, B, key, cids, rng, n_boot):
    ok = [c for c in cids if A[c].get(key) is not None and B[c].get(key) is not None]
    v = np.array([A[c][key] - B[c][key] for c in ok])
    groups = defaultdict(list)
    for i, c in enumerate(ok):
        groups[A[c]["gold"]].append(i)
    garr = [np.array(g) for g in groups.values()]
    b = np.array([v[np.concatenate([garr[j] for j in rng.choice(len(garr), len(garr))])].mean() for _ in range(n_boot)])
    return {"diff": float(v.mean()), "cluster_ci": [float(x) for x in np.percentile(b, [2.5, 97.5])], "p": boot_p(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["greedy", "mnm"], required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--pairs", required=True, help='greedy: "dir:base>post" or "base_dir/tag>post_dir/tag"; mnm: "mnm_dir/tag>other_dir/tag" (paths under --runs)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    severity = {c["cond-name-eng"]: int(c["severity"]) for c in raw.values()}
    match = Matcher(sorted(severity))
    severe = sorted(k for k, s in severity.items() if s <= 2)
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    refs = {}
    for cid, c in cases.items():
        z = sum(p for _, p in c["gt_differential"])
        refs[cid] = {k: p / z for k, p in c["gt_differential"]}
    rng = np.random.default_rng(0)
    cache, report = {}, {"models": {}, "pairs": {}}

    def get(path, mnm):
        if path not in cache:
            f = Path(a.runs) / path / "samples.jsonl"
            cache[path] = load(f if f.exists() else f.with_suffix(".jsonl.gz"), cases, refs, severity, match, a.tau, mnm)
            rows = cache[path]
            keys = ["top1", "rec3", "rec5", "tv_list", "cond_tv", "sev_named", "sev_named_full", "unwarranted", "unwarranted_full", "len", "len_full"]
            m = {k: float(np.mean([r[k] for r in rows.values() if r.get(k) is not None])) for k in keys
                 if any(r.get(k) is not None for r in rows.values())}
            m["valid_rate"] = float(sum(r["n_valid"] for r in rows.values()) / sum(r["n"] for r in rows.values()))
            per = {}
            for k in severe:
                inc = [np.mean([k in f for f in r["named_sets"]]) for cid, r in rows.items() if r.get("named_sets") and refs[cid].get(k, 0) > 0]
                per[k] = float(np.mean(inc)) if inc else None
            m["severe_inclusion"] = per
            report["models"][path] = m
            print(path, json.dumps({k: round(v, 4) for k, v in m.items() if isinstance(v, float)}), flush=True)
        return cache[path]

    for item in a.pairs.split():
        if a.mode == "greedy":
            if ":" in item:  # same directory
                d, pr = item.split(":")
                bt, pt = pr.split(">")
                bt, pt = f"{d}/{bt}", f"{d}/{pt}"
            else:  # "base_dir/tag>post_dir/tag", e.g. matched-prompt runs in different directories
                bt, pt = item.split(">")
            B, P = get(bt, False), get(pt, False)
            keys = ["top1", "rec3", "rec5", "tv_list", "cond_tv", "sev_named"]
        else:
            m_, o_ = item.split(">")
            P, B = get(m_, True), get(o_, "mnm_" in o_)
            keys = ["rec5", "tv_list", "sev_named", "sev_named_full", "unwarranted_full"]
        cids = [c for c in cases if P[c].get("n_valid") and B[c].get("n_valid")]
        out = {k: clustered_diff(P, B, k, cids, rng, a.n_boot) for k in keys}
        report["pairs"][item] = out
        print(item, json.dumps({k: [round(v["diff"], 4), [round(x, 4) for x in v["cluster_ci"]]] for k, v in out.items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
