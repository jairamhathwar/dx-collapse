"""Check the case-report judges against silver labels from string matching (no human labels).

    python medcase_silver.py --cases cases_test.jsonl --runs RUNS --judge judge_mistral24b.jsonl \
        --judge2 judge2_phi4.jsonl --ddx-judge judge_ddx_mistral24b.jsonl --authors-ddx authors_ddx.jsonl \
        --pairs "qwen3-8b-base>qwen3-8b ..." --out silver.json

Final-diagnosis judges. Each sampled top answer (first item of a valid list) gets a silver label where string
matching is unambiguous:
  positive  its normalized words (stop words dropped) are the same set as the final diagnosis's;
  negative  it shares no content word with the final diagnosis (stop words and the qualifiers in GENERIC dropped).
Sensitivity = share of silver positives the judge accepts; the acceptance rate on silver negatives is an upper bound
on its false-positive rate (synonyms and abbreviations with no shared word, e.g. "TB" for tuberculosis, land among
the negatives). Rates for each judge and for strict grading (both judges), per model, and the post-trained minus
base difference per pair with a 95% case bootstrap interval.

Authors'-alternatives judge. Each name in a valid list is a silver positive for item j if its normalized words match
the authors' item j exactly, and a silver negative if it shares no content word with any item; accuracy = share of
positives mapped to the right item, false-positive rate = share of negatives mapped to any item.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from disease_normalizer import STOP, norm_text
from medcase_analysis import sample_files
from parsing import parse_dx

GENERIC = set("""disease diseases disorder syndrome type acute chronic primary secondary infection infections
infectious inflammatory malignant benign tumor tumour neoplasm lesion of the and or with due to induced related
associated left right bilateral""".split())


def words(s):
    return frozenset(norm_text(s, drop_stop=True).split())


def content(s):
    return frozenset(t for t in norm_text(s).split() if t not in STOP and t not in GENERIC)


def silver(cand, gold):
    if words(cand) and words(cand) == words(gold):
        return 1
    if not (content(cand) & content(gold)):
        return 0
    return None


def boot_diff(per_b, per_p, rng, n_boot):
    """per_x: {case: (yes, n)}; difference in pooled rate post minus base, resampling cases."""
    cs = sorted(set(per_b) | set(per_p))
    yb = np.array([per_b.get(c, (0, 0))[0] for c in cs], float)
    nb = np.array([per_b.get(c, (0, 0))[1] for c in cs], float)
    yp = np.array([per_p.get(c, (0, 0))[0] for c in cs], float)
    np_ = np.array([per_p.get(c, (0, 0))[1] for c in cs], float)
    est = yp.sum() / np_.sum() - yb.sum() / nb.sum()
    idx = rng.integers(0, len(cs), (n_boot, len(cs)))
    bs = yp[idx].sum(1) / np_[idx].sum(1) - yb[idx].sum(1) / nb[idx].sum(1)
    return [float(est), *map(float, np.nanpercentile(bs, [2.5, 97.5]))]


def main():
    ap = argparse.ArgumentParser()
    for x in ["cases", "runs", "judge", "judge2", "ddx-judge", "authors-ddx", "pairs", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    a = ap.parse_args()
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(a.cases))}
    J1 = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge))}
    J2 = {(r["case_id"], r["key"]): r["match"] for r in map(json.loads, open(a.judge2))}
    K = {(r["case_id"], r["key"]): r["item"] for r in map(json.loads, open(a.ddx_judge))}
    DDX = {r["case_id"]: r["ddx"] for r in map(json.loads, open(a.authors_ddx))}
    tags = sorted({t for p in a.pairs.split() for t in p.split(">")})
    rng = np.random.default_rng(0)
    report = {"models": {}, "pairs": {}}
    per_case = {}
    for t in tags:
        # counters per case: judge -> label -> [yes, n]
        cnt = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        tot = {"top": 0, "pos": 0, "neg": 0, "ddx_names": 0, "ddx_pos": 0, "ddx_neg": 0}
        dd = {"pos_right": 0, "neg_any": 0}
        for rec in (json.loads(l) for f in sample_files(f"{a.runs}/{t}") for l in open(f)):
            cid = rec["case_id"]
            alts = DDX.get(cid, [])
            for s in rec["samples"]:
                dx = parse_dx(s["text"])
                if not dx:
                    continue
                cand = dx[0]["diagnosis"]
                key = (cid, norm_text(cand))
                lab = silver(cand, gold[cid])
                tot["top"] += 1
                if lab is not None and key in J1 and key in J2:
                    tot["pos" if lab else "neg"] += 1
                    for jn, v in [("judge1", J1[key]), ("judge2", J2[key]), ("strict", J1[key] and J2[key])]:
                        c = cnt[cid][(jn, lab)]
                        c[0] += int(v)
                        c[1] += 1
                if not alts:
                    continue
                for d in dx[:5]:
                    k2 = (cid, norm_text(d["diagnosis"]))
                    if k2 not in K:
                        continue
                    tot["ddx_names"] += 1
                    w = words(d["diagnosis"])
                    hit = [j + 1 for j, x in enumerate(alts) if j >= 1 and w and words(x) == w]
                    if hit:
                        tot["ddx_pos"] += 1
                        dd["pos_right"] += int(K[k2] in hit)
                    elif all(not (content(d["diagnosis"]) & content(x)) for x in alts):
                        tot["ddx_neg"] += 1
                        dd["neg_any"] += int(K[k2] != 0)
        per_case[t] = cnt
        m = dict(tot)
        for jn in ["judge1", "judge2", "strict"]:
            for lab, nm in [(1, "sens"), (0, "fp_upper")]:
                y = sum(cnt[c][(jn, lab)][0] for c in cnt)
                n = sum(cnt[c][(jn, lab)][1] for c in cnt)
                m[f"{nm}_{jn}"] = y / n if n else None
        m["ddx_acc_pos"] = dd["pos_right"] / tot["ddx_pos"] if tot["ddx_pos"] else None
        m["ddx_fp_upper"] = dd["neg_any"] / tot["ddx_neg"] if tot["ddx_neg"] else None
        m["share_silver"] = (tot["pos"] + tot["neg"]) / tot["top"] if tot["top"] else None
        report["models"][t] = m
        print(t, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
    for p in a.pairs.split():
        b, q = p.split(">")
        pr = {}
        for jn in ["judge1", "judge2", "strict"]:
            for lab, nm in [(1, "sens"), (0, "fp_upper")]:
                f = lambda t: {c: tuple(per_case[t][c][(jn, lab)]) for c in per_case[t] if per_case[t][c][(jn, lab)][1]}
                pr[f"{nm}_{jn}"] = boot_diff(f(b), f(q), rng, a.n_boot)
        report["pairs"][p] = pr
        print(p, json.dumps({k: [round(x, 4) for x in v] for k, v in pr.items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
