"""Review re-analyses on DDXPlus (CPU only): drawn pass@32 against the worked-example base model, and coverage
corrected for chance.

    python r23_reanalysis.py --runs RUNS --overflow OVERFLOW_RUNS --cases cases_test.jsonl \
        --conditions-json release_conditions.json --out r23.json

(1) Drawn pass@32: each valid zero-shot sample's top answer of the post-trained model replaced by a draw from its own
    stated list (as r16_breadth.py: same seed and order, invalid samples count as wrong), against the base model's
    sampled pass@32 with worked examples (fewshot_t07; its best prompt in every pair), raw and chance-corrected (minus
    the pass@32 reached against the true pathology of another case with a different pathology; the 20 permutations
    of r18_chance.py).
(2) Coverage corrected for chance, zero-shot: Cov@32 (reference mass of the distinct sampled top answers) and the
    reference mass one list names (mean over valid lists), each minus the same quantity computed against another
    patient's reference (a case with a different pathology; 20 permutations). A patient-blind guesser scores about
    zero on the corrected measures, so they reward only patient-specific coverage.
Post-trained minus base with condition-clustered intervals (r14_an.Boot, 10,000 resamples).
"""
import argparse
import json
import math

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from r14_an import Boot
from stated_vs_sampled import stated_dist

PAIRS = [("Qwen3-8B", "pair_q3", "qwen3-8b-base", "qwen3-8b", "fewshot_t07"),
         ("Qwen2.5-7B", "pair25_t07", "qwen25-7b-base", "qwen25-7b-it", "fewshot_t07"),
         ("Qwen2.5-32B", "big32_ddx", "qwen25-32b-base", "qwen25-32b-it", "fewshot_t07"),
         ("OLMo 3 7B", "ddx_olmo", "olmo3-base", "olmo3-instruct", "fewshot_t07"),
         ("OLMo 3.1 32B", "big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct", "fewshot_t07"),
         ("Qwen2.5-72B", "big72_ddx", "qwen25-72b-base", "qwen25-72b-it", "fewshot72_t07")]


def p_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def read(path, match):
    out = {}
    for rec in map(json.loads, open(path)):
        out[rec["case_id"]] = [stated_dist(parse_dx(s["text"]), match) for s in rec["samples"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "overflow", "cases", "conditions-json", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    cases = [json.loads(l) for l in open(a.cases)]
    gold = {c["case_id"]: c["gold"] for c in cases}
    refs = {}
    for c in cases:
        z = sum(p for _, p in c["gt_differential"])
        refs[c["case_id"]] = {k: p / z for k, p in c["gt_differential"]}
    cl = list(gold)
    rng = np.random.default_rng(0)
    plab = []  # shuffled true pathologies, constructed exactly as in r18_chance.py (same seed and order)
    for _ in range(20):
        lab = {c: gold[c2] for c, c2 in zip(cl, rng.permutation(cl))}
        for c in cl:
            while lab[c] == gold[c]:
                lab[c] = gold[cl[rng.integers(len(cl))]]
        plab.append(lab)
    boot = Boot(cl, gold, a.n_boot, rng)
    crng = np.random.default_rng(2)
    perms = []  # another case with a different pathology, for the coverage correction
    for _ in range(20):
        lab = {c: c2 for c, c2 in zip(cl, crng.permutation(cl))}
        for c in cl:
            while gold[lab[c]] == gold[c]:
                lab[c] = cl[crng.integers(len(cl))]
        perms.append(lab)

    def diff(P, B):
        cs = [c for c in cl if P.get(c) is not None and B.get(c) is not None]
        r = boot.ratio({c: P[c] - B[c] for c in cs}, {c: 1.0 for c in cs})
        r["n"] = len(cs)
        return r

    report = {"drawn_vs_fewshot_base": {}, "coverage_corrected": {}, "models": {}}
    for nm, d, b, p, fsd in PAIRS:
        Pz = read(f"{a.runs}/{d}/{p}/samples.jsonl", match)
        Bz = read(f"{a.runs}/{d}/{b}/samples.jsonl", match)
        Bf = read(f"{a.overflow}/{fsd}/{b}/samples.jsonl", match)
        # (1) drawn pass@32 of the post-trained model (r16 order and seed) vs the worked-example base model
        drng = np.random.default_rng(1)
        draw_raw, draw_sh, base_raw, base_sh = {}, {}, {}, {}
        for c, v in Pz.items():
            valid = [(dd, t) for dd, t in v if dd is not None and t]
            if not valid:
                continue
            draws = [drng.choice(list(dd), p=np.array(list(dd.values())) / sum(dd.values())) for dd, _ in valid]
            n = len(v)
            draw_raw[c] = p_at_k(n, sum(x == gold[c] for x in draws), min(32, n))
            draw_sh[c] = float(np.mean([p_at_k(n, sum(x == lab[c] for x in draws), min(32, n)) for lab in plab]))
        for c, v in Bf.items():
            t = [x for _, x in v]
            n = len(t)
            if not n:
                continue
            base_raw[c] = p_at_k(n, sum(x == gold[c] for x in t), min(32, n))
            base_sh[c] = float(np.mean([p_at_k(n, sum(x == lab[c] for x in t), min(32, n)) for lab in plab]))
        cor_d = {c: draw_raw[c] - draw_sh[c] for c in draw_raw}
        cor_b = {c: base_raw[c] - base_sh[c] for c in base_raw}
        report["models"][f"{nm}/drawn_post"] = {"pass32": float(np.mean(list(draw_raw.values()))), "shuffled": float(np.mean(list(draw_sh.values())))}
        report["models"][f"{nm}/fewshot_base"] = {"pass32": float(np.mean(list(base_raw.values()))), "shuffled": float(np.mean(list(base_sh.values())))}
        report["drawn_vs_fewshot_base"][nm] = {"raw": diff(draw_raw, base_raw), "corrected": diff(cor_d, cor_b)}
        # (2) coverage corrected for chance, zero-shot
        cov = {}
        for t, R in [(b, Bz), (p, Pz)]:
            sc, sc_c, lm, lm_c = {}, {}, {}, {}
            for c, v in R.items():
                valid = [(dd, x) for dd, x in v if dd is not None and x]
                if not valid:
                    continue
                S = {x for _, x in valid}
                sc[c] = sum(refs[c].get(x, 0) for x in S)
                sc_c[c] = sc[c] - float(np.mean([sum(refs[lab[c]].get(x, 0) for x in S) for lab in perms]))
                ms = [sum(refs[c].get(k, 0) for k in dd) for dd, _ in valid]
                lm[c] = float(np.mean(ms))
                lm_c[c] = lm[c] - float(np.mean([np.mean([sum(refs[lab[c]].get(k, 0) for k in dd) for dd, _ in valid]) for lab in perms]))
            cov[t] = {"cov32": sc, "cov32_corr": sc_c, "list_mass": lm, "list_mass_corr": lm_c}
            report["models"][f"{nm}/{t}"] = {k: float(np.mean(list(v.values()))) for k, v in cov[t].items()}
        report["coverage_corrected"][nm] = {k: diff(cov[p][k], cov[b][k]) for k in ["cov32", "cov32_corr", "list_mass", "list_mass_corr"]}
        print(nm, json.dumps({k: round(v, 3) for k, v in report["models"][f"{nm}/drawn_post"].items()}),
              json.dumps({k: round(v, 3) for k, v in report["models"][f"{nm}/fewshot_base"].items()}),
              json.dumps({k: [round(x["diff"], 3), [round(y, 3) for y in x["cluster_ci"]]] for k, x in report["drawn_vs_fewshot_base"][nm].items()}),
              json.dumps({k: [round(x["diff"], 3), [round(y, 3) for y in x["cluster_ci"]]] for k, x in report["coverage_corrected"][nm].items()}), flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
