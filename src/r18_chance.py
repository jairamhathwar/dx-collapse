"""Chance-corrected pass@32 with the headline loader's validity rule (DDXPlus, CPU only).

    python r18_chance.py --runs RUNS --overflow OVERFLOW_RUNS --cases cases_test.jsonl \
        --conditions-json release_conditions.json --out r18.json

A sample's top answer is its first listed item mapped to one of the 49 conditions (stated_vs_sampled.stated_dist, as in
review2.py and every main-text table); invalid samples count as wrong. Per case: real pass@32, and shuffled pass@32
(scored against the true pathology of another case with a different pathology; mean of 20 permutations shared by
all models); corrected = real - shuffled. Settings: zero-shot (headline runs), worked examples with reference lists
(fewshot_t07) and with another model's lists (fewshot_rf_t07); 'best' compares each model at the setting where its
own mean pass@32 is highest. Post-trained minus base with condition-clustered intervals (r14_an.Boot, 10,000).
"""
import argparse
import json
import math

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from r14_an import Boot
from stated_vs_sampled import stated_dist

PAIRS = [("Qwen3-8B", "pair_q3", "qwen3-8b-base", "qwen3-8b"), ("Qwen2.5-7B", "pair25_t07", "qwen25-7b-base", "qwen25-7b-it"),
         ("Qwen2.5-32B", "big32_ddx", "qwen25-32b-base", "qwen25-32b-it"), ("OLMo 3 7B", "ddx_olmo", "olmo3-base", "olmo3-instruct"),
         ("OLMo 3.1 32B", "big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct"), ("Mistral-7B", None, "mistral7b-base", "mistral7b-it"),
         ("Qwen2.5-72B", "big72_ddx", "qwen25-72b-base", "qwen25-72b-it"),
         ("MedGemma-4B", "@medgemma_ddx", "medgemma-4b-pt", "medgemma-4b-it")]
FS_DIR = {"qwen25-72b-base": "fewshot72_t07", "qwen25-72b-it": "fewshot72_t07",  # worked-example runs without a reference-free run
          "medgemma-4b-pt": "fewshotmg_t07", "medgemma-4b-it": "fewshotmg_t07"}
SETTINGS = ["zs", "fs", "rf"]


def p_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def tops(path, match):
    out = {}
    for rec in map(json.loads, open(path)):
        out[rec["case_id"]] = [stated_dist(parse_dx(s["text"]), match)[1] for s in rec["samples"]]
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
    cl = list(gold)
    rng = np.random.default_rng(0)
    perms = []
    for _ in range(20):
        lab = {c: gold[c2] for c, c2 in zip(cl, rng.permutation(cl))}
        for c in cl:
            while lab[c] == gold[c]:
                lab[c] = gold[cl[rng.integers(len(cl))]]
        perms.append(lab)
    boot = Boot(cl, gold, a.n_boot, rng)

    def per_case(T):
        real, sh = {}, {}
        for c in cl:
            t = T.get(c, [])
            n = len(t)
            v = p_at_k(n, sum(x == gold[c] for x in t), min(32, n)) if n else None
            if v is None:
                continue
            real[c] = v
            sh[c] = float(np.mean([p_at_k(n, sum(x == lab[c] for x in t), min(32, n)) for lab in perms]))
        return real, sh

    report = {"models": {}, "pairs": {}}
    for nm, zd, b, p in PAIRS:
        per = {}
        for st in SETTINGS:
            for t in (b, p):
                if st == "zs":
                    if zd is None:
                        continue
                    path = f"{a.overflow}/{zd[1:]}/{t}/samples.jsonl" if zd.startswith("@") else f"{a.runs}/{zd}/{t}/samples.jsonl"
                else:
                    if t in FS_DIR and st == "rf":
                        continue
                    d = FS_DIR.get(t) or ("fewshot_t07" if st == "fs" else "fewshot_rf_t07")
                    path = f"{a.overflow}/{d}/{t}/samples.jsonl"
                real, sh = per_case(tops(path, match))
                per[(st, t)] = (real, sh)
                report["models"][f"{st}/{t}"] = {"pass32": float(np.mean(list(real.values()))), "shuffled": float(np.mean(list(sh.values())))}
                print(nm, st, t, json.dumps(report["models"][f"{st}/{t}"]), flush=True)
        best = {t: max((s for s in SETTINGS if (s, t) in per), key=lambda s: report["models"][f"{s}/{t}"]["pass32"]) for t in (b, p)}
        res = {"best_setting": best}
        combos = [(st, st) for st in SETTINGS if (st, b) in per] + [("best", "best")]
        for sb, sp in combos:
            kb, kp = (best[b] if sb == "best" else sb, b), (best[p] if sp == "best" else sp, p)
            rb, hb = per[kb]
            rp, hp = per[kp]
            cs = [c for c in cl if c in rb and c in rp]
            d_raw = boot.ratio({c: rp[c] - rb[c] for c in cs}, {c: 1.0 for c in cs})
            d_cor = boot.ratio({c: (rp[c] - hp[c]) - (rb[c] - hb[c]) for c in cs}, {c: 1.0 for c in cs})
            res[sb] = {"raw": d_raw, "corrected": d_cor}
        report["pairs"][nm] = res
        print(nm, json.dumps({k: [round(100 * v["raw"]["diff"], 1), round(100 * v["corrected"]["diff"], 1)] for k, v in res.items() if k != "best_setting"}), best, flush=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
