"""Readout reversal, recall at a call budget, silent errors and equivalence of drawn answers (DDXPlus, CPU only).

    python r25_reversal.py --runs RUNS --overflow OVERFLOW_RUNS --cases cases_test.jsonl \
        --conditions-json release_conditions.json --out r25.json

Per model and case (top answers and lists as in every table, stated_vs_sampled.stated_dist; invalid samples count as
wrong and contribute no list):
  pass_k        unbiased pass@k of the sampled top answers, k = 1, 2, 4, 8, 16, 32
  list_k        the union of the stated lists of k calls contains the true pathology (mean over 20 random orders of the
                32 calls); list_1 is one call's list
  reversal      list_1_first - pass_32 within the model, where list_1_first is the first valid sample's list (as
                the single-call answer): does one list beat 32 calls' answers? Also chance-corrected:
                each term minus the same quantity scored against the true pathology of another case with a different
                pathology (the 20 permutations of r18_chance.py)
  silent        the first answer (top answer of the first valid sample) is wrong and every other valid sample gives the
                same answer, so agreement cannot flag it; share of wrong first answers
  caught20_*    share of wrong first answers among the 20% of cases with the lowest confidence, by agreement (share of
                the other valid samples with the same top answer) or by the stated probability of the first answer;
                caught20_stated_among_silent: share of silent errors the stated probability refers at that rate
  drawn         as r23_reanalysis.py: the post-trained model's zero-shot answers drawn from their own lists vs the base
                model with worked examples (pass@32, raw and chance-corrected), now with 90% intervals for equivalence
                within 5 points (two one-sided tests)
Condition-clustered bootstrap (10,000 resamples) for within-model and post-trained minus base differences.
"""
import argparse
import json
import math

import numpy as np

from ddx_names import Matcher
from parsing import parse_dx
from r14_an import Boot
from stated_vs_sampled import stated_dist

ZS = [("Qwen3-8B", "pair_q3", "qwen3-8b-base", "qwen3-8b", False), ("Qwen2.5-7B", "pair25_t07", "qwen25-7b-base", "qwen25-7b-it", False),
      ("Qwen2.5-32B", "big32_ddx", "qwen25-32b-base", "qwen25-32b-it", False), ("OLMo 3 7B", "ddx_olmo", "olmo3-base", "olmo3-instruct", False),
      ("OLMo 3.1 32B", "big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct", False), ("Qwen2.5-72B", "big72_ddx", "qwen25-72b-base", "qwen25-72b-it", False),
      ("Mistral-7B (worked examples)", "@fewshot_t07", "mistral7b-base", "mistral7b-it", True),
      ("MedGemma-4B (worked examples)", "@fewshotmg_t07", "medgemma-4b-pt", "medgemma-4b-it", True)]
FSBASE = {"Qwen3-8B": "@fewshot_t07", "Qwen2.5-7B": "@fewshot_t07", "Qwen2.5-32B": "@fewshot_t07", "OLMo 3 7B": "@fewshot_t07",
          "OLMo 3.1 32B": "@fewshot_t07", "Qwen2.5-72B": "@fewshot72_t07"}
KS = [1, 2, 4, 8, 16, 32]


def p_at_k(n, c, k):
    if n < k:
        return None
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


class Boot2(Boot):
    def ratio(self, num, den):
        r = super().ratio(num, den)
        S, N = np.zeros(len(self.keys)), np.zeros(len(self.keys))
        for c in num:
            i = self.idx[self.gold[c]]
            S[i] += num[c]
            N[i] += den[c]
        b = (self.W @ S) / np.maximum(self.W @ N, 1e-12)
        b = b[(self.W @ N) > 0]
        r["ci90"] = [float(x) for x in np.percentile(b, [5, 95])]
        return r


def main():
    ap = argparse.ArgumentParser()
    for x in ["runs", "overflow", "cases", "conditions-json", "out"]:
        ap.add_argument(f"--{x}", required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(a.cases))}
    cl = list(gold)
    rng = np.random.default_rng(0)
    plab = []  # as r18_chance.py
    for _ in range(20):
        lab = {c: gold[c2] for c, c2 in zip(cl, rng.permutation(cl))}
        for c in cl:
            while lab[c] == gold[c]:
                lab[c] = gold[cl[rng.integers(len(cl))]]
        plab.append(lab)
    boot = Boot2(cl, gold, a.n_boot, rng)
    orng = np.random.default_rng(3)
    orders = [orng.permutation(32) for _ in range(20)]

    def path(d, t):
        return f"{a.overflow}/{d[1:]}/{t}/samples.jsonl" if d.startswith("@") else f"{a.runs}/{d}/{t}/samples.jsonl"

    def read(d, t):
        return {r["case_id"]: [stated_dist(parse_dx(s["text"]), match) for s in r["samples"]] for r in map(json.loads, open(path(d, t)))}

    def per_case(R):
        out = {}
        for c, v in R.items():
            n = len(v)
            tops = [t if (dd is not None and t) else None for dd, t in v]
            lists = [set(dd) if (dd is not None and t) else set() for dd, t in v]
            r = {f"pass_{k}": p_at_k(n, sum(x == gold[c] for x in tops), min(k, n)) for k in KS}
            for k in KS:
                if k > n:
                    continue
                hits = []
                for o in orders:
                    o = [i for i in o if i < n][:k]
                    hits.append(float(any(gold[c] in lists[i] for i in o)))
                r[f"list_{k}"] = float(np.mean(hits))
            first = next((i for i in range(n) if tops[i]), None)
            r["list_1_first"] = float(first is not None and gold[c] in lists[first])
            r["list_1_first_sh"] = float(np.mean([first is not None and lab[c] in lists[first] for lab in plab]))
            r["pass_32_sh"] = float(np.mean([p_at_k(n, sum(x == lab[c] for x in tops), min(32, n)) for lab in plab]))
            if first is not None:
                ans = tops[first]
                others = [tops[i] for i in range(n) if i != first and tops[i]]
                r["wrong"] = float(ans != gold[c])
                r["agree"] = float(np.mean([x == ans for x in others])) if others else None
                r["stated_p"] = v[first][0][ans]
            out[c] = r
        return out

    report = {"models": {}, "within": {}, "pairs": {}, "drawn": {}, "curves": {}}
    for nm, d, b, p, fs in ZS:
        PC = {}
        for t in (b, p):
            R = read(d, t)
            pc = per_case(R)
            PC[t] = pc
            cs = [c for c in cl if c in pc]
            m = {k: float(np.mean([pc[c][k] for c in cs if pc[c].get(k) is not None])) for k in
                 [f"pass_{k}" for k in KS] + [f"list_{k}" for k in KS] + ["list_1_first", "list_1_first_sh", "pass_32_sh"]}
            # silent errors and referral at 20%
            sc = [c for c in cs if pc[c].get("agree") is not None]
            wrong = {c: pc[c]["wrong"] for c in sc}
            silent = {c: float(pc[c]["wrong"] and pc[c]["agree"] == 1.0) for c in sc}
            tie = np.random.default_rng(5)
            caught = {}
            for sig in ["agree", "stated_p"]:
                vals = np.array([pc[c][sig] for c in sc]) + 1e-9 * tie.random(len(sc))
                thr = np.quantile(vals, 0.2)
                caught[sig] = {c: wrong[c] * float(vv <= thr) for c, vv in zip(sc, vals)}
                m[f"caught20_{sig}"] = boot.ratio(caught[sig], wrong)
            caught_diff = boot.ratio({c: caught["stated_p"][c] - caught["agree"][c] for c in sc}, wrong)
            m["caught20_stated_among_silent"] = boot.ratio({c: caught["stated_p"][c] * silent[c] for c in sc}, silent) if sum(silent.values()) else None
            m["silent_share_of_errors"] = boot.ratio(silent, wrong)
            m["silent_share_of_cases"] = float(np.mean(list(silent.values())))
            m["error_rate_first"] = float(np.mean(list(wrong.values())))
            report["models"][f"{nm}/{t}"] = m
            rev = {c: pc[c]["list_1_first"] - pc[c]["pass_32"] for c in cs if pc[c].get("pass_32") is not None}
            revc = {c: (pc[c]["list_1_first"] - pc[c]["list_1_first_sh"]) - (pc[c]["pass_32"] - pc[c]["pass_32_sh"]) for c in rev}
            report["within"][f"{nm}/{t}"] = {"reversal": boot.ratio(rev, {c: 1.0 for c in rev}), "reversal_corrected": boot.ratio(revc, {c: 1.0 for c in revc}),
                                            "caught20_stated_minus_agree": caught_diff}
            report["curves"][f"{nm}/{t}"] = {"pass": [m[f"pass_{k}"] for k in KS], "list": [m.get(f"list_{k}") for k in KS]}
            print(nm, t, json.dumps({k: (round(v, 3) if isinstance(v, float) else round(v["diff"], 3)) for k, v in m.items()}), flush=True)
        Pb, Pp = PC[b], PC[p]
        cs = [c for c in cl if c in Pb and c in Pp]
        dd = {}
        rb = {c: Pb[c]["list_1_first"] - Pb[c]["pass_32"] for c in cs if Pb[c].get("pass_32") is not None and Pp[c].get("pass_32") is not None}
        dd["reversal_post_minus_base"] = boot.ratio({c: (Pp[c]["list_1_first"] - Pp[c]["pass_32"]) - rb[c] for c in rb}, {c: 1.0 for c in rb})
        sb = [c for c in cs if Pb[c].get("agree") is not None and Pp[c].get("agree") is not None]
        dd["silent_cases_post_minus_base"] = boot.ratio({c: float(Pp[c]["wrong"] and Pp[c]["agree"] == 1.0) - float(Pb[c]["wrong"] and Pb[c]["agree"] == 1.0) for c in sb}, {c: 1.0 for c in sb})
        report["pairs"][nm] = dd
        # drawn answers vs the worked-example base (zero-shot pairs only), with 90% intervals
        if not fs and nm in FSBASE:
            R = read(d, p)
            drng = np.random.default_rng(1)
            draw_raw, draw_sh = {}, {}
            for c, v in R.items():
                valid = [(x, t) for x, t in v if x is not None and t]
                if not valid:
                    continue
                draws = [drng.choice(list(x), p=np.array(list(x.values())) / sum(x.values())) for x, _ in valid]
                n = len(v)
                draw_raw[c] = p_at_k(n, sum(y == gold[c] for y in draws), min(32, n))
                draw_sh[c] = float(np.mean([p_at_k(n, sum(y == lab[c] for y in draws), min(32, n)) for lab in plab]))
            Bf = read(FSBASE[nm], b)
            base_raw, base_sh = {}, {}
            for c, v in Bf.items():
                t = [x for _, x in v]
                n = len(t)
                base_raw[c] = p_at_k(n, sum(x == gold[c] for x in t), min(32, n))
                base_sh[c] = float(np.mean([p_at_k(n, sum(x == lab[c] for x in t), min(32, n)) for lab in plab]))
            cs2 = [c for c in cl if c in draw_raw and c in base_raw]
            raw_d = boot.ratio({c: draw_raw[c] - base_raw[c] for c in cs2}, {c: 1.0 for c in cs2})
            cor_d = boot.ratio({c: (draw_raw[c] - draw_sh[c]) - (base_raw[c] - base_sh[c]) for c in cs2}, {c: 1.0 for c in cs2})
            for r_ in (raw_d, cor_d):
                r_["equivalent_5pt"] = bool(r_["ci90"][0] > -0.05 and r_["ci90"][1] < 0.05)
            report["drawn"][nm] = {"raw": raw_d, "corrected": cor_d}
            print(nm, "drawn", json.dumps({k: [round(v["diff"], 3), [round(x, 3) for x in v["ci90"]], v["equivalent_5pt"]] for k, v in report["drawn"][nm].items()}), flush=True)
        print(nm, "within", json.dumps({t: {k: [round(v["diff"], 3), [round(x, 3) for x in v["cluster_ci"]]] for k, v in report["within"][f"{nm}/{t}"].items() if v} for t in (b, p)}), flush=True)
    report["ks"] = KS
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
