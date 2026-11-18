"""Item-level view of a base-vs-RL pass@k crossover: are the base model's extra solves chance-like?

    python math_items.py CASES.jsonl "BASE_GLOB" "RL_GLOB" out.json [K]

Uses all n samples per problem (any-of-n). For each model and problem: c = samples matching the gold answer.
  prior(g)      how often the model emits answer string g on OTHER problems (share of all its samples elsewhere)
  For each problem solved by only one model: gold, c, the model's hit rate c/n, and prior(gold). A solve is called
  'chance-like' when c/n <= 3 * prior(gold) (the model produces the gold answer here no more than about as often as it
  produces that string anywhere) and 'specific' otherwise.
Paired bootstrap over problems (2,000 resamples) for: raw gap (base - RL any-of-n), null gap (permuted gold), and the
gap after dropping chance-like solves from both models.
"""
import glob
import json
import random
import sys
from collections import Counter

from math_an import last_boxed, norm, N_PERM


def load(pat):
    ans = {}
    for f in sorted(glob.glob(pat)):
        for rec in map(json.loads, open(f)):
            ans[rec["case_id"]] = [norm(b) if (b := last_boxed(s["text"])) is not None else None for s in rec["samples"]]
    return ans


def main():
    cases = [json.loads(l) for l in open(sys.argv[1])]
    gold = {c["case_id"]: norm(c["answer"]) for c in cases}
    A = {"base": load(sys.argv[2]), "rl": load(sys.argv[3])}
    ids = [c["case_id"] for c in cases if all(c["case_id"] in A[m] for m in A)]
    n = min(len(A[m][i]) for m in A for i in ids)
    if len(sys.argv) > 5:
        n = min(n, int(sys.argv[5]))
    info = {}
    for m in A:
        tot = Counter()
        per = {}
        for i in ids:
            per[i] = Counter(a for a in A[m][i][:n] if a is not None)
            tot.update(per[i])
        N = n * len(ids)
        info[m] = {}
        for j, i in enumerate(ids):
            g = gold[i]
            c = per[i][g]
            prior = (tot[g] - c) / (N - n)
            nullhit = 0.0
            for p in range(1, N_PERM + 1):
                g2 = gold[ids[(j + p * (len(ids) // (N_PERM + 1) or 1)) % len(ids)]]
                if g2 != g:
                    nullhit += float(per[i][g2] > 0) / N_PERM
            info[m][i] = {"c": c, "prior": prior, "solved": c > 0, "nullhit": nullhit,
                          "chance_like": c > 0 and c / n <= 3 * prior}
    rows = []
    for i in ids:
        b, r = info["base"][i], info["rl"][i]
        if b["solved"] != r["solved"]:
            w = "base" if b["solved"] else "rl"
            x = info[w][i]
            rows.append({"case_id": i, "only": w, "gold": gold[i][:40], "c": x["c"], "rate": x["c"] / n,
                         "prior": x["prior"], "chance_like": x["chance_like"]})
    rng = random.Random(0)

    def gaps(sample):
        k = len(sample)
        raw = sum(info["base"][i]["solved"] - info["rl"][i]["solved"] for i in sample) / k
        null = sum(info["base"][i]["nullhit"] - info["rl"][i]["nullhit"] for i in sample) / k
        spec = sum((info["base"][i]["solved"] and not info["base"][i]["chance_like"])
                   - (info["rl"][i]["solved"] and not info["rl"][i]["chance_like"]) for i in sample) / k
        return raw, null, spec, raw - null

    point = gaps(ids)
    bs = [gaps([rng.choice(ids) for _ in ids]) for _ in range(2000)]
    ci = lambda j: [sorted(x[j] for x in bs)[int(q * len(bs))] for q in (0.025, 0.975)]
    summ = {"n_items": len(ids), "n_samples": n,
            "any_of_n": {m: sum(info[m][i]["solved"] for i in ids) / len(ids) for m in A},
            "null_any_of_n": {m: sum(info[m][i]["nullhit"] for i in ids) / len(ids) for m in A},
            "chance_like_solves": {m: sum(info[m][i]["chance_like"] for i in ids) for m in A},
            "only_base": sum(r["only"] == "base" for r in rows), "only_rl": sum(r["only"] == "rl" for r in rows),
            "only_base_chance_like": sum(r["only"] == "base" and r["chance_like"] for r in rows),
            "only_rl_chance_like": sum(r["only"] == "rl" and r["chance_like"] for r in rows),
            "raw_gap": [point[0], ci(0)], "null_gap": [point[1], ci(1)], "specific_gap": [point[2], ci(2)],
            "raw_minus_null": [point[3], ci(3)]}
    print(json.dumps(summ, indent=1))
    for r in sorted(rows, key=lambda r: (r["only"], -r["rate"])):
        print(r["only"], "gold=%-22s c=%3d rate=%.4f prior=%.5f %s" % (r["gold"], r["c"], r["rate"], r["prior"],
                                                                      "CHANCE-LIKE" if r["chance_like"] else ""))
    json.dump({"summary": summ, "items": rows}, open(sys.argv[4], "w"), indent=1)


if __name__ == "__main__":
    main()
