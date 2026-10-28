"""How often the 32 samples never name the true pathology while the first valid stated list does (DDXPlus, zero-shot,
T=0.7). Top answers and lists as in every table (stated_vs_sampled.stated_dist). Workshop Figure 1 caption and text.

    python miss_list_rate.py RUNS DATA_DIR out.json
"""
import json
import sys

from ddx_names import Matcher
from parsing import parse_dx
from stated_vs_sampled import stated_dist
R, D = sys.argv[1], sys.argv[2]
raw = json.load(open(f"{D}/release_conditions.json"))
match = Matcher(sorted(c["cond-name-eng"] for c in raw.values()))
gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(f"{D}/cases_test.jsonl"))}
PAIRS = [("pair_q3", "qwen3-8b-base", "qwen3-8b"), ("pair25_t07", "qwen25-7b-base", "qwen25-7b-it"), ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it"),
         ("ddx_olmo", "olmo3-base", "olmo3-instruct"), ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct"), ("big72_ddx", "qwen25-72b-base", "qwen25-72b-it")]
out = {}
for d, b, p in PAIRS:
    for t in (b, p):
        n = hit = sampmiss = 0
        for rec in map(json.loads, open(f"{R}/{d}/{t}/samples.jsonl")):
            v = [stated_dist(parse_dx(s["text"]), match) for s in rec["samples"]]
            tops = [x for _, x in v if x]; first = next((x for x, _ in v if x), None)
            if not tops or first is None: continue
            g = gold[rec["case_id"]]; n += 1
            sampmiss += g not in tops
            hit += g not in tops and g in first
        out[f"{d}/{t}"] = {"n": n, "samples_miss": sampmiss / n, "samples_miss_list_has": hit / n}
        print(t, n, round(sampmiss / n, 3), round(hit / n, 3), flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
