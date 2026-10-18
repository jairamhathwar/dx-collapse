"""Print a few sampled differentials with their grades, for eyeballing the grader.

    python inspect_samples.py SAMPLES CASES NORMALIZER [n_cases]
"""
import json
import sys

from disease_normalizer import DiseaseNormalizer

samples, cases_path, norm_path = sys.argv[1:4]
n_show = int(sys.argv[4]) if len(sys.argv) > 4 else 6
cases = {c["case_id"]: c for c in map(json.loads, open(cases_path))}
norm = DiseaseNormalizer.load(norm_path)

for i, line in enumerate(open(samples)):
    if i >= n_show:
        break
    rec = json.loads(line)
    case = cases[rec["case_id"]]
    s = rec["samples"][0]
    print(f"\nGOLD: {case['disease_label']} ({case['disease_id']}) | finish={s['finish']} chars={len(s['text'])}")
    for d in s["dx"][:5]:
        g = norm.grade(d["diagnosis"], case["disease_id"])
        print(f"  p={d['probability']}  {d['diagnosis'][:70]:70s} -> {g['tier']} ({g['method']})")
