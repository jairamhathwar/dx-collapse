"""Why are samples unparsed or unmapped? Prints finish reasons and the most common unmapped names.

    python diagnose_runs.py --runs-dir $THESIS/runs/go --normalizer $THESIS/data/ontology/normalizer.pkl
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from disease_normalizer import DiseaseNormalizer

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", required=True)
ap.add_argument("--normalizer", required=True)
a = ap.parse_args()
norm = DiseaseNormalizer.load(a.normalizer)
cache = {}

for f in sorted(Path(a.runs_dir).glob("*/samples.jsonl")):
    finish, unmapped, lengths = Counter(), Counter(), Counter()
    unparsed_tail = None
    for rec in map(json.loads, open(f)):
        for s in rec["samples"]:
            finish[(s["finish"], "parsed" if s["dx"] else "UNPARSED")] += 1
            if not s["dx"] and unparsed_tail is None and s["finish"] == "stop":
                unparsed_tail = s["text"][-400:]
            if s["dx"]:
                name = s["dx"][0]["diagnosis"]
                if name not in cache:
                    cache[name] = norm.resolve(name)[1]
                if cache[name] == "unmapped":
                    unmapped[name] += 1
    print(f"\n=== {f.parent.name}")
    print("finish x parsed:", dict(finish))
    print("top unmapped top-1 names:")
    for n, c in unmapped.most_common(25):
        print(f"   {c:4d}  {n}")
    if unparsed_tail:
        print("example unparsed (finish=stop) tail:\n   ", unparsed_tail.replace("\n", " | "))
