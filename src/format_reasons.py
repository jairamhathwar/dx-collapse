"""Tally why DDXPlus samples fail the strict reward format, and which off-list names appear.

    python format_reasons.py SAMPLES_JSONL CONDITIONS_JSON
"""
import json
import sys
from collections import Counter

from ddx_names import Matcher
from parsing import parse_dx
from rewards import parse_distribution

samples, conditions = sys.argv[1:3]
allowed = Matcher(json.load(open(conditions)))
reasons, offlist, n = Counter(), Counter(), 0
for rec in map(json.loads, open(samples)):
    for s in rec["samples"]:
        n += 1
        _, why = parse_distribution(s["text"], allowed)
        reasons[why or "ok"] += 1
        if why == "offlist":
            offlist.update(d["diagnosis"] for d in parse_dx(s["text"]) if allowed(d["diagnosis"]) is None)
print(f"{n} samples")
for r, c in reasons.most_common():
    print(f"  {r:13s} {c:7d}  {c / n:.1%}")
print("top off-list names:", offlist.most_common(15))
