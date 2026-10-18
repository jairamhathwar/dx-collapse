"""Sanity checks for the grader's tiers. Run: python test_grading.py NORMALIZER_PKL"""
import sys

from disease_normalizer import DiseaseNormalizer

EXPECT = [
    ("Turner syndrome", "OMIM:147791", "wrong"),          # sibling via broad parent
    ("Noonan syndrome", "OMIM:147791", "wrong"),
    ("Williams syndrome", "OMIM:147791", "wrong"),
    ("Jacobsen syndrome", "OMIM:147791", "exact"),
    ("Hereditary spastic paraplegia type 7 (SPG7)", "OMIM:607259", "exact"),
    ("Spastic paraplegia 7", "OMIM:607259", "exact"),
    ("Hereditary spastic paraplegia", "OMIM:607259", "related"),
    ("Noonan syndrome", "OMIM:163950", None),             # report only: group vs. NS1
    ("Marfan syndrome", "OMIM:607259", "wrong"),
]

norm = DiseaseNormalizer.load(sys.argv[1])
fails = 0
for name, gold, want in EXPECT:
    r = norm.grade(name, gold)
    m = r.get("mondo")
    nd = norm.n_descendants(m) if m else None
    ok = want is None or r["tier"] == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:45s} vs {gold}: {r['tier']:8s} {r.get('rel', ''):8s} ndesc={nd}")
print(f"{fails} failures")
