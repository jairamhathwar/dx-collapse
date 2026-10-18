"""Sanity checks for reward functions and name canonicalization. Run: python test_rewards.py CONDITIONS_JSON"""
import json
import sys

from ddx_names import Matcher
from rewards import REWARDS, make_trl_reward, parse_distribution

conds = json.load(open(sys.argv[1]))
m = Matcher(conds)
for name, want in [("Acute bronchitis", "Bronchitis"), ("Pulmonary Embolism", "Pulmonary embolism"),
                   ("Laryngospasm", "Larygospasm"), ("Acute myocardial infarction (STEMI/NSTEMI)", "Possible NSTEMI / STEMI"),
                   ("Acute Pulmonary Edema", "Acute pulmonary edema"), ("Aortic dissection", None), ("Pleurisy", None),
                   ("Acute pancreatitis", None), ("Pulmonary_embolism", "Pulmonary embolism")]:
    got = m(name)
    print(f"{'ok  ' if got == want else 'FAIL'} {name!r:48} -> {got!r}")

def j(*pairs): return json.dumps([{"diagnosis": d, "probability": p} for d, p in pairs])
gt = [["Pneumonia", 0.5], ["Bronchitis", 0.3], ["URTI", 0.2]]
text = j(("Pneumonia", .5), ("Acute bronchitis", .3), ("Upper respiratory tract infection", .2))
print("canonicalized dist:", parse_distribution(text, m))
for k in REWARDS:
    print(f"  {k:10s} {REWARDS[k](text, gold='Pneumonia', gt_differential=gt, allowed=m):.3f}")
f = make_trl_reward("overlap", conds)
print("trl overlap:", f(completions=[text], gold=["Pneumonia"], gt_differential=[json.dumps(gt)]))
