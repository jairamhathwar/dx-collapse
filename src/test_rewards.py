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

# --- group_match: sampled-distribution matching within a GRPO group ---
from rewards import group_match_rewards

def ans(name):
    return j((name, 0.9))

p = [["Pneumonia", 0.5], ["Bronchitis", 0.25], ["URTI", 0.25]]
cases = {
    "collapsed 8x Pneumonia": ["Pneumonia"] * 8,
    "matches reference 4/2/2": ["Pneumonia"] * 4 + ["Bronchitis"] * 2 + ["URTI"] * 2,
    "6 Pneumonia + 2 Bronchitis": ["Pneumonia"] * 6 + ["Bronchitis"] * 2,
    "off-reference + invalid": ["Pneumonia"] * 6 + ["Asthma2", None],
}
for label, names in cases.items():
    texts = [ans(nm) if nm else "no answer" for nm in names]
    r, sizes = group_match_rewards(texts, ["c1"] * len(texts), [p] * len(texts), m)
    print(f"{label:28s} rewards={[round(x, 2) for x in r]} sum={sum(r):.2f} group_sizes={sizes}")
# two cases interleaved in one batch must be grouped by case_id, not by position
texts = [ans("Pneumonia"), ans("URTI"), ans("Pneumonia"), ans("URTI")]
r, sizes = group_match_rewards(texts, ["a", "b", "a", "b"], [p, [["URTI", 1.0]], p, [["URTI", 1.0]]], m)
print("interleaved a,b,a,b         ", [round(x, 2) for x in r], sizes)
gm = make_trl_reward("group_match", conds)
print("trl group_match:", gm(completions=[ans("Pneumonia")] * 2 + [ans("Bronchitis")] * 2,
                             case_id=["x"] * 4, gt_differential=[json.dumps(p)] * 4))
