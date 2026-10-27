"""Must-not-miss condition sets defined outside DDXPlus, written as release_conditions.json copies whose severity field
is 1 for conditions in the set and 5 otherwise, so every analysis that reads "severity <= 2" uses the external set.

    python make_mnm_sets.py --conditions-json release_conditions.json --out-dir DIR

harm: DDXPlus conditions among the 15 diseases that cause about half of serious misdiagnosis-related harms in the USA
      (Newman-Toker et al., BMJ Qual Saf 2024;33:109-120): myocardial infarction, venous thromboembolism, pneumonia,
      lung cancer. Stroke, sepsis, arterial thromboembolism, aortic aneurysm/dissection, meningitis/encephalitis, spinal
      abscess, endocarditis and the other cancers are not DDXPlus conditions.
chestpain: life-threatening causes of acute chest pain in the 2021 AHA/ACC chest pain guideline (Gulati et al.,
      Circulation 2021;144:e368-e454): acute coronary syndrome, pulmonary embolism, esophageal rupture, and
      (tension) pneumothorax among the nonvascular emergencies. Aortic dissection is not a DDXPlus condition.
union: either set.
"""
import argparse
import copy
import json
from pathlib import Path

SETS = {
    "harm": ["Possible NSTEMI / STEMI", "Pulmonary embolism", "Pneumonia", "Pulmonary neoplasm"],
    "chestpain": ["Possible NSTEMI / STEMI", "Unstable angina", "Pulmonary embolism", "Boerhaave", "Spontaneous pneumothorax"],
}
SETS["union"] = sorted(set(SETS["harm"]) | set(SETS["chestpain"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions-json", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    raw = json.load(open(a.conditions_json))
    names = {c["cond-name-eng"] for c in raw.values()}
    for name, conds in SETS.items():
        missing = set(conds) - names
        assert not missing, missing
        out = copy.deepcopy(raw)
        for c in out.values():
            c["severity"] = 1 if c["cond-name-eng"] in conds else 5
        p = Path(a.out_dir) / f"release_conditions_mnm_{name}.json"
        json.dump(out, open(p, "w"), ensure_ascii=False)
        print(name, len(conds), p)


if __name__ == "__main__":
    main()
