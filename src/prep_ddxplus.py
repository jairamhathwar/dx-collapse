"""Render DDXPlus patients as text vignettes with their ground-truth differential.

    python prep_ddxplus.py --dir $THESIS/data/ddxplus --split test \
        --out $THESIS/data/ddxplus/cases_test.jsonl --per-condition 20

Each record: case_id, gold (condition name), gt_differential [[name, prob], ...],
initial_evidence, vignette, age, sex.
"""
import argparse
import ast
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# Known bad French->English translations in release_evidences.json
FIX_EN = {
    "haunting": "shooting",            # lancinante
    "heartbreaking": "tearing",        # déchirante
    "a knife stroke": "stabbing",      # un coup de couteau
    "a pulse": "pulsating",            # une pulsation
    "palace": "palate",                # palais
    "scary": "frightening",            # épeurante
    "sickening": "nauseating",         # écoeurante
    "heavy": "heaviness",              # une lourdeur
    "a cramp": "cramping",
    "burning": "burning",
}
SKIP_VALUES = {"NA", "N", "nowhere"}


def lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def load_evidences(path):
    ev = json.load(open(path))
    for e in ev.values():
        e["value_meaning"] = {k: FIX_EN.get(v["en"], v["en"]) for k, v in lit(e["value_meaning"]).items()}
    return ev


def render(row, ev):
    sex = {"M": "male", "F": "female"}.get(row["SEX"], "patient")
    age = int(row["AGE"])
    who = f"{age}-year-old {sex}" if age >= 1 else f"infant ({sex})"
    article = "An" if who[0] in "8aeiou" or who.startswith(("11-", "18-")) else "A"

    answers = defaultdict(list)  # evidence -> list of values (in data order)
    for tok in lit(row["EVIDENCES"]):
        name, _, val = tok.partition("_@_")
        answers[name].append(val)

    def line(name):
        e = ev[name]
        q = e["question_en"].strip()
        vals = answers[name]
        if e["data_type"] == "B":
            return f"- {q} Yes."
        shown = [e["value_meaning"].get(v, v) for v in vals]
        shown = [s for s in shown if s not in SKIP_VALUES]
        if not shown:
            return None
        scale = " (scale 0-10)" if not e["value_meaning"] and all(s.isdigit() for s in shown) else ""
        return f"- {q}{scale} {', '.join(shown)}."

    def is_history(name):
        return str(ev[name]["is_antecedent"]).lower() == "true"

    init = row["INITIAL_EVIDENCE"]
    symptoms = [n for n in answers if not is_history(n)]
    history = [n for n in answers if is_history(n)]
    # chief complaint first
    symptoms.sort(key=lambda n: n != init)
    out = [f"{article} {who} presents for evaluation.", "Symptoms (patient answers):"]
    out += [l for n in symptoms if (l := line(n))]
    if history:
        out.append("Relevant history and risk factors:")
        out += [l for n in history if (l := line(n))]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "validate", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-condition", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    d = Path(a.dir)
    ev = load_evidences(d / "release_evidences.json")
    conds = sorted(c["cond-name-eng"] for c in json.load(open(d / "release_conditions.json")).values())
    by_cond = defaultdict(list)
    for i, row in enumerate(csv.DictReader(open(d / f"release_{a.split}_patients"))):
        by_cond[row["PATHOLOGY"]].append((i, row))

    rng = random.Random(a.seed)
    recs = []
    for c in conds:
        rows = by_cond.get(c, [])
        for i, row in rng.sample(rows, min(a.per_condition, len(rows))):
            recs.append({
                "case_id": f"ddx_{a.split}_{i}",
                "gold": row["PATHOLOGY"],
                "gt_differential": lit(row["DIFFERENTIAL_DIAGNOSIS"]),
                "initial_evidence": row["INITIAL_EVIDENCE"],
                "age": int(row["AGE"]),
                "sex": row["SEX"],
                "vignette": render(row, ev),
            })
    rng.shuffle(recs)
    with open(a.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    (Path(a.out).parent / "conditions.json").write_text(json.dumps(conds, indent=1))
    print(f"{len(recs)} cases over {len({r['gold'] for r in recs})} conditions -> {a.out}")
    print("\nexample:\n" + recs[0]["vignette"] + f"\nGOLD: {recs[0]['gold']}\nGT DDX: {recs[0]['gt_differential'][:5]}")


if __name__ == "__main__":
    main()
