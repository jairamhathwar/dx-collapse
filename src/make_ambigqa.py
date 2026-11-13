"""Build AmbigQA cases for sample_diagnoses.py --task ambigqa.

    python make_ambigqa.py OUT_DIR        # needs `datasets`; downloads sewon/ambig_qa (config "light")

Writes cases_train.jsonl and cases_test.jsonl (the public validation split; AmbigQA has no public test set):
    {case_id, question, answers: [[alias, ...], ...], type: "single" | "multi"}
`answers` has one alias list per acceptable answer: the single answer, or one per disambiguated interpretation, pooled
over annotators (alias lists that share a normalized alias are merged). type is "multi" if any annotator marked the
question ambiguous.
"""
import json
import sys
from pathlib import Path

from qa_norm import normalize


def merge(groups):
    """Union alias lists that share a normalized alias."""
    out = []
    for g in groups:
        keys = {normalize(a) for a in g} - {""}
        if not keys:
            continue
        hit = [o for o in out if o[0] & keys]
        for o in hit:
            keys |= o[0]
            g = o[1] + g
            out.remove(o)
        out.append((keys, list(dict.fromkeys(g))))
    return [g for _, g in out]


def convert(row):
    ann, groups, multi = row["annotations"], [], False
    for t, ans, qa in zip(ann["type"], ann["answer"], ann["qaPairs"]):
        if t == "singleAnswer":
            groups.append(list(ans))
        else:
            multi = True
            groups.extend(list(a) for a in qa["answer"])
    return {"case_id": f"ambigqa_{row['id']}", "question": row["question"], "answers": merge(groups),
            "type": "multi" if multi else "single"}


def main():
    from datasets import load_dataset
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("sewon/ambig_qa", "light")
    for split, name in (("train", "cases_train.jsonl"), ("validation", "cases_test.jsonl")):
        rows = [c for c in map(convert, ds[split]) if c["answers"]]
        with open(out / name, "w") as fh:
            for c in rows:
                fh.write(json.dumps(c) + "\n")
        n_multi = sum(c["type"] == "multi" for c in rows)
        print(f"{name}: {len(rows)} cases, {n_multi} multi-answer")


if __name__ == "__main__":
    main()
