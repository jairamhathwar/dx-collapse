"""MedCaseReasoning test split (897 published PMC case reports) as cases JSONL for sample_diagnoses.py --task casereport.

    python -I prep_medcase.py --parquet test-00000-of-00001.parquet --out cases_test.jsonl

case_id = PMC id; vignette = case_prompt (the presentation up to the point where a differential is made);
gold = final_diagnosis with underscores and CamelCase split; year = publication year.
"""
import argparse
import json
import re

import pandas as pd


def clean(s):
    s = s.replace("_", " ").strip()
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return re.sub(r"\s+", " ", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_parquet(a.parquet)
    with open(a.out, "w") as fh:
        for r in d.itertuples():
            m = re.search(r"(19|20)\d\d", str(r.publication_date))
            fh.write(json.dumps({"case_id": r.pmcid, "vignette": r.case_prompt.strip(), "gold": clean(r.final_diagnosis),
                                 "gold_raw": r.final_diagnosis, "year": int(m.group(0)) if m else None}) + "\n")
    print(len(d), "cases ->", a.out)


if __name__ == "__main__":
    main()
