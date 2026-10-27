"""Extract the case authors' own differential from MedCaseReasoning's diagnostic_reasoning field (vLLM, greedy).

    python extract_medcase_ddx.py --parquet test-00000-of-00001.parquet --cases cases_test.jsonl \
        --model mistralai/Mistral-Small-24B-Instruct-2501 --out authors_ddx.jsonl [--limit 8]

diagnostic_reasoning holds numbered statements that quote the published report. The extractor lists every diagnosis the
authors considered other than the final diagnosis (each alternative suspected, considered or ruled out) as a JSON array.
Output per case: {case_id, ddx: [final diagnosis, alternative, ...], n_alternatives, raw}; item 1 is always the final diagnosis.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from disease_normalizer import norm_text
from parsing import _DECODER

PROMPT = ("Below are reasoning statements, with quotations, from a published medical case report. "
          "List every alternative diagnosis that the authors suspected, considered or ruled out for this patient, other "
          "than the final diagnosis. Give each as a short disease name, without explanations; do not repeat the final "
          "diagnosis or variants of it, and do not add diagnoses the text does not mention.\n\nFinal diagnosis: {gold}\n\n"
          "Reasoning statements:\n{reasoning}\n\n"
          'Answer with a JSON array of strings only, e.g. ["Diagnosis A", "Diagnosis B"], or [] if there are none.')


def parse_list(text):
    for i in range(len(text)):
        if text[i] == "[":
            try:
                arr, _ = _DECODER.raw_decode(text, i)
            except Exception:
                continue
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return [x.strip() for x in arr if x.strip()]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    d = pd.read_parquet(a.parquet)
    gold = {c["case_id"]: c["gold"] for c in map(json.loads, open(a.cases))}
    rows = [(r.pmcid, r.diagnostic_reasoning) for r in d.itertuples()][:a.limit]
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=8192, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=300)
    res = llm.chat([[{"role": "user", "content": PROMPT.format(gold=gold[c], reasoning=t)}] for c, t in rows], sp, use_tqdm=False)
    with open(a.out, "w") as fh:
        for (c, _), r in zip(rows, res):
            t = r.outputs[0].text
            alts = [x for x in parse_list(t) if norm_text(x) != norm_text(gold[c])]
            # item 1 is always the report's final diagnosis; the rest are the authors' alternatives
            fh.write(json.dumps({"case_id": c, "ddx": [gold[c]] + alts, "n_alternatives": len(alts), "raw": t}) + "\n")
    print(len(rows), "cases ->", a.out)


if __name__ == "__main__":
    main()
