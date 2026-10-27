"""Grade open-ended diagnoses against MedCaseReasoning gold diagnoses with a judge model (vLLM, greedy).

    python judge_medcase.py --cases cases_test.jsonl --runs RUNS/medcase_t07 --tags a,b,... \
        --model mistralai/Mistral-Small-24B-Instruct-2501 --out judge.jsonl [--sample 2000 --seed 0]

Every distinct (case, normalized name) among the parsed list items of all samples of the given runs is judged once.
--sample N judges a random subset of N pairs (for a second judge to measure agreement). Resumable.
"""
import argparse
import json
import random
from pathlib import Path

from disease_normalizer import norm_text
from parsing import parse_dx

PROMPT = ("You are grading a diagnosis against the confirmed final diagnosis of a published case report.\n"
          "Final diagnosis: {gold}\nCandidate diagnosis: {cand}\n"
          "Does the candidate name the same disease as the final diagnosis? Count it as the same if it is a synonym, "
          "an abbreviation, or a more specific form of the final diagnosis. Count it as different if it is a broader "
          "category, a related but distinct disease, or only a complication, cause or finding.\n"
          "Answer with one word: yes or no.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=4096)
    a = ap.parse_args()
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    pairs = {}
    for tag in a.tags.split(","):
        for rec in map(json.loads, open(Path(a.runs) / tag / "samples.jsonl")):
            for s in rec["samples"]:
                for d in parse_dx(s["text"]):
                    k = norm_text(d["diagnosis"])
                    if k:
                        pairs.setdefault((rec["case_id"], k), d["diagnosis"][:200])
    keys = sorted(pairs)
    if a.sample:
        keys = sorted(random.Random(a.seed).sample(keys, min(a.sample, len(keys))))
    out = Path(a.out)
    done = {(r["case_id"], r["key"]) for r in map(json.loads, out.open())} if out.exists() else set()
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} pairs, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_model_len, seed=0, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=3)
    for b in range(0, len(todo), a.batch):
        batch = todo[b:b + a.batch]
        msgs = [[{"role": "user", "content": PROMPT.format(gold=cases[c]["gold"], cand=pairs[(c, k)])}] for c, k in batch]
        res = llm.chat(msgs, sp, use_tqdm=False)
        with out.open("a") as fh:
            for (c, k), r in zip(batch, res):
                t = r.outputs[0].text.strip().lower()
                fh.write(json.dumps({"case_id": c, "key": k, "cand": pairs[(c, k)], "raw": t,
                                     "match": t.startswith("yes")}) + "\n")
        print(f"judged {b + len(batch)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
