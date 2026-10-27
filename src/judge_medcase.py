"""Grade open-ended diagnoses against MedCaseReasoning gold diagnoses with a judge model (vLLM, greedy).

    python judge_medcase.py --cases cases_test.jsonl --runs RUNS/medcase_t07 --tags a,b,... \
        --model mistralai/Mistral-Small-24B-Instruct-2501 --out judge.jsonl [--sample 2000 --seed 0]

Every distinct (case, normalized name) among the parsed list items of all samples of the given runs is judged once.
--sample N judges a random subset of N pairs (for a second judge to measure agreement). With --ddx, each name is
matched against the numbered authors' differential instead (output: item number, 0 = none). Resumable.
"""
import argparse
import json
import random
from pathlib import Path

from disease_normalizer import norm_text
from parsing import parse_dx

def sample_files(d):
    """the merged samples.jsonl if present, else the shards samples.s*.jsonl"""
    d = Path(d)
    return [d / "samples.jsonl"] if (d / "samples.jsonl").exists() else sorted(d.glob("samples.s*.jsonl"))


PROMPT = ("You are grading a diagnosis against the confirmed final diagnosis of a published case report.\n"
          "Final diagnosis: {gold}\nCandidate diagnosis: {cand}\n"
          "Does the candidate name the same disease as the final diagnosis? Count it as the same if it is a synonym, "
          "an abbreviation, or a more specific form of the final diagnosis. Count it as different if it is a broader "
          "category, a related but distinct disease, or only a complication, cause or finding.\n"
          "Answer with one word: yes or no.")

PROMPT_DDX = ("You are matching a diagnosis against the differential diagnosis written by the authors of a published case report.\n"
              "Authors' differential:\n{items}\nCandidate diagnosis: {cand}\n"
              "Which item names the same disease as the candidate? Count a synonym, an abbreviation or a more specific form of "
              "an item as the same; a broader category or a related but distinct disease is not the same.\n"
              "Answer with the item's number only, or 0 if none matches.")


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
    ap.add_argument("--ddx", help="authors' differential JSONL (extract_medcase_ddx.py): match names against it instead of the final diagnosis")
    a = ap.parse_args()
    cases = {c["case_id"]: c for c in map(json.loads, open(a.cases))}
    ddx = {r["case_id"]: r["ddx"] for r in map(json.loads, open(a.ddx))} if a.ddx else None
    pairs = {}
    for tag in a.tags.split(","):
        for rec in (json.loads(l) for f in sample_files(Path(a.runs) / tag) for l in open(f)):
            for s in rec["samples"]:
                for d in parse_dx(s["text"]):
                    k = norm_text(d["diagnosis"])
                    if k:
                        pairs.setdefault((rec["case_id"], k), d["diagnosis"][:200])
    keys = sorted(k for k in pairs if not ddx or ddx.get(k[0]))
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
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    for b in range(0, len(todo), a.batch):
        batch = todo[b:b + a.batch]
        if ddx:
            msgs = [[{"role": "user", "content": PROMPT_DDX.format(items="\n".join(f"{i + 1}. {x}" for i, x in enumerate(ddx[c])),
                                                                   cand=pairs[(c, k)])}] for c, k in batch]
        else:
            msgs = [[{"role": "user", "content": PROMPT.format(gold=cases[c]["gold"], cand=pairs[(c, k)])}] for c, k in batch]
        res = llm.chat(msgs, sp, use_tqdm=False)
        with out.open("a") as fh:
            for (c, k), r in zip(batch, res):
                t = r.outputs[0].text.strip().lower()
                if ddx:
                    num = "".join(ch for ch in t.split()[0] if ch.isdigit()) if t.split() else ""
                    item = int(num) if num and 1 <= int(num) <= len(ddx[c]) else 0
                    fh.write(json.dumps({"case_id": c, "key": k, "cand": pairs[(c, k)], "raw": t, "item": item}) + "\n")
                else:
                    fh.write(json.dumps({"case_id": c, "key": k, "cand": pairs[(c, k)], "raw": t,
                                         "match": t.startswith("yes")}) + "\n")
        print(f"judged {b + len(batch)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
