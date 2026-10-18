"""Sample K ranked differentials per case with vLLM. Resumable and shardable.

    python sample_diagnoses.py --model Qwen/Qwen3-8B --cases cases.jsonl \
        --out runs/qwen3-8b/samples.jsonl --k 32 [--thinking] [--limit 200] \
        [--shard 0 --num-shards 4]

Writes one JSON line per case: {case_id, model, k, samples: [{text, dx: [{diagnosis, probability}]}]}.
Cases already present in --out are skipped, so a preempted job can simply be resubmitted.
"""
import argparse
import json
import os
import random
from pathlib import Path

from parsing import parse_dx

SYSTEM = "You are an expert clinical geneticist and diagnostician."
INSTRUCTIONS = (
    "Based on the clinical features above, give your differential diagnosis.\n"
    "List the {n} most likely specific diagnoses (use the specific disease or subtype name, "
    "e.g. 'Spastic paraplegia 7' rather than 'hereditary spastic paraplegia', when you can), "
    "most likely first, with your probability that each is the correct diagnosis.\n"
    "End your answer with a JSON array in exactly this format:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>}}, ...]'
)
MAX_EXCLUDED = 10

DDX_SYSTEM = "You are an experienced emergency and primary care physician."
DDX_INSTRUCTIONS = (
    "Based on the patient's answers above, give your differential diagnosis.\n"
    "Choose ONLY from this list of possible conditions:\n{conditions}\n\n"
    "List the {n} most likely conditions from the list, most likely first, with your probability "
    "that each is the correct diagnosis. Use the condition names exactly as written in the list.\n"
    "End your answer with a JSON array in exactly this format:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>}}, ...]'
)


def build_ddx_prompt(case, n_dx, conditions):
    return case["vignette"] + "\n\n" + DDX_INSTRUCTIONS.format(
        conditions="\n".join(f"- {c}" for c in conditions), n=n_dx)


def build_prompt(case, n_dx, rng):
    lines = case["vignette"].split("\n")
    # cap the 'absent' list so prompts stay short; deterministic per case
    if "Features explicitly noted as absent:" in lines:
        i = lines.index("Features explicitly noted as absent:")
        absent = lines[i + 1:]
        if len(absent) > MAX_EXCLUDED:
            absent = sorted(rng.sample(absent, MAX_EXCLUDED), key=absent.index)
        lines = lines[:i + 1] + absent
    return "\n".join(lines) + "\n\n" + INSTRUCTIONS.format(n=n_dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-dx", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking", action="store_true", help="enable_thinking for Qwen3-style templates")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--case-ids", help="file with one case_id per line (evaluation subset)")
    ap.add_argument("--shard", type=int, default=0, help="-1 = use SLURM_ARRAY_TASK_ID")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-cases", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true", help="skip torch.compile/CUDA graphs")
    ap.add_argument("--task", choices=["phenopacket", "ddxplus"], default="phenopacket")
    ap.add_argument("--conditions", help="ddxplus: JSON list of allowed condition names")
    a = ap.parse_args()
    conditions = json.load(open(a.conditions)) if a.task == "ddxplus" else None
    # Slurm job arrays: shard = array index, and --out may contain "{shard}"
    if a.shard < 0:
        a.shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    a.out = a.out.format(shard=a.shard)

    cases = [json.loads(l) for l in open(a.cases)]
    if a.case_ids:
        keep = set(Path(a.case_ids).read_text().split())
        cases = [c for c in cases if c["case_id"] in keep]
    cases = cases[a.shard::a.num_shards]
    if a.limit:
        cases = cases[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["case_id"] for l in out.open()} if out.exists() else set()
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(cases)} cases in shard, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_model_len, seed=a.seed, trust_remote_code=True,
              enforce_eager=a.enforce_eager)
    # Base models don't treat the chat end-of-turn token as EOS; stop there explicitly
    # (no-op for instruct models, which already stop on it).
    vocab = llm.get_tokenizer().get_vocab()
    stop_ids = [vocab[t] for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>") if t in vocab]
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, seed=a.seed, stop_token_ids=stop_ids)
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": a.thinking}}

    for b in range(0, len(todo), a.batch_cases):
        batch = todo[b:b + a.batch_cases]
        if a.task == "ddxplus":
            msgs = [[{"role": "system", "content": DDX_SYSTEM},
                     {"role": "user", "content": build_ddx_prompt(c, a.n_dx, conditions)}] for c in batch]
        else:
            msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_prompt(c, a.n_dx, random.Random(c["case_id"]))}]
                    for c in batch]
        try:
            res = llm.chat(msgs, sp, use_tqdm=False, **chat_kwargs)
        except TypeError:  # template without enable_thinking support
            res = llm.chat(msgs, sp, use_tqdm=False)
        with out.open("a") as fh:
            for c, r in zip(batch, res):
                samples = [{"text": o.text, "dx": parse_dx(o.text),
                            "finish": o.finish_reason} for o in r.outputs]
                fh.write(json.dumps({"case_id": c["case_id"], "model": a.model, "k": a.k,
                                     "temperature": a.temperature, "thinking": a.thinking,
                                     "samples": samples}) + "\n")
        print(f"wrote {b + len(batch)}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
