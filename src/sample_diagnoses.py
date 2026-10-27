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

# Published case reports (MedCaseReasoning): open-ended differential, graded later by a judge model.
CASE_SYSTEM = "You are an experienced physician and diagnostician."
CASE_INSTRUCTIONS = (
    "Based on the case above, give your differential diagnosis.\n"
    "List the {n} most likely specific diagnoses, most likely first, with your probability that each is the "
    "correct diagnosis.\n"
    "End your answer with a JSON array in exactly this format:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>}}, ...]'
)

DDX_SYSTEM = "You are an experienced emergency and primary care physician."
DDX_INSTRUCTIONS = (
    "Based on the patient's answers above, give your differential diagnosis.\n"
    "Choose ONLY from this list of possible conditions:\n{conditions}\n\n"
    "List the {n} most likely conditions from the list, most likely first, with your probability "
    "that each is the correct diagnosis. Use the condition names exactly as written in the list.\n"
    "End your answer with a JSON array in exactly this format:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>}}, ...]'
)


# Prompt-robustness variants (DDXPlus only). "default" is the prompt used for all main results.
DDX_PLAIN_INSTRUCTIONS = (
    "Which of the following conditions could explain this patient's presentation?\n{conditions}\n\n"
    "Rank the {n} most likely ones, most likely first, and give the probability you assign to each. "
    "Copy the condition names exactly.\n"
    "Finish with a JSON array in exactly this format:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>}}, ...]'
)
# Must-not-miss elicitation (clinical review control): the most likely conditions, then dangerous ones to rule out.
DDX_MNM_INSTRUCTIONS = (
    "Based on the patient's answers above, give your differential diagnosis.\n"
    "Choose ONLY from this list of possible conditions:\n{conditions}\n\n"
    "List the {n} most likely conditions from the list, most likely first, with your probability that each is "
    "the correct diagnosis. Then add up to 3 dangerous conditions from the list that must be ruled out even "
    "though they are less likely (do not repeat conditions already listed), each with your probability. "
    "Use the condition names exactly as written in the list.\n"
    "End your answer with a JSON array in exactly this format, the most likely conditions first:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>, "must_not_miss": false}}, ..., '
    '{{"diagnosis": "<name>", "probability": <number between 0 and 1>, "must_not_miss": true}}, ...]'
)
DDX_COT_PREFIX = ("First reason step by step about the patient's findings and which conditions they support "
                  "or argue against. Then give your answer.\n")
# Full-distribution elicitation: a probability for every listed condition (stated-readout robustness).
DDX_FULL_INSTRUCTIONS = (
    "Based on the patient's answers above, give your differential diagnosis.\n"
    "Here is the list of possible conditions:\n{conditions}\n\n"
    "Give your probability that each condition is the correct diagnosis, for EVERY condition in the list, "
    "so that the probabilities sum to 1. List them from most to least likely and use the condition names "
    "exactly as written in the list.\n"
    "End your answer with a JSON array in exactly this format:\n"
    '[{{"diagnosis": "<name>", "probability": <number between 0 and 1>}}, ...]'
)


# Few-shot elicitation (review control for base models): three fixed DDXPlus training patients, each
# followed by the top five of its reference differential. Base models get a plain completion prompt;
# post-trained models get the same examples as chat turns.
FEWSHOT_HEADER = ("Emergency and primary care patients' answers, each followed by the physician's differential "
                  "diagnosis: the five most likely conditions, most likely first, with probabilities, chosen ONLY "
                  "from this list of possible conditions:\n{conditions}\n\n")
FEWSHOT_Q = "### Patient\n{vignette}\n### Differential diagnosis\n"


def fewshot_examples(train_path, k=3, seed=0, answers=None, export=None):
    """answers: optional {case_id: answer text} replacing the reference-derived lists (reference-free control);
    export: optional path to write the chosen training cases as JSONL (to elicit answers for them)."""
    rows = [json.loads(l) for l in open(train_path)]
    rng = random.Random(seed)
    rng.shuffle(rows)
    out, golds, chosen = [], set(), []
    for r in rows:
        if r["gold"] in golds:
            continue
        z = sum(p for _, p in r["gt_differential"])
        top = sorted(((c, p / z) for c, p in r["gt_differential"]), key=lambda x: -x[1])[:5]
        ans = json.dumps([{"diagnosis": c, "probability": round(p, 2)} for c, p in top])
        if answers is not None:
            ans = answers[r["case_id"]]
        out.append((r["vignette"], ans))
        chosen.append(r)
        golds.add(r["gold"])
        if len(out) == k:
            break
    if export:
        with open(export, "w") as fh:
            for r in chosen:
                fh.write(json.dumps(r) + "\n")
    return out


def build_fewshot_completion(case, conditions, shots):
    cond = "\n".join(f"- {c}" for c in conditions)
    text = FEWSHOT_HEADER.format(conditions=cond)
    for v, ans in shots:
        text += FEWSHOT_Q.format(vignette=v) + ans + "\n\n"
    return text + FEWSHOT_Q.format(vignette=case["vignette"])


def build_fewshot_messages(case, n_dx, conditions, shots):
    msgs = [{"role": "system", "content": DDX_SYSTEM}]
    for v, ans in shots:
        msgs += [{"role": "user", "content": v + "\n\n" + DDX_INSTRUCTIONS.format(conditions="\n".join(f"- {c}" for c in conditions), n=n_dx)},
                 {"role": "assistant", "content": ans}]
    msgs.append({"role": "user", "content": case["vignette"] + "\n\n" + DDX_INSTRUCTIONS.format(
        conditions="\n".join(f"- {c}" for c in conditions), n=n_dx)})
    return msgs


def build_ddx_messages(case, n_dx, conditions, variant="default"):
    cond = "\n".join(f"- {c}" for c in conditions)
    if variant == "plain":
        return [{"role": "user", "content": case["vignette"] + "\n\n" + DDX_PLAIN_INSTRUCTIONS.format(conditions=cond, n=n_dx)}]
    if variant == "full":
        return [{"role": "system", "content": DDX_SYSTEM},
                {"role": "user", "content": case["vignette"] + "\n\n" + DDX_FULL_INSTRUCTIONS.format(conditions=cond)}]
    if variant == "mustnotmiss":
        return [{"role": "system", "content": DDX_SYSTEM},
                {"role": "user", "content": case["vignette"] + "\n\n" + DDX_MNM_INSTRUCTIONS.format(conditions=cond, n=n_dx)}]
    body = DDX_INSTRUCTIONS.format(conditions=cond, n=n_dx)
    if variant == "cot":
        body = DDX_COT_PREFIX + body
    return [{"role": "system", "content": DDX_SYSTEM}, {"role": "user", "content": case["vignette"] + "\n\n" + body}]


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
    ap.add_argument("--dtype", default="auto", help="vLLM dtype; use bfloat16 for checkpoints stored in float32 "
                    "(vLLM's auto would load those as float16)")
    ap.add_argument("--task", choices=["phenopacket", "ddxplus", "casereport"], default="phenopacket")
    ap.add_argument("--conditions", help="ddxplus: JSON list of allowed condition names")
    ap.add_argument("--chat-template", help="jinja file to use instead of the tokenizer's chat template "
                    "(for base models that ship without one)")
    ap.add_argument("--prompt-variant", choices=["default", "plain", "cot", "full", "fewshot", "mustnotmiss"], default="default",
                    help="ddxplus prompt-robustness variant")
    ap.add_argument("--fewshot-train", help="fewshot: DDXPlus training cases to draw the examples from")
    ap.add_argument("--fewshot-answers", help="fewshot: JSON {case_id: answer} replacing the reference-derived example lists")
    ap.add_argument("--fewshot-export", help="write the chosen few-shot training cases as JSONL and exit")
    ap.add_argument("--completion", action="store_true",
                    help="fewshot: plain completion prompt without a chat template (for base models)")
    a = ap.parse_args()
    conditions = json.load(open(a.conditions)) if a.task == "ddxplus" else None
    if a.fewshot_export:
        fewshot_examples(a.fewshot_train, export=a.fewshot_export)
        return
    answers = json.load(open(a.fewshot_answers)) if a.fewshot_answers else None
    shots = fewshot_examples(a.fewshot_train, answers=answers) if a.prompt_variant == "fewshot" else None
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
              enforce_eager=a.enforce_eager, dtype=a.dtype)
    # Base models don't treat the chat end-of-turn token as EOS; stop there explicitly
    # (no-op for instruct models, which already stop on it).
    vocab = llm.get_tokenizer().get_vocab()
    stop_ids = [vocab[t] for t in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>") if t in vocab]
    sp = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p,
                        max_tokens=a.max_tokens, seed=a.seed, stop_token_ids=stop_ids,
                        stop=["\n###", "\n\n"] if a.completion else None)
    chat_kwargs = {"chat_template_kwargs": {"enable_thinking": a.thinking}}
    if a.chat_template:
        chat_kwargs["chat_template"] = Path(a.chat_template).read_text()

    for b in range(0, len(todo), a.batch_cases):
        batch = todo[b:b + a.batch_cases]
        if a.task == "ddxplus" and a.completion:
            res = llm.generate([build_fewshot_completion(c, conditions, shots) for c in batch], sp, use_tqdm=False)
            msgs = None
        elif a.task == "ddxplus" and a.prompt_variant == "fewshot":
            msgs = [build_fewshot_messages(c, a.n_dx, conditions, shots) for c in batch]
        elif a.task == "ddxplus":
            msgs = [build_ddx_messages(c, a.n_dx, conditions, a.prompt_variant) for c in batch]
        elif a.task == "casereport":
            msgs = [[{"role": "system", "content": CASE_SYSTEM},
                     {"role": "user", "content": c["vignette"] + "\n\n" + CASE_INSTRUCTIONS.format(n=a.n_dx)}] for c in batch]
        else:
            msgs = [[{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build_prompt(c, a.n_dx, random.Random(c["case_id"]))}]
                    for c in batch]
        try:
            if msgs is not None:
                res = llm.chat(msgs, sp, use_tqdm=False, **chat_kwargs)
        except TypeError:  # template without enable_thinking support
            res = llm.chat(msgs, sp, use_tqdm=False,
                           **{k: v for k, v in chat_kwargs.items() if k != "chat_template_kwargs"})
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
