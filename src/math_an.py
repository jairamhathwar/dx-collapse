"""pass@k on MATH-500 with a permuted-gold null: how much of large-k pass@k is reachable without solving the item?

    python math_an.py make OUT.jsonl [math500|aime]           # needs `datasets`; writes {case_id, question, answer}
    python math_an.py score CASES.jsonl out.json label:SAMPLES.jsonl [...]   # SAMPLES may be a glob of shards

A sample's answer is the content of its last \\boxed{...}, normalized (spaces, \\left/\\right, \\dfrac/\\tfrac -> \\frac,
\\text{}, \\!, trailing period, degree marks, a leading "x="); correct if it equals the normalized gold string (the usual
MATH string match, without symbolic equivalence, so some correct answers are missed for both models alike).
Per model, for k in 1, 2, 4, ..., n:
  pass_k       unbiased pass@k (1 - C(n-c, k)/C(n, k), c = number of correct samples)
  null_k       the same with each question's samples scored against the gold answers of other questions (mean over
               N_PERM derangement-like shifts): what any-of-k reaches when the answer does not depend on the item
  distinct     distinct normalized answers per question; none: share of samples with no boxed answer
"""
import glob
import json
import math
import re
import sys

N_PERM = 20


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def norm(s):
    s = s.strip().rstrip(".")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(left|right|!|,|;| )", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("\\$", "").replace("$", "").replace("dollars", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"^[a-zA-Z]\s*=\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def make(out, name="math500"):
    from datasets import load_dataset
    with open(out, "w") as fh:
        if name == "math500":
            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            for r in ds:
                fh.write(json.dumps({"case_id": r["unique_id"], "question": r["problem"], "answer": r["answer"],
                                     "subject": r["subject"], "level": r["level"]}) + "\n")
        else:  # aime: AI-MO/aimo-validation-aime (AIME 2022-2024, integer answers)
            ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
            for i, r in enumerate(ds):
                fh.write(json.dumps({"case_id": f"aime_{i}", "question": r["problem"], "answer": str(r["answer"])}) + "\n")
    print(len(ds), "problems")


def score(cases_path, out, specs):
    cases = [json.loads(l) for l in open(cases_path)]
    gold = {c["case_id"]: norm(c["answer"]) for c in cases}
    ids = [c["case_id"] for c in cases]
    report = {}
    for spec in specs:
        label, pat = spec.split(":", 1)
        ans = {}
        for f in sorted(glob.glob(pat)):
            for rec in map(json.loads, open(f)):
                ans[rec["case_id"]] = [norm(b) if (b := last_boxed(s["text"])) is not None else None for s in rec["samples"]]
        have = [i for i in ids if i in ans]
        if not have:
            print(label, "no samples; skipped")
            continue
        n = min(len(ans[i]) for i in have)
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= n]
        r = {"n_items": len(have), "n_samples": n,
             "distinct": sum(len({a for a in ans[i][:n] if a is not None}) for i in have) / len(have),
             "none": sum(a is None for i in have for a in ans[i][:n]) / (n * len(have))}
        raw = {k: 0.0 for k in ks}
        null = {k: 0.0 for k in ks}
        m = len(have)
        for j, i in enumerate(have):
            a = ans[i][:n]
            c = sum(x == gold[i] for x in a)
            for k in ks:
                raw[k] += pass_at_k(n, c, k) / m
            for p in range(1, N_PERM + 1):
                g = gold[have[(j + p * (m // (N_PERM + 1) or 1)) % m]]
                if g == gold[i]:
                    continue
                c0 = sum(x == g for x in a)
                for k in ks:
                    null[k] += pass_at_k(n, c0, k) / (m * N_PERM)
        r["pass"] = {str(k): raw[k] for k in ks}
        r["null"] = {str(k): null[k] for k in ks}
        report[label] = r
        print(label, json.dumps({"n": n, "items": m, "distinct": round(r["distinct"], 2), "none": round(r["none"], 3),
                                 "pass": {k: round(v, 3) for k, v in r["pass"].items()},
                                 "null": {k: round(v, 4) for k, v in r["null"].items()}}), flush=True)
    json.dump(report, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2], *sys.argv[3:4])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4:])
