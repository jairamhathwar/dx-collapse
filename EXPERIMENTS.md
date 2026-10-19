# Experiment log

All runs: 32 samples per case, top-5 differential with stated probabilities, graded automatically.
Hyak paths are under `/gscratch/scrubbed/jairamh/thesis/runs/`.

## Datasets
- **Phenopackets** (`data/phenopackets`): Phenopacket Store 0.1.27. 9,163 cases / 758 OMIM diseases (cases with ≥3 phenotypes kept).
  - Go set: 500 cases / 418 diseases.
  - Full eval set: 1,401 / 749 (≤2 cases per disease; 77 cases with ambiguous gold labels removed).
  - Grading: Mondo-based. "exact" = the gold disease. "related" = an ancestor group of the gold with ≤150 descendants within 2 hops, or a subtype of the gold.
- **DDXPlus** (`data/ddxplus`): test split, 980 cases (20 per condition × 49 conditions).
  - Closed-set prompt that lists all 49 conditions.
  - Ground-truth differential used for coverage and total-variation distance.
  - Worst French→English evidence translations fixed.

## Models
| Tag | HF id | Base | Post-training |
|---|---|---|---|
| qwen25-7b-it | Qwen/Qwen2.5-7B-Instruct | n/a | general SFT+RLHF (starting point) |
| alphamed-7b | che111/AlphaMed-7B-instruct-rl | Qwen2.5-7B-Instruct | rule-based RL only |
| huatuo-o1-7b | FreedomIntelligence/HuatuoGPT-o1-7B | Qwen2.5-7B-Instruct | SFT + PPO with verifier |
| m1-7b | UCSC-VLAA/m1-7B-23K | Qwen2.5-7B-Instruct | SFT distillation only (8k tokens) |
| qwen3-8b-base | Qwen/Qwen3-8B-Base | n/a | none (pretrained) |
| qwen3-8b / -think | Qwen/Qwen3-8B | Qwen3-8B-Base | Qwen's general post-training (non-thinking / thinking) |

Base model for AlphaMed, HuatuoGPT-o1 and m1 was verified from each `config.json` `_name_or_path`.

## Runs
| Run dir | Dataset | Models | Temp | Status |
|---|---|---|---|---|
| go | Phenopackets go | Qwen2.5 trio + base | 0.7 | done (m1 rerun at 8k queued; old 4k run in go_old) |
| ddx | DDXPlus | Qwen2.5 trio + base | 0.7 | base + AlphaMed done; HuatuoGPT and m1 running |
| ddx_t1 | DDXPlus | base, AlphaMed | 1.0 (top-p 1.0) | queued |
| ddx_qwen3 | DDXPlus | Qwen3 base, instruct, think | 0.7 | running |
| ddx_qwen3_t1 | DDXPlus | Qwen3 base, instruct | 1.0 (top-p 1.0) | queued |
| go_qwen3 | Phenopackets go | Qwen3 base, instruct, think | 0.7 | running |

## Findings so far
- **Phenopackets (7B)**
  - Accuracy floor: pass@1 is 1–2%, pass@32 is 6–7%, so pass@k differences are within noise.
  - Overconfidence grows with post-training. Mean stated confidence: base 0.51, m1 0.56, HuatuoGPT 0.66, AlphaMed 0.69. Accuracy is under 2% for all of them.
  - Diversity (distinct top-1 answers per 32 samples, unmapped answers included): base 11.5, AlphaMed 11.2, HuatuoGPT 9.4, m1 14.8.
  - Cases published ≤2022 are much easier than 2023–24 cases (pass@32 about 8–9% vs about 1%), which points to contamination.
- **DDXPlus, base vs AlphaMed (T=0.7)**
  - AlphaMed is better at every k, and the gap widens with k: +3.9 points at k=1, +8.4 at k=32 (95% CI +6.0 to +10.6). This does not support a medical-RL sharpening tax.
  - Both models are far more peaked than the ground-truth differential: about 1.8 distinct top-1 answers per 32 samples, entropy about 0.3 vs 1.87 for the ground truth, and about 24–27% of the differential's mass covered.
  - AlphaMed names off-list conditions about 5x more often than the base (15,112 vs 2,712 listed names).
  - Calibration gap is mild: ECE 0.11 (base) and 0.12 (AlphaMed).

- **DDXPlus, early cross-family read (T=0.7, 618 shared cases): Qwen3-8B-Base vs Qwen2.5-7B-Instruct.** Not yet a matched pair.
  - The pass@k curves cross between k=2 and k=4. The base model scores lower at k=1 (0.375 vs 0.471) and higher at k=32 (0.720 vs 0.592). The k=32 gap is +12.8 points, 95% CI +9.6 to +16.3.
  - The base model is about twice as diverse: 3.8 vs 1.8 distinct top-1 answers per 32 samples, entropy 0.80 vs 0.28. It covers more of the reference differential's mass (0.33 vs 0.24) and solves more conditions (44 vs 37).
  - The base model left 10.8% of its samples unparsed; these count as wrong, so its accuracy is if anything understated.
  - This is consistent with general post-training sharpening the differential. The matched pairs (Qwen3-8B, Qwen2.5-7B base) are pending.

- **DDXPlus, matched pair Qwen3-8B-Base vs Qwen3-8B (non-thinking), T=0.7, 980 cases. Result: sharpening tax from general post-training.**
  - **pass@k.** Qwen3-8B is better at k=1 (0.558 vs 0.388; +17.0 points, CI +14.8 to +19.2). The two are tied at k=4 (Δ −0.002, not significant). The base model is better from k=8 on, and at k=32 by 12.2 points (0.730 vs 0.607; CI +9.9 to +14.9).
  - **Collapse.**
    | | Distinct top-1 answers per 32 samples | Entropy | Reference mass covered at k=32 | Covered at k=1 |
    |---|---|---|---|---|
    | Qwen3-8B | 1.27 | 0.11 | 0.217 | 0.193 |
    | Qwen3-8B-Base | 3.76 | 0.78 | 0.322 | 0.140 |

    Reference entropy is 1.87. The instruct model puts its accuracy into the first answer, while the base model keeps exploring.
  - **Calibration of the top-1 answer is better after post-training:** ECE 0.063 vs 0.121.
  - **Contrast with medical post-training.** Medical RL applied on top of an instruct model (AlphaMed vs Qwen2.5-7B-Instruct) did *not* reduce pass@k at large k. So the collapse happens at general post-training, not at the medical stage.
  - **Caveat on parse rates.** 11% of the base model's samples were unparsed and counted as wrong, so the base model's pass@k is a lower bound.

- **DDXPlus at T=1.0, top-p 1.0 (sampling from the model's own distribution): the collapse holds in both families.**
  | Pair (cases) | pass@1 | pass@32 | Δ at k=32 (95% CI) | Distinct top-1 | Entropy (ref 1.87) | Unparsed |
  |---|---|---|---|---|---|---|
  | Qwen2.5-7B base / Instruct (811) | 0.235 / **0.457** | **0.757** / 0.657 | base +10.0 (7.1 to 12.8) | **8.3** / 2.6 | **1.59** / 0.49 | 16% / ~0% |
  | Qwen3-8B-Base / Qwen3-8B (490) | 0.276 / **0.546** | **0.800** / 0.647 | base +15.3 (11.6 to 19.0) | **6.6** / 1.6 | **1.33** / 0.21 | 19% / 0% |

  - The pass@k curves cross at k≈8 for Qwen2.5 and at k=4–8 for Qwen3.
  - Even at T=1.0, the instruct models stay collapsed (entropy 0.2–0.5). The base models' spread comes close to the reference differential's.
  - Unparsed samples count as wrong, so base-model pass@k is a lower bound.
  - Top-1 calibration is better in the instruct models: ECE 0.06–0.14, against 0.22–0.31 for base.
- **Qwen3-1.7B (the GRPO starting point), DDXPlus.**
  - pass@1 0.378, pass@32 0.435 at T=0.7 (0.472 at T=1.0).
  - Collapsed: 1.4 distinct top-1 answers at T=0.7, 1.8 at T=1.0.
  - Reward-format validity after name canonicalization: 90.8%. Failures are 6.0% off-list names (mostly real non-DDXPlus diagnoses such as acute sinusitis or ARDS) and 3.2% duplicates.
- **Per-condition analysis (`src/per_condition.py`, results in `results/per_condition/`).**
  - Metric: *considered@32*, the share of (case, condition) pairs with reference probability ≥0.1 in which the condition is a top-1 answer in at least one of 32 samples. Grouped by DDXPlus severity (1 = most severe), with case-bootstrap CIs.
  - **Post-training lowers consideration in every tier, including the severe tier (17 conditions).**
    | Comparison | Severe tier: base → post-trained | Δ (95% CI) |
    |---|---|---|
    | Qwen3-8B, T=1 | 0.45 → 0.27 | −18.4 (−20.8, −15.9) |
    | Qwen2.5-7B, T=1 | 0.52 → 0.34 | −18.2 (−21.4, −15.2) |
    | Qwen3-8B, T=0.7 | 0.34 → 0.24 | −9.8 |
  - **Some conditions vanish as top-1 answers in Qwen2.5-7B-Instruct.**
    - Unstable angina is the top answer in 1 of 107 eligible cases (base: 47%). When it is the true pathology, pass@32 falls from 1.00 to 0.00. It still appears in 50% of listed differentials (base: 95%).
    - Stable angina: gold pass@32 0.89 → 0.06. Myocarditis: 0.41 → 0.
  - **Attractor diagnoses.** Qwen2.5-7B-Instruct makes NSTEMI/STEMI its top answer in 13.1% of samples (reference 4.4%, base 6.4%), pulmonary embolism 8.2% (reference 3.2%), pneumonia 8.0% (reference 2.4%). NSTEMI pass@1 rises from 0.44 to 0.99.
  - **AlphaMed vs Instruct:** consideration rises +2–4 points in every tier. Acute pulmonary edema drops from 0.31 to 0.13. AlphaMed over-predicts spontaneous pneumothorax (10.3% vs reference 1.0%).
- **E7/E8 GRPO, Qwen3-1.7B, 300 steps** (`slurm/submit_e7.sh`, queued 2026-10-18 00:04).
  - Eight configs:
    | Tag | Rewards | Seed |
    |---|---|---|
    | acc-s0 | acc | 0 |
    | dist-s0 | acc,brier,overlap | 0 |
    | brier-s0 | acc,brier | 0 |
    | overlap-s0 | acc,overlap | 0 |
    | puri-rlvr-s0 | multi_rlvr | 0 |
    | puri-rlcr-s0 | multi_rlvr,set_brier | 0 |
    | acc-s1 | acc | 1 |
    | dist-s1 | acc,brier,overlap | 1 |
  - When each run finishes, its merged model is sampled automatically on DDXPlus test (980 × 32) and on Phenopackets go (500 × 32).
  - Scoring compares every config against the untrained Qwen3-1.7B: `runs/e7_ddx/report` and `runs/e7_go/report`.
  - Puri et al.'s rewards are adapted to our 5-answer prompt with per-answer probabilities (their setup uses K=3 answers).
- **E7 v1 results (2026-10-18): unstable; several runs collapsed to a constant answer.**
  - Settings: lr 2e-5, LoRA r=32, beta 0, 8 generations per prompt, 16 prompts per step, group std normalization.
  - DDXPlus pass@1 (untrained 0.378):
    | Run | pass@1 | Notes |
    |---|---|---|
    | dist-s0 | **0.630** | coverage@32 0.22 vs 0.16 for untrained |
    | overlap-s0 | 0.557 | |
    | puri-rlcr-s0 | 0.440 | |
    | puri-rlvr-s0 | 0.361 | |
    | acc-s0 | 0.095 | answers "Possible NSTEMI / STEMI" in 88% of cases |
    | brier-s0 | 0.037 | "Spontaneous rib fracture" 98%, stated confidence 1.0 |
    | dist-s1 | 0.020 | "Bronchitis" 100% |
  - **Diagnosis.** The starting policy is nearly deterministic (token entropy about 0.06). In acc-s0, 88–100% of GRPO groups had zero reward variance, so noisy updates drove the policy to an input-independent answer, and it could not recover. Collapsed runs also shrank their completions to 21–27 tokens.
  - **Phenopackets transfer** (1.7B is at the accuracy floor, under 1%). Stated confidence: untrained 0.45, overlap-s0 **0.39**, acc-s0 0.33, dist-s0 0.65, puri-rlvr 0.65, puri-rlcr **0.80**, brier-s0 1.00.
    - Tentative: Brier-type calibration rewards learned on DDXPlus *increase* out-of-domain overconfidence; the overlap reward slightly reduces it.
    - Must re-check with stable training.
- **E7 v2 (stabilized), queued 2026-10-18** (`slurm/submit_e7v2.sh`).
  - Settings: lr 5e-6, KL beta 0.04, no reward std normalization, 16 generations × 16 prompts per step, 200 steps. All checkpoints are kept, and a collapse monitor logs `collapse/distinct_top1_per_case`.
  - Pilot: acc and dist, seeds 0 and 1. If it is stable, the remaining recipes follow.

## Pipeline fixes (affect all scores; old runs are re-scored from raw text)
- The JSON parser takes the last valid array, recovers bare objects without brackets, and strips think blocks.
- The name normalizer handles CamelCase, possessives and underscores.
- Unmapped answers count toward diversity, keyed by their normalized text.
- vLLM sampling stops at the chat end-of-turn token (needed for base models).
- vLLM uses its native sampler, because the FlashInfer JIT needs a CUDA toolkit matching torch 13.0.
