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
- **E7 v2 pilot results (seed 1, 2026-10-19).**
  - **Stable.** No collapse: the monitor stayed around 1, entropy and length were flat, and KL stayed ≤0.03.
  - **DDXPlus.** Both recipes improve on the untrained model (pass@1 0.378, pass@32 0.435) by the same amount:
    | Run | pass@1 | pass@32 | Δ pass@1 (95% CI) | Δ pass@32 |
    |---|---|---|---|---|
    | acc-s1 | 0.463 | 0.502 | +8.5 (+7.1, +10.1) | +6.7 |
    | dist-s1 | 0.462 | 0.502 | +8.4 | +6.7 |
  - **The two recipes are indistinguishable.** Distinct answers stay at 1.36 (untrained 1.41), entropy at 0.13–0.14, coverage@32 at 0.18 (untrained 0.16), and ECE at about 0.075.
  - **Phenopackets:** no change, at the accuracy floor.
  - **Interpretation.** These rewards score each sample's own stated list. The collapse is *across* samples (the same top-1 answer in every sample), which a per-sample reward cannot see.
- **Group-level reward `group_match`** (`rewards.py`, added 2026-10-19).
  - Within a GRPO group of n samples for one case, sample i with top-1 answer a gets r_i = clip(n·p_ref(a) − (k_a − 1), 0, 1). Here k_a is the number of samples in the group whose top answer is a.
  - This is the sample's marginal contribution to n·Σ_c min(q_c, p_c), so it targets the *sampled* answer distribution: the open gap in the novelty check, distinct from Puri et al.'s within-output sets.
  - Unit-tested (collapsed group gets 0; a group matching the reference gets 1 each; over-represented answers get 0, under-represented ones 1). CPU smoke test passed with complete groups.
  - **Runs:** gm-s0 and gm-s1 (`acc,group_match`), queued 2026-10-19 with stable settings and chained evaluation.
  - **Early curves for gm-s0 (step 73): the group reward is sparse.**
    - Its mean is 0.00–0.07. The policy gives about 1 distinct top-1 answer per case across 16 samples (training entropy about 0.05), so the modal answer always exceeds its reference capacity and earns 0. Only rare dissenting samples earn credit.
    - The bottleneck is exploration rather than reward shape.
  - **Boosted variant gm3t12-s0**, queued on ckpt-all: weights format/acc/gm = 1/1/3, training sampling temperature 1.2, seed 0, other stable settings unchanged, chained evaluation.

- **E7 v2 full results, DDXPlus (Qwen3-1.7B, 980 × 32, T=0.7; 2 seeds each; `src/compare_recipes.py` paired bootstrap).**
  | Recipe | pass@1 | pass@32 | Distinct | Entropy | Cov@32 | TV |
  |---|---|---|---|---|---|---|
  | untrained | 0.378 | 0.435 | 1.41 | 0.151 | 0.162 | 0.850 |
  | acc | 0.458 | 0.499 | 1.36 | 0.142 | 0.181 | 0.829 |
  | dist | 0.458 | 0.502 | 1.34 | 0.131 | 0.181 | 0.829 |
  | gm | 0.469 | 0.510 | 1.40 | 0.155 | 0.186 | 0.825 |
  | **gm3t12** (gm weight 3, train T=1.2) | **0.474** | **0.516** | **1.46** | **0.176** | **0.196** | **0.815** |
  - **gm vs acc:** pass@1 +1.0 (+0.6, +1.6), distinct +0.04, entropy +0.013, cov@32 +0.005, TV −0.004. All CIs exclude 0.
  - **gm3t12 vs acc:** pass@1 +1.6 (+1.0, +2.3), pass@32 +1.7, distinct +0.10 (+0.07, +0.13), entropy +0.034 (+0.022, +0.047), cov@32 +0.015, TV −0.014. All CIs exclude 0.
  - **gm3t12 vs gm:** diversity is higher (distinct +0.056, entropy +0.021, TV −0.010, all significant). pass@1 +0.6, not significant.
  - **Reading:**
    - Per-sample rewards (acc, dist) sharpen the model slightly.
    - The group reward removes that sharpening; boosted, it *raises* diversity above the starting model while gaining about 9.6 pass@1.
    - Diversity increases with group-reward pressure (dose-response), with no accuracy cost.
    - Absolute effects are modest because the starting policy is already collapsed.
  - **Phenopackets:** no change for any recipe. Accuracy is at the floor; confidence is 0.43–0.47 (untrained 0.45).
- **E9: GRPO from Qwen3-8B-Base** (diverse start), launched 2026-10-20 on Tillicum: `b8-acc-s0` and `b8-gm-s0` (plain gm).
  - Runtime about 320 s/step on an H200, so 200 steps take about 18 h (20 h limit, about $17 per run).
  - Evaluated on full H200s.
  - Comparison: base vs acc-RL vs gm-RL vs official Qwen3-8B.
  - A first attempt failed: two jobs on one node clashed on MASTER_PORT 29500. Fixed with a per-job port in `slurm/tillicum/train.sbatch`.
  - Qwen3-1.7B-Base was rejected as a starting point: its format validity was about 8%.

- **E9 results: GRPO from Qwen3-8B-Base (DDXPlus, 980 × 32, T=0.7, seed 0; scored on Klone `runs/e9_ddx`).**
  | Model | pass@1 | pass@32 | Distinct | Entropy | Cov@32 | TV |
  |---|---|---|---|---|---|---|
  | Qwen3-8B-Base (start) | 0.389 | 0.732 | 3.77 | 0.78 | 0.323 | 0.749 |
  | b8-acc-s0 | 0.536 | 0.757 | 2.72 | 0.52 | 0.296 | 0.750 |
  | b8-gm-s0 | 0.538 | 0.770 | 2.87 | 0.56 | 0.308 | 0.741 |
  | Qwen3-8B (official) | 0.558 | 0.607 | 1.27 | 0.11 | 0.217 | 0.793 |
  - **Both RL runs vs official Qwen3-8B:** pass@1 −2.1 / −2.3 (CIs about −4 to −0.1), pass@32 +16.3 (+14.0, +18.8) for gm and +15.0 for acc, distinct +1.5 to +1.6.
  - **gm vs acc:** pass@1 +0.2 (not significant), pass@32 +1.3 (+0.4, +2.2), distinct +0.14 (+0.10, +0.19), entropy +0.041 (+0.033, +0.049), cov@32 +1.3 (+0.8, +1.8), TV −0.009.
  - Both runs narrow the base model (distinct 2.7–2.9 vs 3.8); gm narrows it less.
  - Training took about 5.5 h per run on an H200, about 100 s/step after warm-up.
- **E9 seed 1 and Puri baselines (finished on Tillicum overnight 2026-10-20/21; scored on Klone 2026-10-21 with `slurm/score_new.sbatch`, compared with `slurm/compare_seed1.sh`).**
  - **8B, per seed (DDXPlus):**
    | Run | pass@1 | pass@32 | Distinct | Entropy | Cov@32 | TV |
    |---|---|---|---|---|---|---|
    | b8-acc-s0 | 0.536 | 0.757 | 2.72 | 0.52 | 0.296 | 0.750 |
    | b8-acc-s1 | 0.538 | 0.760 | 2.71 | 0.52 | 0.298 | 0.749 |
    | b8-gm-s0 | 0.538 | 0.770 | 2.87 | 0.56 | 0.308 | 0.741 |
    | b8-gm-s1 | 0.541 | 0.770 | 2.84 | 0.55 | 0.307 | 0.742 |
  - **gm vs acc, 2 seeds:** pass@1 +0.2 (n.s.), pass@32 +1.2 (+0.5, +1.9), distinct +0.14 (+0.11, +0.17), entropy +0.035 (+0.029, +0.041), cov@32 +1.1 (+0.8, +1.5), TV −0.008. Each seed shows the same ordering.
  - **vs official Qwen3-8B, 2 seeds:** gm pass@1 −1.9 (−3.9, +0.1), pass@32 +16.3 (+14.0, +18.8); acc pass@1 −2.1 (−4.1, −0.1), pass@32 +15.2.
  - **Phenopackets:** both 8B recipes stay indistinguishable from base. gm has +0.21 (+0.09, +0.32) distinct answers vs acc.
  - **Puri baselines at 1.7B (2 seeds each, DDXPlus):**
    | Run | pass@1 | pass@32 | Distinct | Entropy | Cov@32 | List recall |
    |---|---|---|---|---|---|---|
    | Multi-RLVR | 0.385 | 0.443 | 1.39 | 0.153 | 0.166 | 0.317 |
    | Multi-RLCR | 0.383 | 0.441 | 1.39 | 0.150 | 0.165 | 0.301 |
    | acc | 0.458 | 0.499 | 1.36 | 0.142 | 0.181 | 0.300 |
    | gm3t12 | 0.474 | 0.516 | 1.46 | 0.176 | 0.196 | 0.309 |
    - Training was healthy: multi_rlvr reward 0.27–0.31 → 0.32–0.35, format 0.85 → 0.98, no collapse.
    - The reward targets reference-set membership, not the gold answer, so pass@1 barely moves (+0.5 to +0.8 vs untrained). Only Multi-RLVR improves list recall (reference mass covered by one completion's list).
    - gm3t12 vs Multi-RLVR: pass@1 +8.9 (+7.4, +10.5), distinct +0.07 (+0.03, +0.11), cov@32 +3.0 (+2.3, +3.8). Multi-RLCR is similar.
  - **Pitfall:** `sbatch --export=ALL,SPEC=a,b` splits on commas, so the first scoring pass only scored the first tag in each list. Tags are now joined with `+`.
  - Tillicum spend after these runs: $40.79 / $250.
- **OLMo 3 7B post-training stages (IN PROGRESS, launched 2026-10-23 ~10:06 on Tillicum, `submit_llama.sh olmo`).** This replaces the Llama pair: Meta access was still pending after more than 2 days, and OLMo is ungated.
  - **Checkpoints:** `Olmo-3-1025-7B` (base; uses `templates/olmo3_sft.jinja` via `--chat-template`), `-Instruct-SFT`, `-Instruct-DPO`, `-Instruct` (RLVR), `-RL-Zero-General` (RL from base).
  - **Output dirs:** DDXPlus T=0.7 → `meas_ddx/`, T=1 → `meas_ddx_t1/`, full 1,401 Phenopackets → `meas_full/`.
  - **Status at 12:13:** SFT, DPO, Instruct and RL-Zero-General are complete (980/980/1401). Base is partial (512, 640, 512) and may need resubmission after the 4 h limit; sampling is resumable.
  - **RL-Zero-Mix failed:** model type `olmo2-retrofit` isn't supported by our Transformers/vLLM. Skipped.
  - **Smoke test:** vLLM resolves `Olmo3ForCausalLM`. Instruct output is clean JSON; base output is often invalid (refusals, plain lists, rambling).
  - **Next:** relay to Klone `runs/{ddx_olmo,ddx_olmo_t1,go_full_olmo}`, then run `slurm/olmo_klone.sh` (scoring, format robustness, stage-to-stage comparisons, severity).
  - **Spend:** $65.68 at 12:13, up from $50.78 before OLMo.
- **Full Phenopacket eval set: 1,401 cases / 749 diseases (2026-10-21).** Tillicum ran the remaining 901 cases (`submit_llama.sh fullpheno`, `splits/eval_rest.txt`, $3.17) with the same settings as the go set. Merged with the go-set samples in `runs/go_full` by `runs/fullpheno_klone.sh`; reports are in `go_full{,_pair25,_q3,_base_rl}`.
  | Model | pass@1 | pass@32 | Distinct | Conf | ECE |
  |---|---|---|---|---|---|
  | Qwen2.5-7B base | 0.014 | 0.057 | 15.59 | 0.61 | 0.59 |
  | Qwen2.5-7B-Instruct | 0.016 | 0.056 | 11.46 | 0.51 | 0.49 |
  | AlphaMed | 0.021 | 0.066 | 11.14 | 0.69 | 0.67 |
  | AlphaMed base-RL | 0.017 | 0.064 | 13.23 | 0.71 | 0.70 |
  | HuatuoGPT-o1 | 0.022 | 0.062 | 9.24 | 0.67 | 0.64 |
  | m1 (8k) | 0.018 | 0.070 | 15.26 | 0.57 | 0.55 |
  | Qwen3-8B-Base | 0.015 | 0.059 | 13.37 | 0.61 | 0.59 |
  | Qwen3-8B | 0.022 | 0.037 | 6.36 | 0.46 | 0.44 |
  - **vs Instruct:** AlphaMed is significant at every k (+0.6 at k=1, +1.0 at k=32). HuatuoGPT-o1 is significant for k≤8. m1 is significant for k≥2 (+1.4 at k=32).
  - **Qwen3-8B vs base:** +0.7 (+0.4, +1.1) at k=1 and −2.2 (−3.1, −1.3) at k=32. The crossing is now significant on rare disease as well; on 500 cases it was n.s. The Qwen2.5 pair is n.s. everywhere.
  - **Era, pass@32:** ≤2022 (n=901) 5.2–9.5%; 2023–24 (n=348) 0.3–1.7%; 2025+ (n=152) 2.6–4.6%.
- **Prompt robustness and output format (2026-10-21).** Ran on Tillicum with `submit_llama.sh prompts` ($3.89). The flag is `--prompt-variant plain|cot` in `sample_diagnoses.py`; the default prompt is unchanged. Scored in `runs/pv_{plain,cot}_{t07,t1}`.
  | Prompt | T | Δpass@1 | Δpass@32 | Distinct base / post |
  |---|---|---|---|---|
  | default | 0.7 | +17.0 | −12.2 | 3.76 / 1.27 |
  | default | 1.0 | +27.5 | −14.6 | 6.60 / 1.53 |
  | plain | 0.7 | +23.0 | −7.1 (−9.3, −5.0) | 3.72 / 1.28 |
  | plain | 1.0 | +34.2 | −8.7 (−11.0, −6.4) | 5.57 / 1.54 |
  | cot | 0.7 | +24.2 | **+4.7** (+2.9, +6.6) | 3.89 / 2.08 |
  | cot | 1.0 | +36.7 | 0.0 (−2.1, +1.9) | 6.31 / 2.24 |
  - **CoT finding:** with reasoning, the instruct model recovers pass@32 (0.607 to 0.835), but diversity stays low. The paper now says the collapse is largest for direct answers.
  - **Format:** invalid rates (unparsable or off-list top-1). Qwen3-8B-Base: 11.6% (T=0.7) and 24.6% (T=1). Qwen2.5 base: 1.3% and 21.0%. Post-trained: ≤0.27%. `score_ddxplus.py` now writes `n_parsed`.
  - `src/format_robust.py`: Δpass@1 among valid samples only is +12.0 / +18.7 (Qwen3 T=0.7 / T=1) and +12.2 / +16.8 (Qwen2.5).
    - Cases where all samples of both models are valid: Qwen3 T=0.7, 105 cases, Δpass@32 −15.2 (−21.9, −8.6). Qwen2.5 T=0.7, 748 cases, −10.2 (−12.7, −7.6).
    - At T=1 almost no case has 32 valid base samples.
  - **Parser fix:** deeply nested brackets in one base sample caused a RecursionError and crashed shard 0 of `pv_plain_t1` base. `parsing.py` now catches it, and the shard was rerun.
- **Second family and medical RL from base (Tillicum 2026-10-21, `slurm/tillicum/submit_llama.sh descendants`, $2.63; scored on Klone in `runs/{ddx,go}_{base_rl,llama}`).**
  - **AlphaMed-7B-base-rl** (AlphaMed RL applied directly to Qwen2.5-7B base), DDXPlus T=0.7:
    | Model | pass@1 | pass@32 | Distinct | Entropy | Cov@32 | ECE |
    |---|---|---|---|---|---|---|
    | Qwen2.5-7B base | 0.352 | 0.695 | 4.91 | 1.00 | 0.335 | 0.29 |
    | Qwen2.5-7B-Instruct | 0.478 | 0.598 | 1.82 | 0.30 | 0.239 | 0.11 |
    | AlphaMed (from Instruct) | 0.516 | 0.682 | 1.87 | 0.28 | 0.266 | 0.12 |
    | AlphaMed (from base) | 0.495 | 0.790 | 3.14 | 0.62 | 0.320 | 0.20 |
    - vs base: pass@1 +14.3 (+13.1, +15.7), pass@32 +9.5 (+7.4, +11.6), distinct −1.77.
    - vs AlphaMed-from-Instruct: pass@1 −2.2 (−3.9, −0.4), pass@32 +10.8 (+8.6, +13.2), distinct +1.28.
    - Severe-tier Considered@32 vs base: base-RL −0.7 (−3.0, +1.7), Instruct −12.4, AlphaMed-from-Instruct −8.1 (`runs/analysis/pc_base_rl`).
    - **Takeaway:** a third-party model reproduces our GRPO-from-base result. RL from base does not collapse; the collapse comes from instruction tuning.
    - **Provenance (checked 2026-10-22):** arXiv 2505.17952 has only v1, which describes only instruct-initialized models. Training: GRPO, binary boxed-answer reward, 19,178 MedQA + MedMCQA QA pairs, batch 64×8, 300 steps, verl; lr and KL not reported. AlphaMed-7B (Qwen2.5-7B-Instruct) is in Appendix A.3.
      - The base-rl checkpoint is not mentioned in the paper. Its `config.json` `_name_or_path` is `.../Qwen2.5-7B`; the instruct-rl one is `.../Qwen2.5-7B-Instruct`. Both were uploaded the same day with the same model card.
      - The paper therefore treats it as corroborating, not controlled, evidence.
    - Phenopackets: pass@k vs base n.s.; conf 0.71, ECE 0.70, acc 1.4%. It is the most overconfident model we evaluated.
  - **Llama-3.1-8B-Instruct medical descendants**, DDXPlus T=0.7. Parent pending Meta gate approval.
    | Model | pass@1 | pass@32 | Distinct | Entropy | ECE |
    |---|---|---|---|---|---|
    | HuatuoGPT-o1-8B | 0.554 | 0.810 | 2.84 | 0.53 | 0.02 |
    | AlphaMed-8B | 0.495 | 0.791 | 2.78 | 0.53 | 0.20 |
    | UltraMedical-8B | 0.484 | 0.702 | 2.55 | 0.49 | 0.11 |
    - Phenopackets: pass@32 9.4–9.8% (UltraMedical 6.6%); conf 0.58–0.64; acc about 3%.
  - **Pitfall:** `score_new.sbatch` chose the scorer by a `*_ddx` suffix, so `ddx_*` dirs went to the Phenopacket scorer. It now matches `*ddx*`.
- **Measurement runs completed on Tillicum (2026-10-20), scored on Klone.**
  - **Medical family on DDXPlus** (T=0.7, against Qwen2.5-7B-Instruct at 0.478 / 0.598, 1.82 distinct):
    | Model | pass@1 | pass@32 | Distinct | ECE |
    |---|---|---|---|---|
    | AlphaMed | 0.516 | 0.682 | 1.87 | 0.12 |
    | HuatuoGPT-o1 | 0.536 (+5.8) | 0.735 (+13.7) | 2.49 | 0.17 |
    | m1 (8k tokens) | 0.486 (+0.9, n.s.) | 0.779 (+18.1) | 3.29 | 0.21 |

    The reasoning-trained models widen the differential.
  - **Full Qwen2.5-7B base vs Instruct pair** (980 cases):
    - T=0.7: Instruct +12.6 at k=1, base +9.7 at k=32.
    - T=1.0: Instruct +22.5 at k=1, base +10.3 at k=32.
  - **AlphaMed at T=1.0:** better than its parent at every k (+5.0 to +5.7).
  - **Phenopacket results (500 cases, 32 samples, T=0.7; scored on Klone job 42097417, 2026-10-21).** Deltas are vs the first-listed reference with 95% case-bootstrap CIs.
    | Run | Model | pass@1 | pass@32 | Distinct | Conf | ECE | Unparsed / 16000 |
    |---|---|---|---|---|---|---|---|
    | go | Qwen2.5-7B-Instruct (ref) | 0.013 | 0.062 | 11.50 | 0.51 | 0.50 | 9 |
    | go | AlphaMed | 0.018 (n.s.) | 0.072 (n.s.) | 11.20 | 0.69 | 0.67 | 30 |
    | go | HuatuoGPT-o1 | 0.019 (+0.6, sig) | 0.060 (n.s.) | 9.46 | 0.66 | 0.64 | 70 |
    | go | m1 (8k) | 0.019 (+0.6 [+0.2,+1.1]) | 0.082 (+2.0 [+0.2,+3.8]) | 15.36 | 0.56 | 0.54 | 72 |
    | pheno25 | Qwen2.5-7B base (ref) | 0.011 | 0.054 | 15.76 | 0.60 | 0.59 | 53 |
    | pheno25 | Qwen2.5-7B-Instruct | 0.013 (n.s.) | 0.062 (n.s.) | 11.50 | 0.51 | 0.50 | 9 |
    | go_qwen3 | Qwen3-8B-Base (ref) | 0.015 | 0.054 | 13.51 | 0.61 | 0.59 | 1359 |
    | go_qwen3 | Qwen3-8B | 0.022 (+0.7 [+0.2,+1.3]) | 0.040 (-1.4 [-3.0,+0.2]) | 6.22 | 0.46 | 0.44 | 0 |
    | e9_go | b8-acc-s0 | 0.015 (n.s.) | 0.048 (n.s.) | 12.43 | 0.62 | 0.61 | 354 |
    | e9_go | b8-gm-s0 | 0.014 (n.s.) | 0.048 (n.s.) | 12.65 | 0.61 | 0.60 | 392 |
    - m1 (SFT-only) is the only model that beats its parent at every k, and it is the most diverse. The RL-trained descendants (AlphaMed, HuatuoGPT-o1) raise confidence the most.
    - General post-training lowers confidence on rare disease (Qwen3 0.61 to 0.46, Qwen2.5 0.60 to 0.51). Medical RL raises it again.
    - The 8B GRPO runs are indistinguishable from their base.
- **Publication-era check (Phenopackets, Qwen2.5 family incl. m1 8k):** pass@32 is 8.1–10.8% for cases published ≤2022 (n=334), 0.9–2.6% for 2023–24 (n=115) and 3.9% for 2025+ (n=51), which points to memorization (`src/era_table.py`).

## Tillicum (UW H200 cluster; RCC student credit $250, expires 2027-04-19)
- **Paths.** `/gpfs/projects/stf/jairamh/thesis` holds code, env, data and runs (persistent). `/gpfs/scrubbed/jairamh/thesis` holds caches. `~/.thesis_env` sets these.
- **Environment** (`slurm/tillicum/build_env.sbatch`): torch 2.13 cu130, **vLLM 0.31.0 (pinned to match Klone)**, TRL 1.16. Needs `VLLM_WORKER_MULTIPROC_METHOD=spawn`, set in the Tillicum job scripts.
- **Job scripts** (`slurm/tillicum/`):
  - `train.sbatch`: full H200, 8 CPUs, 128 GB.
  - `sample.sbatch`: MIG 1g.18gb slice, 2 CPUs per GPU maximum, about 1/7 the cost.
  - `submit_train_eval.sh`: train, then DDXPlus and Phenopackets evaluation.
  - Outputs go to `runs/eval_ddx/` and `runs/eval_go/`.
- **QOS.** `debug` (1 h, usage factor 0) for builds and dry runs; `normal` for runs (1 day maximum).
- **Speed.** GRPO on Qwen3-1.7B runs at about 25 s/step on an H200, against about 150 s/step on a Klone A40. A 200-step run takes about 85 min, roughly $1.30.
- **Budget log** (from `hyakusage`):
  | Date | Item | Cumulative |
  |---|---|---|
  | 2026-10-20 | env build attempts, vLLM smoke test, GRPO dry run | $0.11 |
  | 2026-10-20 | gm3t12-s0/s1 training + evals; Klone-run evals on MIG/H200; 8B dry runs; first (failed/cancelled) 8B attempt | $4.54 |
  | 2026-10-20 | launched b8-acc-s0, b8-gm-s0 (8B, 20 h limit; expected about $36–40 incl. eval) | |

## Pipeline fixes (affect all scores; old runs are re-scored from raw text)
- The JSON parser takes the last valid array, recovers bare objects without brackets, and strips think blocks.
- The name normalizer handles CamelCase, possessives and underscores.
- Unmapped answers count toward diversity, keyed by their normalized text.
- vLLM sampling stops at the chat end-of-turn token (needed for base models).
- vLLM uses its native sampler, because the FlashInfer JIT needs a CUDA toolkit matching torch 13.0.
