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
  - **Status at 14:45:** DDXPlus complete for all five stages (980 cases each, both temperatures). Base T=0.7 finished just inside the 4 h limit; its resume (383707) exited after 5 s. Phenopacket base was cut at 1024/1401 and is resuming (383706, 5 h limit).
  - **RL-Zero-Mix failed:** model type `olmo2-retrofit` isn't supported by our Transformers/vLLM. Skipped.
  - **Smoke test:** vLLM resolves `Olmo3ForCausalLM`. Instruct output is clean JSON; base output is often invalid (refusals, plain lists, rambling).
  - **DDXPlus results (Klone `runs/ddx_olmo{,_t1}`, `slurm/olmo_ddx_analysis.sh`, job 42217232).** Δ = vs base, points, 95% paired case bootstrap.

    | T=0.7 | pass@1 | pass@32 | Δ pass@32 | Distinct | cov@32 | Severe Considered@32 (Δ) |
    |---|---|---|---|---|---|---|
    | Base | 0.178 | 0.705 | | 4.93 | 0.351 | 0.412 |
    | SFT | 0.222 | 0.559 | −14.6 [−17.5, −11.7] | 4.01 | 0.257 | −3.9 [−6.5, −1.3] |
    | DPO | 0.324 | 0.613 | −9.2 [−12.1, −6.1] | 3.30 | 0.272 | −7.4 [−10.0, −4.9] |
    | Instruct (RLVR) | 0.266 | 0.422 | −28.3 [−31.5, −25.0] | 2.37 | 0.195 | −16.1 [−18.6, −13.5] |
    | RL-Zero-General | 0.295 | 0.572 | −13.3 [−16.1, −10.5] | 4.06 | 0.286 | −8.5 [−10.9, −6.2] |

    | T=1 | pass@1 | pass@32 | Δ pass@32 | Distinct | cov@32 | Severe Considered@32 (Δ) |
    |---|---|---|---|---|---|---|
    | Base | 0.128 | 0.713 | | 7.53 | 0.398 | 0.469 |
    | SFT | 0.191 | 0.634 | −8.0 [−11.0, −4.9] | 6.32 | 0.321 | −4.3 [−7.4, −1.3] |
    | DPO | 0.313 | 0.688 | −2.5 [−5.3, +0.4] | 4.48 | 0.312 | −5.9 [−8.9, −3.1] |
    | Instruct (RLVR) | 0.272 | 0.561 | −15.2 [−18.3, −12.1] | 3.58 | 0.259 | −14.4 [−17.5, −11.4] |
    | RL-Zero-General | 0.279 | 0.695 | −1.8 [−4.5, +0.8] | 6.28 | 0.367 | −4.0 [−6.9, −1.2] |

    - **Stage to stage (T=0.7):** SFT vs base pass@32 −14.6, distinct −0.92. DPO vs SFT pass@32 +5.4, distinct −0.71. **Instruct (RLVR) vs DPO: pass@1 −5.9, pass@32 −19.1, distinct −0.92, cov@32 −7.7** (all significant). The T=1 pattern is the same (Instruct vs DPO pass@32 −12.7).
    - **Format:** base invalid rate 39.7% at T=0.7 and 47.2% at T=1 (SFT 16% / 21%). No case has all 32 base samples valid, so the fully-parsed subset check is empty. Valid-only pass@1 vs base: SFT −1.7, DPO +4.0, Instruct −1.9, RL-Zero +2.0 at T=0.7 (T=1: +1.3, +8.1, +3.8, +5.8). **So OLMo's raw pass@1 gains are mostly format.** Invalid samples count as misses for pass@k and are excluded from distinct/entropy, so base coverage and diversity are understated and the large-k losses are conservative.
    - **Phenopackets (Klone `runs/go_full_olmo`, 1,401 cases, T=0.7; base finished via resume 383706, scored by job 42221122):**

      | Stage | pass@1 | pass@32 | Δ pass@32 vs base | Distinct | Conf | ECE | Unparsable |
      |---|---|---|---|---|---|---|---|
      | Base | 0.007 | 0.045 | | 12.66 | 0.74 | 0.73 | 36% |
      | SFT | 0.006 | 0.025 | −2.0 [−3.0, −1.1] | 16.60 | 0.84 | 0.83 | 3.6% |
      | DPO | 0.007 | 0.030 | −1.5 [−2.5, −0.5] | 18.81 | 0.73 | 0.72 | 0.2% |
      | Instruct (RLVR) | 0.008 | 0.031 | −1.4 [−2.4, −0.5] | 16.03 | 0.76 | 0.76 | 0.4% |
      | RL-Zero-General | 0.009 | 0.039 | −0.6 [−1.4, +0.2] | 14.51 | 0.74 | 0.73 | 0.9% |

      - Everything is near the floor, as for Qwen (pass@32 3.7–7.0%); OLMo is lower. Every instruct-pipeline stage loses at k=32 (−1.4 to −2.0, significant), and RL-Zero does not.
      - Distinct answers are higher after post-training here, but base distinct is counted over its valid samples only (36% unparsable), so this does not show post-trained models are more diverse.
      - Stated confidence is 0.73–0.84 for every stage, at about 1% accuracy. SFT is the most overconfident (ECE 0.83). Unlike Qwen, general post-training does not lower confidence.
    - **Reading:** the largest single narrowing is the RLVR step after SFT+DPO, which also lowers pass@1. RL-Zero from base keeps base-level pass@32 at T=1 (not at T=0.7) and loses the least on the mild and moderate tiers.
  - **Spend:** $65.68 at 12:13, up from $50.78 before OLMo.
- **Stated list vs sampled top-1 (2026-10-23, `src/stated_vs_sampled.py`, `slurm/stated_vs_sampled.sh`, job 42218824, CPU only).** Question: does post-training collapse the differential a model *states* (one sample's top-5 list with probabilities), or only the distribution of its *sampled* top-1 answers? Valid samples only; results in Klone `runs/analysis/svs/*.json`.
  - **Note:** `list_recall` in `compare_runs.py` reports is gold-in-list rate, not reference-mass coverage. Do not quote it as differential coverage.

  | Pair | TV one list (base → post) | TV sampled top-1 (base → post) | Ref. mass in one list | Ref. mass, top-1 of 5 samples | Severe named in one list |
  |---|---|---|---|---|---|
  | Qwen3-8B, T=0.7 | 0.743 → **0.700** | 0.749 → 0.793 | 0.335 → **0.396** | 0.245 → 0.207 | 0.347 → **0.437** |
  | Qwen3-8B, T=1 | 0.772 → **0.700** | 0.712 → 0.783 | 0.293 → **0.396** | 0.275 → 0.215 | 0.312 → **0.440** |
  | Qwen2.5-7B, T=0.7 | 0.779 → **0.716** | 0.756 → 0.785 | 0.291 → **0.364** | 0.233 → 0.214 | 0.345 → **0.440** |
  | Qwen2.5-7B, T=1 | 0.803 → **0.717** | 0.715 → 0.761 | 0.255 → **0.363** | 0.257 → 0.234 | 0.304 → **0.438** |
  | Qwen3-8B CoT, T=0.7 | 0.726 → **0.684** | 0.733 → 0.735 | 0.351 → **0.409** | 0.263 → 0.261 | 0.365 → **0.446** |
  | OLMo base → Instruct, T=0.7 | 0.795 → 0.823 (worse) | 0.757 → 0.851 | 0.268 → 0.257 (worse) | 0.237 → 0.150 | 0.256 → **0.317** |
  | OLMo base → Instruct, T=1 | 0.822 → 0.817 (n.s.) | 0.733 → 0.824 | 0.227 → **0.262** | 0.247 → 0.173 | 0.245 → **0.324** |
  | OLMo base → DPO, T=0.7 | 0.795 → 0.795 (n.s.) | 0.757 → 0.804 | 0.268 → 0.277 | 0.237 → 0.193 | 0.256 → **0.378** |
  | Oracle (reference itself) | 0.265 (reference top 5) | 0.18–0.25 (draws from reference) | 0.735 | 0.56 | 0.975 |

  - **Qwen (all four pair × temperature settings and CoT):** the two views disagree in sign, and all differences are significant (paired case bootstrap). The post-trained model's single list is closer to the reference (TV −0.04 to −0.09), covers more reference mass (+0.06 to +0.11) and names severe conditions more often (+0.08 to +0.13). Its sampled top-1 distribution is further from the reference (+0.03 to +0.07; CoT: no difference), and 5 samples cover less (−0.02 to −0.06). Within the post-trained model the list beats its own samples by 0.04–0.09 TV; for base models the gap is about zero or negative.
  - **OLMo:** the reversal does not hold cleanly. DPO and Instruct lists are no closer to the reference than the base model's (Instruct at T=0.7 is slightly worse). Only severe-condition naming improves (+0.06 to +0.13), plus RL-Zero's lists (TV −0.02 to −0.04).
  - **Medical (vs Qwen2.5-7B-Instruct):** AlphaMed and HuatuoGPT-o1 lists slightly better (TV −0.01). m1 lists worse (+0.017) while its sampling is better (−0.038).
  - **Absolute level:** every model is far from the reference. Single lists cover 0.22–0.41 of reference mass, against 0.735 for the reference's own top 5.
  - **Reading:** for Qwen, post-training improves the stated differential and collapses the sampled one, so pass@k and sample diversity report a collapse the stated list does not show. The paper's "differential collapse" wording overstates the effect for Qwen. Close prior work: Verbalized Sampling (2510.01171) shows that aligned models recover diversity when asked to verbalize a distribution.
- **Consequences: aggregation and uncertainty (2026-10-23, `src/consequences.py`, `slurm/consequences.sh`, job 42219843, CPU only).** The single-call answer is the top-1 of the first valid sample. Klone `runs/analysis/cons/*.json`.
  - **Self-consistency (vote@32 minus single-call accuracy, points):** base models gain a lot: Qwen3 +3.4 / +10.4 (T=0.7 / T=1), Qwen2.5 +5.2 / +14.1, OLMo +4.0 / +14.2. Direct-answer post-trained models gain little: Qwen3-8B +0.3 (n.s.) / +1.4, Qwen2.5-Instruct +0.8 (n.s.) / +4.4, OLMo DPO and Instruct about 0. Reasoning-style models still gain: HuatuoGPT-o1 +4.8, m1 +3.8, Qwen3-8B with CoT +4.5.
  - **Averaging stated lists instead of voting:** no accuracy difference anywhere (within ±2 points).
  - **Top-3 from one stated list vs the 3 most frequent of 32 sampled answers:** post-trained Qwen3-8B +13.7 [+11.4, +15.9] / +10.0; Qwen2.5-Instruct +7.3 / +2.0 (n.s.); OLMo Instruct +1.9 (n.s.) / −4.7; OLMo DPO −7.8 / −9.6; base models −7 to −19. So for post-trained Qwen, one call's list beats 32 samples; for base models, sampling wins.
  - **Uncertainty: AUROC for the single-call answer being correct.** Signals: stated probability of the answer (no extra calls) vs agreement of other samples (4 or 31 extra calls).

    | Model | Stated p | Agreement, 5 samples | Agreement, 32 | Stated − agreement@32 |
    |---|---|---|---|---|
    | Qwen3-8B-Base, T=0.7 | 0.595 | 0.772 | 0.810 | −0.216 [−0.257, −0.173] |
    | Qwen3-8B, T=0.7 | **0.734** | 0.613 | 0.655 | **+0.079 [+0.046, +0.110]** |
    | Qwen3-8B, T=1 | **0.726** | 0.658 | 0.701 | +0.025 (n.s.) |
    | Qwen2.5-7B base, T=0.7 | 0.576 | 0.764 | 0.811 | −0.235 |
    | Qwen2.5-7B-Instruct, T=0.7 | 0.571 | 0.653 | **0.702** | −0.131 [−0.174, −0.087] |
    | OLMo base, T=0.7 | 0.538 | 0.761 | 0.802 | −0.264 |
    | OLMo DPO, T=0.7 | 0.738 | 0.702 | 0.751 | −0.013 (n.s.) |
    | OLMo Instruct, T=0.7 | **0.796** | 0.668 | 0.731 | **+0.064 [+0.027, +0.101]** |
    | OLMo Instruct, T=1 | 0.769 | 0.724 | 0.780 | −0.011 (n.s.) |
    | HuatuoGPT-o1 / m1 / AlphaMed | 0.59 / 0.68 / 0.67 | 0.72 / 0.74 / 0.66 | 0.78 / 0.78 / 0.72 | agreement wins |

    - In every family, post-training lowers the agreement signal (Qwen3 0.81 → 0.66, Qwen2.5 0.81 → 0.70, OLMo 0.80 → 0.73) and moves the best signal toward the stated probability (by +0.10 to +0.33 AUROC). The stated probability overtakes agreement only for Qwen3-8B and OLMo Instruct at T=0.7.
    - AUROCs are for each model's own answers and accuracy, so they compare within a model, not across models.
  - **Does uncertainty track true ambiguity? (Spearman correlation with the entropy of the reference differential):** almost not at all for any signal or model (−0.10 to +0.13). The exception is AlphaMed's stated probability (+0.30). Neither readout tells ambiguous cases from clear ones.
  - **What predicts the crossover, and a small audit (2026-10-23, `src/audit.py`, `slurm/audit.sh`, Klone job 42243519, CPU only; `runs/analysis/audit.json`).** 44 scored DDXPlus models: Qwen pairs at both temperatures and three prompts, medical models (Qwen and Llama parents), OLMo stages at both temperatures, and our GRPO runs (1.7B, 8B).
    - Across models, the gap (AUROC of stated probability minus agreement) correlates with the single-answer rate, the share of cases whose samples all give the same top-1 (Spearman 0.76 with 5 samples, 0.77 with 32), with distinct answers (−0.78), and with the stated probability's own AUROC (0.83). Stated wins in 16 of 44 models, all with a single-answer rate of at least 0.30 at 5 samples. No single-answer threshold separates them cleanly: Qwen2.5-7B-Instruct (0.67) and Qwen3-8B with CoT (0.65) still favor agreement, while OLMo DPO and Instruct favor the stated probability at 0.30–0.58 because their stated probability is unusually informative (AUROC 0.72–0.80).
    - **Audit:** using only 5 samples per case on n random cases, pick the signal with the higher AUROC. With n=100 the pick matches the full-data pick (980 cases, same 5-sample budget) in 90% of draws on average (min 52%, where the two signals nearly tie); the mean AUROC lost is 0.004 (n=50: 85%, 0.008; n=200: 93%, 0.002).
    - Our 8B GRPO models keep agreement informative (0.80–0.81) while their stated probability is weak (0.53–0.57); the 1.7B runs, starting from a collapsed instruct model, favor the stated probability.
  - **32B replication (Tillicum `submit_llama.sh big32`, 2026-10-23/24, T=0.7, 980 cases, 4 shards; scored on Klone `runs/big32_ddx` by `slurm/big32_full49.sh`, job 42260193).** Qwen2.5-32B base/Instruct and OLMo 3.1 32B: base `Olmo-3-1125-32B` (verified as the base of the 3.1 Instruct SFT/DPO/final checkpoints; same chat template as 7B), Instruct-SFT, Instruct-DPO, Instruct. The SFT and DPO checkpoints are stored in float32 and were loaded as bfloat16 (`--dtype bfloat16`). Runtime about 18 GPU-h (OLMo base 11.9).

    | Model | pass@1 | pass@32 | Distinct | Cov@32 | Severe C@32 | TV one list | TV samples | Vote@32 gain | AUROC stated / agree@32 |
    |---|---|---|---|---|---|---|---|---|---|
    | Qwen2.5-32B base | 0.500 | 0.795 | 3.78 | 0.352 | 0.350 | 0.726 | 0.726 | +7.0 | 0.730 / 0.797 |
    | Qwen2.5-32B-Instruct | 0.604 | 0.746 | 1.68 | 0.276 | 0.289 | 0.674 | 0.759 | +1.9 | 0.709 / 0.721 |
    | OLMo 3.1 32B base | 0.326 | 0.796 | 4.32 | 0.347 | 0.385 | 0.746 | 0.742 | +8.4 | 0.629 / 0.831 |
    | SFT | 0.461 | 0.674 | 2.94 | 0.265 | 0.266 | 0.736 | 0.776 | +3.6 | 0.719 / 0.836 |
    | SFT+DPO | 0.584 | 0.784 | 2.29 | 0.293 | 0.324 | 0.694 | 0.755 | +2.1 | 0.693 / 0.755 |
    | SFT+DPO+RLVR (Instruct) | 0.560 | 0.756 | 2.15 | 0.283 | 0.301 | 0.701 | 0.763 | +0.9 (n.s.) | 0.696 / 0.772 |

    - **Sampled collapse replicates, smaller:** Qwen2.5-32B-Instruct vs base pass@1 +10.4, pass@32 −4.9 [−7.1, −2.8] (on the 421 cases where all base samples are valid: −4.8 [−7.8, −1.7]); distinct 3.78 → 1.68; cov@32 −7.6; severe tier −6.1 [−8.3, −4.0]. OLMo 32B Instruct vs base pass@32 −4.0 [−5.9, −2.0], distinct 4.32 → 2.15, severe −8.4. Base invalid rates: Qwen 3.0%, OLMo 26.6%.
    - **Stage attribution does not replicate.** At 32B the largest narrowing is SFT (pass@32 −12.1 [−14.4, −9.9], distinct −1.39, severe −11.9); DPO recovers most of it (+10.9 vs SFT); the final RL step is small (pass@32 −2.8 [−4.3, −1.1], distinct −0.14). At 7B the RL step was the largest (−19.1). OLMo 3.1 32B and OLMo 3 7B use different post-training recipes, so "which stage narrows most" depends on the pipeline.
    - **Stated list improves at 32B in both families**, including OLMo, unlike OLMo 7B: Qwen2.5-32B-Instruct TV −0.053, reference mass +6.7 points, severe named +9.0; OLMo 32B DPO and Instruct TV −0.052 and −0.045, mass +6.3 and +5.6, severe named +5.2 and +2.1 (SFT: TV −0.009, severe −5.9). Sampled TV worsens in every case (+0.014 to +0.034). The list beats the model's own samples by 0.04–0.09 TV for every post-trained 32B model, against about zero for both bases.
    - **Consequences:** voting gain falls from +7.0/+8.4 (bases) to +1.9/+0.9; agreement weakens (0.797 → 0.721, 0.831 → 0.772); one list's top three beats 32 samples for Qwen2.5-32B-Instruct (+5.6) but not for OLMo 32B (SFT −1.6, DPO +0.7, Instruct −0.6; all n.s.). **The stated probability does not overtake 32-sample agreement at 32B** (Qwen: −0.011, tie; OLMo Instruct: −0.077); against 5-sample agreement it wins for Qwen (+0.047) and ties for OLMo.
  - **Full 49-condition elicitation (`--prompt-variant full`, scored with `--top-by-prob`, `runs/pv_full_t07`, T=0.7).** Qwen pairs: the stated full distribution is closer to the reference after post-training (TV −0.052 Qwen3, −0.067 Qwen2.5), the samples are farther (+0.038, +0.012). Qwen3-8B's stated probability beats 32-sample agreement (0.717 vs 0.647, +0.069 [+0.033, +0.104]); Qwen2.5-7B-Instruct's does not (−0.074). Top three of one distribution vs 32 samples: +14.6 (Qwen3-8B), +1.6 n.s. (Qwen2.5). All as with the top-5 prompt. OLMo base/Instruct runs were resumed after the 4 h limit (OLMo Instruct writes long reasoning before the list).
    - **OLMo 3 7B with the full prompt (Klone `~/thesis_overflow/runs/pv_full_t07`, job 42264783; scratch was over quota):** unlike the top-5 prompt, the Instruct model's stated distribution is closer to the reference than the base model's (TV −0.032 [−0.038, −0.027]; top-5 prompt: +0.028) and names severe conditions more often (+6.8), while its samples move away (+0.021). Stated probability vs agreement: +0.044 over 5 samples, tie over 32 (−0.010). Base valid samples 18.6 of 32. So OLMo 7B's lack of list improvement is specific to the top-5 prompt.
  - **Audit rerun on 56 models** (adds the six 32B models and six full-prompt runs; `~/thesis_overflow/analysis/audit_all.json`): gap vs single-answer rate Spearman 0.75 (5 samples), vs stated AUROC 0.82, vs distinct −0.75. With 100 cases × 5 samples the audit matches the full-data pick in 88% of draws on average (min 43% for OLMo 32B DPO, where the two signals tie: 0.693 vs 0.694); mean AUROC lost 0.004 (n=50: 83%, 0.009; n=200: 92%, 0.002).
  - **Prior work found while checking:** "The Alignment Tax" (Liu, arXiv 2603.24124, Mar 2026, single author, not peer reviewed): aligned models give one answer cluster on 40–79% of TruthfulQA questions, sampling-based uncertainty then has AUROC 0.500, and its Zephyr ablation puts the collapse at DPO. Meister et al. (NAACL 2025, 2411.05403): stated distributions beat sampled ones on opinion surveys; instruct and closed models only, no base comparison. Verbalized Sampling §5.3: Tulu-3-70B poem diversity 45.4% (base) → 20.8% (SFT) → 10.8% (DPO) with direct prompting, about 30% with verbalized sampling; the RLVR value is only in the figure.
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

## Mock review round and robustness checks (2026-10-24/25)
- **Three independent mock CHIL reviews** (clinical relevance, ML novelty, statistics; fresh agents given only `paper/chil.tex`) all scored the draft 4/10 (borderline reject). Verified main points:
  - **Uncited close prior work:** Jang, Lee & Kim, "Instruction-Tuned Language Models Cannot Sample from Distributions They Can Describe", EMNLP 2026 Findings, arXiv 2607.25292 (real; posted 2026-07-28). On synthetic and survey targets: instruct models sample worse than their own bases (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B), TV grows at each OLMo-2 stage (0.236/0.321/0.411/0.442), and one-call descriptions reach TV 0.03 ("KNOWS/DOES"). Our contribution is now positioned as the diagnosis extension: correct reference distribution, two pipelines at 7–32B, consequences for voting and uncertainty. Also added: Savage et al. JAMIA 2025 (consistency beats verbalized confidence for diagnosis), Xiong et al. ICLR 2024, Murthy et al. NAACL 2025, Springer et al. ICML 2026, McDuff et al. Nature 2025, Kanjee et al. JAMA 2023, MEDDxAgent ACL 2025 (all verified).
  - Missing chance baselines, possible format/accuracy/floor confounds in the stated readout, a weak audit (no baseline, not held out, label cost), equivalence-from-non-significance wording, and about a dozen text/table inconsistencies (fixed).
- **Robustness (`src/robustness.py`, Klone job 42287775, CPU; `~/thesis_overflow/analysis/robustness.json`):**
  - Chance: uniform guesser pass@32 0.483 (analytic), reference-prior guesser 0.45; severe Considered@32 0.48 (uniform) and 0.56–0.61 (prior), **above every model** including bases (0.34–0.52). Severe Considered@32 on the samples is demoted to the appendix with this caveat.
  - List sampling (top-1 drawn from each sample's own stated list): pass@32 Qwen3-8B 0.607 → 0.811 (base samples 0.730), Qwen2.5-7B-Instruct 0.598 → 0.802 (0.695), Qwen2.5-32B-Instruct 0.746 → 0.880 (0.795), OLMo 3.1 32B Instruct 0.756 → 0.853 (0.796); OLMo 3 7B Instruct 0.422 → 0.577 (0.705, the exception).
  - Format: lists are 81–99% complete for the general models (Qwen bases at T=1 61–62%, medical 61–80%); complete-only TV gaps unchanged.
  - Accuracy strata (Qwen pairs): post list closer among top-1-correct samples (−0.018 to −0.034) and top-1-wrong samples (−0.036 to −0.079).
  - Floor-matched excess TV: Qwen3-8B list 0.479 → 0.435, samples 0.557 → 0.613 (all Qwen pairs same direction).
  - Over-naming falls after post-training (named conditions with reference < 0.01: Qwen3-8B 0.59 → 0.55; named severe with reference < 0.1: Qwen2.5-32B 0.735 → 0.645).
  - First sample counted even if invalid: base vote gains 5.6–18.4 points, final post-trained ≤ 1.7.
- **Audit with held-out reference and baseline (`src/audit.py`, `~/thesis_overflow/analysis/audit_all_v2.json`):** always-agreement right for 66% of 56 combinations (54% of 39 post-trained), mean AUROC loss 0.025 (0.032). Audit on 100 cases × 5 samples, judged on the other 880: 86% (82% post-trained), loss 0.005 (0.007); 50 cases 82%/0.010; 200 cases 90%/0.003. Match rate 96% when |gap| > 0.05 (39 combinations), 75% for 0.02–0.05 (12), 53% below 0.02 (5).
- **Phenopacket stated vs sampled (`src/pheno_consequences.py`, Klone jobs 42287306 + 42290817–21; `~/thesis_overflow/analysis/pheno/`).** Correct = exact or related Mondo match (36–130 correct single-call answers per model; exact only 8–31). Voting gain base 3.1–4.6 vs final post-trained 0.4–1.5 (Qwen3-8B +0.4 n.s.). Agreement AUROC base → post: Qwen3 0.830 → 0.763, Qwen2.5 0.845 → 0.745, OLMo 0.814 → 0.726; stated 0.567 → 0.707, 0.598 → 0.717, 0.464 → 0.652; agreement minus stated 0.25–0.35 (base) vs 0.03–0.08 (post; significant only for Qwen3-8B). Top-3 list minus 32 samples: −4.6 to −7.0 (base) → −2.7 to −3.6 (Qwen2.5-Instruct, OLMo Instruct), +0.4 n.s. (Qwen3-8B). Direction replicates; no crossover.
- Paper: both drafts revised (positioning, baselines, Table 1 list-sampling columns, new appendix table `tab:robust`, `tab:pheno`, prompts/vignette/grader appendix, audit rewrite, wording). CHIL main body ends on page 10.

## Second mock-review round and fixes (2026-10-26)
- **Second round of three fresh mock reviews** on the revised chil.tex: clinical 4/10, novelty 4/10, statistics 5/10 (first round 4/4/4). New verified prior work: Lake, Choi & Durrett, NAACL 2025 (aligned answers aggregate what base samples spread out), Lyu et al. arXiv 2402.13904 (consistency calibration), Ren et al. arXiv 2601.15645 (medical confidence benchmark; DDXPlus use not confirmed from the abstract). Reviewer asks also included few-shot base elicitation, matched list-sampling, chance baselines for every metric, TV control without the gold, clustered intervals and multiplicity, a combined-signal baseline for the audit, and threshold-free can't-miss inclusion.
- **`src/review2.py` (Klone job 42354381, CPU; `~/thesis_overflow/analysis/review2.json`)**, 8 base > final-post pairs:
  - Chance: uniform guesser Cov@32 0.48 (analytic) and prior guesser 0.56, above every model (max 0.42); TV of 32 prior draws 0.774, which base models' samples beat (0.71–0.76) and 5 of 8 post-trained settings do not (0.78–0.85). Patient-blind prior list (top 5 of the average reference): TV 0.794, mass 0.238, DDF1 0.272, severe named 0.149.
  - TV without the true pathology: post list closer for Qwen (−0.018 to −0.056) and OLMo 3.1 32B (−0.023); OLMo 3 7B worse (+0.034 at T=0.7, n.s. at T=1). Samples farther (+0.03 to +0.08).
  - DDF1 improves for Qwen and OLMo 3.1 32B (+0.040 to +0.083), not OLMo 3 7B.
  - Over-naming count per list mixed: Qwen2.5-32B −0.16, OLMo 3.1 32B −0.22, OLMo 3 7B +0.35, Qwen2.5-7B T=1 +0.20, others n.s.
  - Severe inclusion per condition (any reference weight): 4–8 of 17 severe conditions named less often per pair; acute pulmonary edema in most pairs (e.g. Qwen2.5-32B 0.63→0.44, OLMo 3 7B 0.56→0.31), NSTEMI/STEMI (OLMo 3.1 32B 0.49→0.36), stable/unstable angina (Qwen2.5-7B 0.25→0.08, 0.19→0.11), myocarditis (Qwen2.5-32B 0.14→0.05), PE (OLMo 3.1 32B 0.39→0.31).
  - Condition-clustered bootstrap (49 clusters) with Holm over 112 tests: not surviving are the 32B pass@32 losses, sampled-TV increases for Qwen2.5-7B and OLMo 3.1 32B, OLMo 3 7B list changes, Qwen3-8B T=0.7 list-without-gold (p_holm 0.052), and the OLMo agreement-AUROC drops.
- **Combined signal (`src/audit.py`, `~/thesis_overflow/analysis/audit_all_v3.json`):** mean held-out AUROC for the 39 post-trained combinations, fitted on 100 cases × 5 samples and evaluated on 880: logistic on (stated p, agreement@5) 0.753; rank average 0.746; audit's single-signal pick 0.720; always agreement 0.695; always stated 0.671. The logistic beats the single pick for 35/39.
- **Paper:** reframed as a clinical measurement and replication study (new title). Abstract, introduction, related work (Lake, Lyu, Ren, Tian as the stated-readout method; accurate Puri and H-DDx descriptions; DDXPlus set metrics), Table 2 restructured (TV without gold, DDF1, patient-blind row), stated section with per-condition severe losses and mixed over-naming, matched list-sampling comparison, combined-signal recommendation, new appendix section on clustered intervals (`tab:clustered`), and number fixes.
- **Few-shot re-run (Tillicum jobs 389433–389442, about 5 GPU-hours, about $4; `~/thesis_overflow/runs/fewshot_t07`):** 3 fixed DDXPlus training patients with their reference top 5 as JSON; base models as plain completion (stop at a blank line), post-trained models with the same examples as chat turns; T=0.7, 32 samples, 980 cases, 5 pairs. Invalid samples ≤ 0.5% for every model.
- **Few-shot analyses (Klone job 42385061, CPU; `~/thesis_overflow/analysis/{review2,cons,robust}_fewshot.json`; Holm over 70 tests):**
  - Base models gain most from the examples: pass@1 OLMo 3 7B 0.178 → 0.306, OLMo 3.1 32B 0.326 → 0.528, Qwen2.5-32B 0.500 → 0.647; base pass@32 0.76–0.90. Base stated lists barely move (TV 0.712–0.787 vs 0.726–0.794 zero-shot).
  - Stated list, base → post: Qwen pairs survive and grow (TV −0.057 to −0.087, without gold −0.055 to −0.073, DDF1 +0.051 to +0.081, all Holm-significant). OLMo 3 7B stays the exception (TV +0.021 n.s., without gold +0.024). OLMo 3.1 32B weakens: TV −0.022 (cluster p 0.018, Holm 0.32), without gold −0.006 n.s., DDF1 +0.009 n.s.; reference mass +0.047 still significant.
  - Sampled side stronger: pass@32 losses now Holm-significant for all 5 pairs (−0.15 to −0.30, including both 32B pairs), Cov@32 −0.09 to −0.24, sampled TV up +0.04 to +0.12; still below the uniform guesser's Cov@32 (0.467) for every model.
  - Consequences replicate in all 5 pairs: vote@32 gain base +0.06 to +0.11 vs post ≤ 0.011; top-3 list minus top-3 of samples base −0.08 to −0.22 vs post +0.002 to +0.167; agreement@32 beats stated p for every base (by 0.09–0.20 AUROC), stated p beats agreement@5 for every post-trained model (+0.07 to +0.16).
  - Severe conditions named less often after post-training: 5–10 of 17 per pair.
- **Paper (few-shot):** new appendix section `sec:fewshot` with `tab:fewshot` (levels) and `tab:fewshotdiff` (zero-shot / few-shot differences, Holm daggers) in both drafts; abstract and contributions narrowed (stated-list gain robust for Qwen, zero-shot only for OLMo 3.1 32B); 32B sampled-loss sentence, stated section, 32B appendix and limitations updated. CHIL main body still ends on page 10.

## Stated-list aggregation across calls (2026-10-26)
- **`src/aggregate_lists.py` (Klone job 42398374, CPU; `~/thesis_overflow/analysis/aggregate_{zs,fs}.json`)**, 8 zero-shot and 5 few-shot pairs. Per case, the mean of the first m valid stated lists (agg@m), its top 5 renormalized (agg5@m), and the top answers of the same m samples (samp@m), m = 1, 3, 5, 10, 32; cluster-bootstrap intervals over the 49 pathologies.
  - **Averaging base lists closes most of the gap to one post-trained list.** One post-trained list matches the mean of about 3–5 base lists in TV (Qwen3-8B 0.701 vs base agg@5 0.699; Qwen2.5-7B 0.716 vs 0.717; OLMo 3.1 32B 0.701 vs base agg@3 0.701) and for Qwen2.5-32B the mean of 32 (0.674 vs 0.674). OLMo 3 7B Instruct is worse than one base list. At equal size (top 5 of the 5-list mean) the post-trained single list stays closer for Qwen (0.701 vs 0.713; 0.716 vs 0.746; 0.674 vs 0.697).
  - **Equal budget (5 calls, both averaged):** post-trained still closer for Qwen2.5 (TV −0.024 to −0.035, CIs exclude 0), not significant for Qwen3-8B (−0.004, −0.009) and OLMo 3.1 32B (−0.014); OLMo 3 7B worse (+0.06 to +0.07). Few-shot: Qwen −0.015 to −0.030, OLMo 3.1 32B +0.031 (base better), OLMo 3 7B +0.083. Severe naming of averaged lists is lower after post-training for the 32B pairs (OLMo 3.1 32B −0.088; few-shot Qwen2.5-32B −0.094) and OLMo 3 7B.
  - **Averaging post-trained lists adds little:** agg@5 vs one list TV −0.006 to −0.040, mass +0.04 to +0.10 and severe named +0.06 to +0.17, but with 1.4–3.2 more named conditions (a random condition set of the same size would add 0.03–0.07 mass). Post-trained lists saturate (agg@32 names 7.6–13.4 conditions vs 16.6–28.8 for base).
  - **Coverage, not knowledge:** five post-trained stated lists cover 0.44–0.51 of the reference mass with 6–8 conditions, about twice the 32 samples' Cov@32 (0.22–0.28) and level with a uniform guesser that names about 24 conditions (0.48).
  - Reading: post-training largely amortizes aggregation (one call ≈ the average of several base calls), consistent with Lake et al. 2025; it does not add differential knowledge that averaged base lists lack, and averaged base lists name severe conditions at least as often.

## Third family (Mistral-7B v0.3) and Qwen3-8B thinking mode (2026-10-26)
- **Runs (Tillicum `submit_llama.sh mistral|think`, jobs 389965–389971, 4.1 GPU-h, about $3.7; smoke tests 389918–389922 about $0.3).** Mistral-7B-v0.3 base (Mistral-Instruct chat template, `templates/mistral_v03_instruct.jinja`) and Mistral-7B-Instruct-v0.3: zero-shot T=0.7 and T=1, few-shot T=0.7. Qwen3-8B with `enable_thinking`, default prompt, T=0.7, max 8192 tokens (traces about 1,500 tokens; 0 invalid, all traces closed). Samples on Klone in `~/thesis_overflow/runs/{mistral_t07,mistral_t1,fewshot_t07,think_t07}`; analyses (`slurm/new_an.sh`, Klone job 42403868) in `~/thesis_overflow/analysis/third/`; Holm over 70 tests.
- **Mistral base zero-shot is degenerate:** 52% invalid at T=0.7 and 76% at T=1 (continues the prompt, "Click here to see the solution"), and valid answers favor the first listed condition (Acute COPD exacerbation, 36% of valid samples); single-call accuracy 6.3%. The zero-shot pair is therefore not interpretable (post-training "wins" everything, pass@32 +13 to +30). The few-shot pair is the fair comparison.
- **Mistral few-shot (base 0.4% invalid):** replicates the sampled-side pattern: pass@1 +14.1, pass@32 −14.5, distinct −3.7, Cov@32 −0.138, sampled TV +0.058, agreement@32 AUROC −0.185; vote@32 gain base +5.6 vs post +0.8; top-3 list minus samples base −12.6 vs post +7.0; agreement beats stated p for the base (0.826 vs 0.671), stated beats agreement for Instruct (0.747 vs 0.613). Stated list improves modestly: TV −0.036, without gold −0.023, DDF1 +0.046, mass +0.071 (all Holm-significant); severe named +0.008 n.s.; 11/17 severe conditions named less often. Instruct names 0.38–0.47 off-list conditions per list. Aggregation: base agg@5 matches post agg@5 (TV +0.015 n.s.) and names severe conditions more (−0.118 after post-training).
- **Qwen3-8B thinking vs non-thinking (same weights):** pass@1 0.672 vs 0.559, pass@32 0.815 vs 0.607 (base 0.732; thinking minus base +8.4, Holm-significant), distinct 1.76 vs 1.30 (base 3.8), Cov@32 +0.066 (still −0.040 vs base), sampled TV 0.745 vs 0.793 (base 0.750). Stated list best of all 7–8B models: TV 0.670 (non-thinking 0.701, base 0.732), severe named +0.057 vs non-thinking. Error signal flips back: agreement@32 AUROC 0.732 vs stated p 0.638 (non-thinking 0.655 vs 0.734). Vote@32 gain +1.5 (n.s.). Aggregation: thinking agg@5 beats base agg@5 (TV −0.049, severe +0.129) and loses no severe condition against base samples; the only post-trained readout that beats averaged base lists.
- **Paper (third family, thinking, aggregation):** new appendix sections `sec:mistral`, `sec:thinking`, `sec:aggregate` with `tab:third` and `tab:aggregate` (generated by `src/gen_third_tables.py`) in both drafts; one abstract sentence, design sentence, sampled/stated/consequences/discussion sentences and limitations updated. CHIL main body still ends on page 10.

## Third mock-review round and fixes (2026-10-26)
- **Fresh mock reviews** of chil.pdf (no knowledge of earlier rounds): clinical 5/10, novelty 4/10, statistics 5/10. Main asks: per-condition severe claim untested; guesser framing; thinking overclaim; label-free rank average; TV decomposition (stated gain = set selection); clustered intervals everywhere; Holm resolution (1,000 resamples cannot reach 0.05/112); missing work (Arabi, Smith & Yadav arXiv 2609.34929, close to our gm reward; Song, Kempe & Munos 2509.06941; Gu et al. COLM 2026; Reese et al. EJHG 2026, all verified); inconsistencies (T=1 numbers quoted against T=0.7 tables, OLMo 3 7B list w/o gold significantly worse, "only thinking beats averaged base lists", Phenopacket ranges, family counts).
- **Checks of reviewer claims:** deterministic oracles beat the patient-blind guesser (always reference mode TV 0.694, always true pathology 0.714 vs 0.774), so mode-seeking alone does not lose to it; Table 2 one-list TV is a per-case mean (the reviewer's pooling concern was wrong); Table 15 used the first valid list and now uses the per-case mean.
- **CPU analyses (`slurm/review3.sh`, Klone job 42408296; `~/thesis_overflow/analysis/r3/`):** `review2.py` now uses 10,000 cluster resamples with add-one p-values (minimum 2/10001 < 0.05/112), reports the misallocation term TV − (1 − mass) outside the Holm family, and tests each severe condition (cluster bootstrap, BH-FDR over 17, case-swap null for the count of declines). `cluster_within.py` gives clustered intervals for within-model comparisons (vote gain, top-3, AUROC differences).
  - Holm status of the 112 headline tests unchanged except Qwen2.5-7B T=1 sampled TV (now n.s.); OLMo 3 7B list TV w/o gold at T=0.7 is a significant increase; OLMo 3 7B T=1 DDF1 a significant gain.
  - Misallocation rises after post-training in 7 of 8 zero-shot settings (+0.011 to +0.033) and all 5 few-shot pairs.
  - Severe conditions: FDR-significant declines 2–7 and gains 0–6 per final pair; declines never exceed the swap null (p 0.69–0.99). Qwen gains ACS/PE, loses stable angina/APE; OLMo 3 7B gains PE/epiglottitis/myocarditis, loses ACS; OLMo 3.1 32B loses APE and NSTEMI only. Single stages (OLMo 3 7B RLVR, OLMo 3.1 32B SFT and RLVR) do exceed the null (12–13 of 17, p 0.03–0.04).
  - Clustered main-text intervals: 32B pass@32 −4.9 [−10.2,+0.6], −4.0 [−6.8,−1.2]; OLMo 7B RLVR vs DPO −19.1 [−24.6,−13.8]; OLMo 32B SFT −12.1 [−18.0,−6.8], RLVR −2.8 [−5.3,−0.5]; RL-Zero −13.3 [−19.7,−7.3] (T=0.7), −1.8 [−5.8,+2.1] (T=1); AlphaMed pass@32 +8.4 [2.6,14.7]; thinking vs base pass@32 +8.4 [1.7,15.3] (not Holm-significant over 42 tests); Qwen3-8B stated minus agree@32 +0.079 [0.011,0.143]; OLMo-3-7B-Instruct +0.064 [−0.013,0.151] (agree@5: +0.128 [0.052,0.203]).
- **Paper:** reframed around call-to-call variance (new title "Stated or Sampled? Post-Training Packs a Language Model's Differential Diagnosis into a Single Call"); main text rewritten (abstract, intro, setup statistics, sampled with oracles and RL-Zero at both temperatures, stated with decomposition and averaging, severe-condition test, label-free rank average as the default, thinking mode, real cases, discussion, limitations); new Figure 1 (Mistral and thinking rows, base mean-of-five diamonds); Table 2 with misallocation and oracle rows; Table 3 with thinking row; new `tab:severetest`; regenerated `tab:clustered`, `tab:fewshotdiff`, `tab:aggregate`; Appendix H (GRPO) cut to a positioned note in chil.tex (kept in main.tex with a positioning paragraph); appendix inconsistencies fixed. CHIL main body ends on page 10 (35 pages total).

## Matched-temperature and reference-free few-shot controls (2026-10-26)
- **Runs (Tillicum `submit_llama.sh matcht|fewshot-rf`, jobs 391029–391044, about 4.4 GPU-h, about $4; example lists job 391021 and smoke tests 391022–391028 about $0.3).** Matched temperature: Qwen3-8B and Qwen2.5-7B-Instruct at T=1.3 and 1.6 (top-p 1), zero-shot. Reference-free few-shot: the same three training patients (`prompts/fewshot_answers_q32.json`), but their example lists are Qwen2.5-32B-Instruct's own greedy answers (on-list, true pathology ranked first in two and second in one), for all six final pairs. Analyses `slurm/r4_an.sh`, Klone job 42420687, `~/thesis_overflow/analysis/r4/` (Holm over 84 tests per family).
- **Reference-free few-shot: conclusions unchanged.** Sampled collapse in all six pairs (pass@32 −12.7 to −29.7, Cov@32 −0.08 to −0.27, agreement AUROC −0.14 to −0.22, all Holm-significant). Stated list closer for the three Qwen pairs (TV −0.053 to −0.077, without gold −0.049 to −0.059, DDF1 +0.054 to +0.079) and Mistral (−0.059, −0.031, +0.057), all significant; not for OLMo 3 7B (−0.007) or OLMo 3.1 32B (−0.016, n.s.), as with reference-derived examples. Misallocation rises for Qwen and Mistral (+0.013 to +0.024). Five post lists vs five base lists: Qwen −0.016 to −0.040 (closer), Mistral −0.018 n.s., OLMo +0.032/+0.056. Severe conditions: no excess of declines (p 0.30–1.0). Voting gain base +3.7 to +13.7 vs post ≤ +1.4; top-3 list beats samples for every post model (+1.4 to +17.3) and samples beat list for every base; stated p beats agreement@32 for all six post models and agreement wins for all bases. Base pass@1 is lower than with reference-derived examples (e.g. OLMo 3.1 32B 0.453 vs 0.528).
- **Matched temperature: model-dependent.** Qwen3-8B barely broadens with temperature (distinct 1.27/1.53/1.73/1.92 at T=0.7/1/1.3/1.6; pass@32 0.607/0.643/0.668/0.684, below its base at T=0.7, 0.732, and T=1, 0.789: −10.5 Holm-significant at T=1.6 vs base T=1); voting gain ≤1.8; agreement AUROC 0.745 at T=1.6 vs 0.81–0.86 for the base; one-list TV fixed at 0.701. Qwen2.5-7B-Instruct does broaden: at T=1.3 (2.2% invalid) distinct 3.38, pass@32 0.704 vs base 0.695 at T=0.7, voting gain +5.6, agreement@32 AUROC 0.794 vs stated 0.602, pass@1 0.436; at T=1.6 30.5% invalid. Its stated list stays closer than the base's (0.722 at T=1.3 vs 0.779/0.805). So for Qwen3-8B the narrowing is not colder decoding; for Qwen2.5-7B-Instruct much of it can be undone by hotter sampling, at a cost in pass@1 and validity.
- **Paper (controls):** new appendix section `sec:controls` with `tab:refree` and `tab:matcht` (generated by `src/gen_r4_tables.py`) in both drafts; abstract clause and sampled-section sentences on the temperature caveat, a consequences sentence (Qwen2.5-7B-Instruct at T=1.3 restores agreement), discussion and limitations updated. CHIL main body still ends on page 10 (36 pages total).

## Fourth mock-review round and packing test (2026-10-26)
- **Fresh mock reviews** of the reframed chil.pdf: clinical 5/10, novelty 5/10, statistics 5/10 (all confidence 4). Shared asks: "probabilities move away" is a scale artifact (misallocation grows with named mass; normalized by mass it falls in 3 of 5 pairs); central "packing" claim untested; thinking-mode claims n.s. (and few-shot base reaches 0.809, a tie); temperature claim is zero-shot only and vs base at T=0.7; severe "reshuffled" rests on a count null that an overall upward shift passes (APE lost in 4 of 5 final pairs; few-shot 5–10 of 17 and Mistral 11 of 17 untested; five-list averages name fewer severe conditions in 6 of 11 settings); Phenopacket "replicates" overclaims; few-shot amplification is partly post-trained degradation under the few-shot chat format; stated-list gain robust only in Qwen; inconsistencies (−0.035 is a T=1 value not tabulated; "farther than either" wrong for the 32B models; Mistral "smallest gain"; 0.730 vs 0.732 pipelines; GRPO caption "at most 4"; Mistral invalid up to 76%; "five final pairs"). Verified missing work: Karouzos, Tan & Aletras arXiv 2604.16027 (Olmo 3 lineages); Peeperkorn et al. arXiv 2507.20956 (DPO drives OLMo diversity loss); Yang, Li & Holtzman arXiv 2506.17871 (TMLR 2026); Hamidieh et al. arXiv 2604.17112; Luo et al. NeurIPS 2025 arXiv 2505.16690; Zhao, Du & Wang ACL 2026 arXiv 2601.05414; plus URIAL, Zhu et al. 2023, GPT-4 report, Chimirri et al. eBioMedicine 2025.
- **Packing test (`src/packing_test.py`, Klone job 42422017, `~/thesis_overflow/analysis/r5/`).** Per case, TV between a post-trained list and the mean of the base model's lists for the same case ("own"), against the base means of other cases with the same true pathology ("same gold"), random cases, and the reference; within-model benchmarks: a base list vs the mean of the other base lists (base LOO) and a post list vs the mean of the other post lists (post LOO). Clustered intervals.
  - own < same gold in all 15 settings (−0.028 to −0.076) and own << reference (−0.14 to −0.33): post-trained lists keep the base model's case-specific view and are much closer to it than to the reference.
  - own ≈ base LOO at T=0.7 zero-shot for Qwen (0.368 vs 0.372, 0.464 vs 0.470, 0.360 vs 0.375; n.s. or borderline), own < base LOO at T=1 (−0.12 to −0.13) and for OLMo 3.1 32B (−0.046); own > base LOO for OLMo 3 7B (+0.135 at T=0.7), several few-shot pairs (+0.02 to +0.08) and thinking (+0.040). Post LOO 0.10–0.33: post lists barely change between calls.
  - Post top answer equals the argmax of the base mean as often as a single base list does at T=0.7 (n.s.), more often at T=1 (+0.10 to +0.16).
  - Reading: post-training keeps the base model's per-case view and removes call-to-call variation; a post-trained list sits about as close to the base average as one base call at T=0.7 (closer at T=1), so it is a stabilized base call rather than literally the average. OLMo 3 7B Instruct drifts from its base model.

## Fifth round of fixes (2026-10-26)
- **CPU analyses (`slurm/r6_an.sh`, Klone job 42423128, `~/thesis_overflow/analysis/r6/`).** `review2.py` adds a scale-free allocation error (conditional TV between the list and the reference restricted to the named conditions and renormalized), recall@1/3/5 of the true pathology, and a mean-shift swap test for severe conditions; best-elicitation pairs (base few-shot vs post-trained at its better prompt); clustered within-model intervals for few-shot and thinking models; audit uncertainty bootstrapped over the 26 underlying models.
  - Conditional TV falls after post-training for every Qwen pair and OLMo 3.1 32B zero-shot (−0.025 to −0.078, Holm-significant) and rises for OLMo 3 7B at T=1 (+0.064): the earlier "probabilities move away" reading was a scale artifact and is dropped. Few-shot: Qwen3-8B −0.069; Qwen2.5 n.s.; OLMo 3.1 32B +0.078. Reference-free few-shot: Qwen and Mistral −0.051 to −0.100.
  - Recall@5 rises 12–24 points for Qwen and OLMo 3.1 32B (e.g. Qwen3-8B 0.666 → 0.802); OLMo 3 7B 0.470 → 0.485 n.s.
  - Severe mean shift over 17 conditions: positive in 7 of 8 zero-shot settings (p ≤ 0.001; OLMo 3.1 32B +0.001); few-shot Qwen2.5-32B −0.012, OLMo 3 7B −0.018, Mistral −0.009 (p 0.05). Acute pulmonary edema is the largest loss in 4 of 5 zero-shot final pairs.
  - Best elicitation pass@32 losses 13.2–29.9 points.
  - Few-shot stated minus agree@32: Qwen3-8B +0.108, Qwen2.5-7B-Instruct +0.130, Mistral +0.095 (clustered CIs exclude 0), others tie; thinking: agreement better by 0.094 [0.027, 0.170].
  - Audit: rank average − always agreement +0.051 [0.022, 0.074] (29/39), − always stated +0.075 [0.058, 0.098] (39/39), logistic − rank average +0.007 [0.001, 0.015].
- **Paper:** retitled "Stated or Sampled? Post-Training Keeps the Base Model's Differential Diagnosis but Removes Its Variation Between Calls"; abstract, intro, related work (Karouzos, Peeperkorn, Yang, Hamidieh, Luo, Zhao, URIAL, Zhu, GPT-4 report, Chimirri; "what is new" paragraph; Lake and Liu characterizations corrected), setup (readouts with conditional TV and recall, Mistral invalid up to 76%, scorer note), sampled (best-elicitation losses, temperature caveat with T=1 comparison, oracle wording), stated (packing test, recall, conditional TV, robust only for Qwen, severe conditions with mean shift and APE), consequences (agree@32 throughout, tie-limited AUROC, prompt dependence, thinking CI, audit CIs), real cases (what replicates and what does not), discussion and limitations rewritten. Table 1 from the analysis scorer (OLMo stage rows moved to the appendix), Table 2 with conditional TV and recall@5, new `tab:packing` and `tab:ambiguity`, `tab:severetest` with mean shift and largest loss, `tab:refree` with conditional TV, few-shot table with agree@32; appendix fixes (implausible reference example, ECE positioning, severity-tier wording, chest-pain concentration, GRPO caption). CHIL main body ends on page 10 (38 pages total).

## Greedy, temperature frontier and must-not-miss runs (2026-10-27)
- **Runs (Tillicum, 42 jobs, 22.7 GPU-h, about $19.9; estimate was $6–12).** Overrun: OLMo-3-7B-Instruct few-shot at T=1.3 (352 GPU-min, degenerate outputs to the token limit) and the OLMo must-not-miss runs (32B 296, 7B 218 GPU-min). Spend now $156.36 of $250 (stop at $225). Analyses on Tillicum (`slurm/tillicum/r7_an.sbatch`, job 391849; `analysis/r7/` on Tillicum) because Klone home is full and PyPI is blocked locally.
- **Greedy single call (T=0, one answer; base models few-shot, post-trained zero-shot).** Top-1 accuracy post − base: Qwen3-8B +8.7 [−0.4, 18.0], Qwen2.5-7B +5.7 [−3.0, 15.0], Qwen2.5-32B −10.6 [−17.5, −4.3], OLMo 3 7B −14.5 [−25.8, −3.5], OLMo 3.1 32B −7.1 [−14.2, +0.1], Mistral −5.1 [−15.0, +4.5]. Greedy base levels: Qwen2.5-32B 0.712 vs Instruct 0.606; OLMo-3-1025-7B 0.404 vs Instruct 0.258; OLMo-3-1125-32B 0.630 vs 0.558. Recall@5 post − base: Qwen3 +0.242, Qwen2.5-7B +0.168 (both significant); others n.s. except OLMo 3 7B −0.141. List TV: Qwen post closer (−0.03 to −0.09); OLMo 3 7B (+0.083) and Mistral (+0.029) farther. So the post-trained first-answer advantage largely comes from base models' sampling noise: a greedy base call with worked examples matches or beats the post-trained greedy call for four of six families.
- **Few-shot temperature frontier (T = 0.4, 0.7, 1.0, 1.3, top-p 1).** Post-trained pass@1 is flat across T (Qwen3-8B 0.60, Qwen2.5-7B-Instruct 0.43–0.46, OLMo-3-7B-Instruct 0.29–0.35) while pass@32 rises (0.63→0.70, 0.52→0.66, 0.50→0.74); one-list TV does not change with T (0.68, 0.70, 0.81). Base: pass@1 falls with T (0.45→0.18, 0.42→0.22, 0.37→0.11), pass@32 peaks at T=1.0 (0.88, 0.84, 0.88); invalid 25–36% at T=1.3. Frontiers: Qwen3-8B — post dominates at pass@1, base at pass@32 at every T (post at T=1.3 0.596/0.703 vs base at T=0.4 0.447/0.727); Qwen2.5-7B — at matched pass@1 (~0.43) base at T=0.4 and Instruct at T=1.3 coincide (pass@32 0.662 vs 0.660); OLMo 3 7B — base at T=0.4 dominates Instruct at every T on both (0.368/0.803 vs at best 0.345 pass@1, 0.737 pass@32). Agreement AUROC for post models rises with T (Qwen3-8B 0.57→0.68) but stays below base (0.81–0.92).
- **Must-not-miss prompt (six post-trained models, T=0.7).** Valid 87–100% (Mistral 87%). Models shorten the most-likely part (3.7–4.9 conditions) and add 1.5–3.0 must-not-miss conditions. Whole list names more warranted severe conditions than the default list (+0.064 to +0.153; 0.44–0.58 vs 0.32–0.47) and more than base few-shot lists (+0.11 to +0.21), but adds 1.1–2.1 unwarranted severe conditions per list (2.1–2.9 vs 0.7–1.6), mostly pulmonary embolism (named in 88–100% of eligible cases). The most-likely part loses recall@5 for the 7–8B models (Qwen3 −0.087, Qwen2.5-7B −0.095, Mistral −0.051) and is unchanged for the 32B models and OLMo 3 7B. Acute pulmonary edema inclusion rises for every model but Qwen3-8B (e.g. Qwen2.5-32B 0.44→0.63, OLMo 3 7B 0.31→0.58, OLMo 3.1 32B 0.41→0.59). Even when asked, lists name at most 0.58 of warranted severe conditions (reference top five: 0.975).
- **Paper (decoding controls):** new appendix section `sec:decoding` with `tab:greedy`, `tab:frontier` and `tab:mnm` (generated by `src/gen_r7_tables.py`) in both drafts; abstract sentences on greedy decoding and the must-not-miss prompt; sampled-section sentences on greedy and the frontier; severe-condition sentence on the must-not-miss prompt; discussion and limitations updated; text trimmed (ambiguity pointer moved to the discussion, shorter Phenopacket paragraph and Table 3 caption). CHIL main body and AI statement end on page 10 (41 pages total).

## Sixth round: audit, controls and fixes (2026-10-27)
- **Number audit (agent)**: about 1,050 items, about 965 exact; 14 errors (mainly r3 vs r6 bootstrap runs disagreeing on borderline Holm results; T=1 values cited as T=0.7; "tying for the others"; Qwen2.5-32B few-shot recall; range scopes), about 25 untraceable (oracles, severity table, Phenopacket text from older scripts), about 40 small drifts. Fresh reviews: clinical 4, novelty 4, statistics 5.
- **Klone space**: deleted 19 directories' duplicate shard files under ~/thesis_overflow/runs (byte-identical to the merged samples.jsonl; originals on Tillicum; symlinked directories skipped). Home 10 → 8.4 GB.
- **CPU job (`slurm/r8_an.sh`, Klone job 42462909, `~/thesis_overflow/analysis/r8/`)**:
  - Canonical headline bootstrap with 50,000 resamples (`review2_zs_50k.json`): OLMo 3.1 32B agreement drop p_holm 0.058 and OLMo 3 7B severe named p_holm 0.058 (both n.s.); otherwise as before. Main text, tab:clustered and its dagger list now use this run.
  - Cross-family packing (`src/r8_controls.py`): post list closer to its own base mean than to the other family's base means for all five zero-shot pairs (−0.03 to −0.14); within family Qwen2.5-7B-Instruct is as close to other Qwen bases (+0.001 n.s.) and OLMo-3-7B-Instruct closer to OLMo-3-1125-32B's base (+0.026).
  - Severe true pathology (340 cases): zero-shot post recall@5 +0.09 to +0.18 (all five pairs significant), similar to other pathologies; few-shot +0.14 to +0.22 for Qwen3-8B, Qwen2.5-7B, Mistral, n.s. for Qwen2.5-32B and both OLMo.
  - Length-matched must-not-miss proxy (union of two standard lists, 5.6–6.4 conditions): warranted severe named 0.37–0.51 vs must-not-miss 0.44–0.58 (6.2–7.9 conditions); 0.02–0.11 of the 0.06–0.15 gain remains.
  - Complete five-item lists only: recall@5 +0.12 to +0.16, mass, DDF1 and TV gains unchanged for Qwen and OLMo 3.1 32B; OLMo 3 7B n.s.
  - Audit with 32-sample agreement: post-trained (39) agreement 0.742, rank average 0.761 (+0.019 [−0.007, 0.040]), logistic 0.772 (+0.030 [0.013, 0.044]), stated 0.671; base (17) agreement 0.821 > rank average 0.780, logistic 0.827.
- **Paper**: abstract and main text scoped (greedy "pairs" and prompt asymmetry; must-not-miss partly length; combining helps post-trained models modestly; severe true diagnoses recognized; robust stated-list gain for the smaller Qwen models and Mistral with worked examples); packing cross-family sentence; complete-list and severe-true sentences; sharpening, SCPO, Cover@τ, template collapse, CISC and self-consistent-error citations (all verified); audit-error fixes; tab:clustered regenerated from the 50k run; Table 3 moved to the appendix to keep the main body and AI statement on page 10 (42 pages).

## Matched-prompt greedy (2026-10-27)
- **Runs (Tillicum, `submit_llama.sh greedy-matched`, jobs 392114–392125, about 1.0 GPU-hour with the smoke test)**: `runs/greedy_fs/` six post-trained models with the base models' few-shot examples (reference top five), chat; `runs/greedy_rf/` six base models with the reference-free examples (Qwen2.5-32B-Instruct answers), completion. Temperature 0, one answer per case, 980 cases, valid 99.6–100%. Smoke test: clean lists; Mistral base copies the example probabilities (0.4/0.3/0.15/0.1/0.05).
- **Scoring (Klone job 42472858, `slurm/r9_an.sh`, `~/thesis_overflow/analysis/r9/greedy_matched.json`)**, top-1 post minus base, condition-clustered 95% CI:

| pair | same examples (greedy base vs greedy_fs post) | reference-free base vs zero-shot post | original (ref. examples base vs zero-shot post) |
|---|---|---|---|
| Qwen3-8B | +13.7 [6.2, 21.7] | +10.7 [1.1, 20.5] | +8.7 n.s. |
| Qwen2.5-7B | +1.9 n.s. | +7.6 n.s. | +5.7 n.s. |
| Qwen2.5-32B | −4.8 n.s. | −8.0 [−15.5, −0.7] | −10.6 sig. |
| OLMo-3-7B | −5.8 n.s. | −14.5 [−25.4, −3.8] | −14.5 sig. |
| OLMo-3.1-32B | −5.7 n.s. | −0.1 n.s. | −7.2 n.s. |
| Mistral-7B | +10.8 [3.9, 18.5] | −2.9 n.s. | −5.1 n.s. |

  - Same examples: post-trained higher in 2 pairs (both significant), base higher in 3 (none significant). Reference-free base: base ≥ post in 4 of 6 (2 significant), post significantly higher only for Qwen3-8B.
  - Stated list TV, same examples: post closer to reference for the three Qwen pairs (−0.05 to −0.11), farther for OLMo-3-7B (+0.06), n.s. for OLMo-3.1-32B and Mistral; reference-free base gives the same pattern. Recall@5 and severe named rise significantly for the two small Qwen models in both comparisons (and Qwen2.5-32B severe named), otherwise n.s. except OLMo-3-7B recall@5 −17.2 with reference-free base.
  - Implication: the abstract's "as accurate in four of six pairs" holds only for the reference-free base comparison; with identical examples it is three of six (none significant) and post-training wins significantly for Qwen3-8B and Mistral. Under either matched prompt, post-training significantly raises greedy top-1 in at most two pairs, against +12.6 to +27.5 points at one sampled call.
- **Paper (2026-10-27)**: greedy claim rewritten in abstract, intro bullet, main text, discussion and appendix of chil.tex and main.tex ("post-training significantly raises greedy top-1 in at most two of six pairs" with matched prompts); tab:greedy regenerated by `src/gen_r9_tables.py` with all three comparisons; prompt-asymmetry caveat removed from the greedy paragraph and limitations. Fixed main.tex references to chil-only labels (tab:sampled → tab:pairs, sec:sampled → sec:collapse). chil main body and AI statement still end on page 10.

## External must-not-miss sets (2026-10-27)
- **Sets (`src/make_mnm_sets.py`)**: *harm* = DDXPlus conditions among the 15 diseases behind about half of serious misdiagnosis-related harms (Newman-Toker et al., BMJ Qual Saf 2024): NSTEMI/STEMI, pulmonary embolism, pneumonia, pulmonary neoplasm. *chestpain* = life-threatening causes of acute chest pain in the 2021 AHA/ACC guideline (Gulati et al., Circulation 2021): NSTEMI/STEMI, unstable angina, pulmonary embolism, Boerhaave, spontaneous pneumothorax. *union* = 7 conditions. Written as copies of release_conditions.json (severity 1 in set, 5 otherwise), so every "severity <= 2" analysis uses them unchanged.
- **Job**: Klone 42476832 (compute-hugemem; the stf compute allocation was full), `slurm/r10_an.sh`, `~/thesis_overflow/analysis/r10/`. Must-not-miss runs relayed gzipped (65 MB); `r7_analysis.py` reads `.jsonl.gz`.
- **Named when the reference weights it ≥ 0.1 (union)**: zero-shot post − base +0.089 to +0.236 in 7 of 8 pair-temperatures (Qwen3-8B, Qwen2.5-7B, Qwen2.5-32B, OLMo-3-7B), OLMo-3.1-32B −0.003 n.s.; few-shot +0.07 to +0.22 for the three Qwen and OLMo-3-7B, n.s. for OLMo-3.1-32B (+0.065) and Mistral (+0.040); reference-free few-shot +0.10 to +0.23 for Qwen, n.s. otherwise. Harm and chest-pain sets give the same pattern (chest-pain few-shot Mistral −0.071 significant).
- **Per condition (BH q 0.05)**: Qwen loses none significantly at zero-shot (one condition lower, n.s.); OLMo-3-7B loses NSTEMI/STEMI and unstable angina (and lung cancer at T=1) significantly; OLMo-3.1-32B loses 3 (NSTEMI/STEMI, PE, pneumothorax; mean shift −0.024); few-shot Mistral loses 4 (mean shift −0.024), few-shot Qwen2.5-7B 2, Qwen2.5-32B 1 (Boerhaave).
- **Must-not-miss prompt (union)**: whole lists 0.56–0.81 vs 0.36–0.60 standard (+0.15 to +0.26), vs few-shot base +0.21 to +0.48; unwarranted 1.60–2.47 vs 0.58–1.08 per list. Length-matched union of two standard lists: 0.42–0.65 (5.6–6.4 conditions), so 0.09–0.20 of the gain remains.
- **True pathology in the union set (140 cases)**: zero-shot recall@5 +0.10 to +0.26 (OLMo-3-7B +0.10 n.s.); few-shot +0.07 to +0.33 (OLMo-3.1-32B and Mistral n.s.).

## Real case reports: MedCaseReasoning (2026-10-27, running)
- Data: `zou-lab/MedCaseReasoning` test split, 897 PMC case reports (2005–2025; dataset CC-BY 4.0 per GitHub, MIT per HF card), `src/prep_medcase.py` → `data/medcase/cases_test.jsonl` (gold = final_diagnosis, CamelCase/underscores split; year).
- Sampling: `sample_diagnoses.py --task casereport` (system "You are an experienced physician and diagnostician.", open-ended five-item JSON list), zero-shot, 32 samples, T=0.7, 4096 tokens; five Qwen/OLMo pairs (`submit_llama.sh medcase`, jobs 392325–392334). Mistral omitted (its base needs worked examples, and there are no reference lists to build them from). Smoke: Qwen2.5-7B base 29/32 valid, OLMo-3-7B base 20/32 (rambles to the limit), OLMo-3-7B-Instruct 32/32.
- Grading: `judge_medcase.py`, every distinct (case, normalized name) judged by Mistral-Small-24B-Instruct-2501 (outside both families) and Qwen2.5-32B-Instruct; agreement and strict grading (both yes) in `medcase_analysis.py`. Smoke: sensible verdicts, slightly lenient (accepted "Meningioma" for "Sclerosing meningioma").
- **Paper (external sets, claims, structure; 2026-10-27)**: new appendix `sec:extmnm` with `tab:extmnm` (`src/gen_r10_tables.py`) in both drafts; main-text sentence in the severe paragraph; intro bullet and limitations updated; refs newmantoker2024burden, gulati2021chestpain, wu2025medcasereasoning (verified via Crossref/arXiv). chil.tex: new Appendix A `app:claims` (tab:claims, claim → primary evidence / holds / weaker), appendix split into `app:stated`, `app:controls` (opens with tab:controlmatrix, control × claim) and `app:consequences`; Statistics paragraph names the 112 headline tests primary and everything else secondary. MedCaseReasoning row and paragraph pending.
- **MedCaseReasoning chain**: Tillicum's budget plugin counts requested time of queued jobs ($157.51 used + $115.20 pending > $250), so the judges cannot be queued with a dependency; a local script (`$TMPDIR/medcase_chain.sh`) submits `medcase-judge` after sampling and `medcase-an` (`slurm/tillicum/medcase_an.sbatch`) after the judges.

## Seventh round: fresh reviews, retitle, stricter checks (2026-10-27)
- **Fresh reviews** (no knowledge of earlier rounds): CHIL clinical 5, novelty 5, statistics 5 (all confidence 4); workshop draft weak accept. Shared: "keeps the base model's view" overclaimed (lenient packing baseline, within-family blur, OLMo-3-7B); several sentences contradicted tables (Discussion greedy, averaged-list recall, "adds no information", medical "never", "alarms", deployments); first-answer gain conflates prompt format with decoding; pass@32 chance; Qwen2.5-32B-Instruct as judge and evaluated model.
- **Decisions (user)**: retitle both papers "Stated or Sampled? Post-Training Narrows a Language Model's Sampled Differential Diagnosis, Not the One It States"; run the authors'-differential analysis on MedCaseReasoning.
- **Second judge** changed to microsoft/phi-4 (MIT, unevaluated family); smoke agreement with Mistral-Small 98.8% (Phi-4 slightly more lenient). Primary grading = Mistral-Small judge (declared before results); strict = both, sensitivity.
- **Authors' differential** (`src/extract_medcase_ddx.py`, Mistral-Small, job 392537): alternatives other than the final diagnosis from `diagnostic_reasoning`; 893/897 cases have ≥1 (mean 3.5); some vague items ("bacterial", "viral"). `judge_medcase.py --ddx` matches names to the numbered list (item 1 = final diagnosis); `medcase_analysis.py --authors-ddx/--ddx-judge` gives alt_list, alt_samples and the within-model list − samples difference.
- **CPU (Klone 42486874, 42487073; `slurm/r11_an.sh`, `src/r11_an.py`, `~/thesis_overflow/analysis/r11/`)**:
  - Stricter packing: for Qwen3-8B, Qwen2.5-32B, OLMo-3.1-32B the post list is closer to its own base mean than to any other base's (−0.036 to −0.058) but as close to the nearest other post-trained model's (n.s.); Qwen2.5-7B-Instruct's nearest base is Qwen3-8B-Base (+0.014 n.s.); OLMo-3-7B-Instruct is closer to Qwen3-8B-Base and Qwen3-8B.
  - Patient-shuffled pass@32: base 0.05–0.19, post 0.01–0.04. Corrected losses: zero-shot −7.5 (Qwen3-8B), −3.6 (Qwen2.5-7B), −0.5/+0.1 (32B), −25.2 (OLMo-3-7B); best prompt −6.7 to −17.8 (raw −13.2 to −30.0); reference-free examples −4.1 to −14.3.
  - Severe named at τ=0.01 / 0.05: +2.5 to +8.1 points in 7 of 8 settings (OLMo-3.1-32B n.s.).
  - Clustered stated − agree@32 AUROC: OLMo-3-7B-Instruct +0.064 [−0.014, 0.148] (n.s.); others as in the paper.
- **Paper (chil.tex)**: new title and ~200-word abstract; intro bullets; §4.2 heading and packing sentences; first-answer claim restated as prompt format + sampling noise (few-shot pass@1 gain 1.6–5.2 in 4 of 6 pairs); chance sentence; overclaims fixed; new appendix `sec:chance` with `tab:chance`, `tab:packstrict` (`src/gen_r11b_tables.py`); recall@5 at both temperatures; severe thresholds; control and claims tables updated. Workshop: rewritten per review (title, scenario, definitions, grouped table, figure, recommendations), chance and packing nuance.
- **Eighth-round checks (2026-10-27)**:
  - Pooled effects (`src/meta_tost.py`, `src/gen_meta_table.py`, `tab:meta`): zero-shot pooled pass@32 −11.5 [−19.0, −3.9], recall@5 +11.3 [6.6, 15.9], agreement AUROC −0.092 [−0.127, −0.057]; no Qwen-vs-other difference for pass@32, recall@5 or agreement in any setting; TV differs by family with worked examples (p=0.001; 0.015 reference-free) but not zero-shot (p=0.23); severe named differs (p=0.023 zero-shot, 0.006 few-shot). TOST (margins 5 pts / 0.02 TV / 0.03 AUROC): thinking pass@32 non-inferior (90% CI +2.6 to +14.1); 32B zero-shot pass@32 and OLMo-3-7B recall inconclusive.
  - `src/r12_an.py` (Klone 42489270): reference top-5 contains gold 96.9% (97.3% at natural prevalence). Post-trained samples unanimous in 34–57% of cases (Qwen3-8B 78%); on non-unanimous cases agreement AUROC 0.61–0.66 post vs 0.78–0.83 base; Qwen3-8B stated ≈ agreement there (0.611 vs 0.617). Size-matched top-m list − samples: post −0.5 to −5.2, base −8.5 to −14.6 (the post-trained top-3 advantage is list size). Natural prevalence (134,529-patient test split): pass@32 −4.9 to −31.5 (all CIs exclude 0), recall@5 +10.3 to +17.7 (OLMo-3-7B −0.3), TV −0.047 to −0.069. Validity convention changes only pass@1 (OLMo-3-7B +7.1 vs −3.5). Acute pulmonary edema replacements: PE / ACS for most; AF and influenza for OLMo-3.1-32B.
  - Frontier invalid rates at T=0.7/1.0 filled from `robustness.py` (Tillicum job 392581).
  - Text: verified citations (Kadavath 2022, Kuhn 2023, Zhang 2024, Cui 2025, ProRL, Hager 2024, AMIE); positioning sentence; both invalid definitions stated; AUROC script-difference note; agreement sentence uses pooled effect; top-3 and tie sentences corrected; new appendix `sec:further`; pooled-effects paragraph and table in `app:clustered`. chil main body now spills to page 11 (to be fixed in the readability pass).
- **Release package (2026-10-27, local `release/`)**: `release/code/` = `git archive` of src, slurm (gpu scripts under `slurm/gpu_cluster/`), prompts, templates, refs.bib, anonymized (`$TMPDIR/sanitize.py`: cluster paths → `$THESIS`/`$OVERFLOW`/`$REPO`, user/account/partition/cluster names → placeholders; relay/laptop/scrontab scripts dropped; no secrets), README mapping paper items → scripts → outputs, `unpack_outputs.sh`, and 143 analysis JSONs (anonymized). `release/outputs/`: 198 gzipped sampled-output files (1.9 GB) from every paper run except exploratory ones (peek*, go_old, e7 v1), paths anonymized while streaming; leak scan hits are model-generated text only ("Kloner", "klonedl", an invented "Tillicum Syndrome"). Case-report outputs and results to be added.
- **Number audit (3 agents)**: workshop ~130 numbers, 0 mismatches (claims tightened: abstract best-prompt qualifier, losses incl. few-shot Qwen2.5 and APE under DDXPlus labels, strict grading definition, invalid/first-answer definitions, dagger criteria, limitations). CHIL ~3,100 numbers, 21 mismatches fixed: severetest zero-shot rows regenerated from the 50k run (Qwen2.5-7B sig 2/2; OLMo-3.1-32B 5/0, P 0.876; three null P values); 0.01–0.11 length-matched; uniform guesser Distinct 23.6, Cov 0.47; oracle 0.18–0.23; severity moderate 0.31 (four cells); stages pass@1 0.265; 14 case-level intervals; frontier peaks 0.85–0.88; thinking CI [1.7,15.4]; AlphaMed/m1 case CIs; APE replacements; Mistral 3 of 17 significant and 0.5 off-list; 1.4–3.6 more conditions; family-difference wording; chance caption (parser difference up to 1.8 pts); AUROC script tolerance 0.006; Llama-based audit models disclosed. ~190 untraceable numbers (older Phenopacket/medical/era/prompt-variant outputs not in the released JSONs), none in the main text.

## Case-report results (2026-10-27)
- Sampling complete (897 × 8 models; OLMo-3-7B base finished via tail jobs 392589/392590, duplicates removed at merge). Judges 392844–392846; scoring 392916 (`analysis/medcase/medcase.json`, `medcase_strict.json`).
- Judges agree on 97.6% of 391,987 names (κ 0.77; match rate Mistral-Small 4.5%, Phi-4 6.3%).
- Post − base (primary grading, Holm over 36): pass@1 +2.0 to +4.7 (all); pass@32 −12.5 (Qwen3-8B), −3.5 (Qwen2.5-32B), −3.1 and −2.2 n.s.; distinct −5.9/−2.1/−2.4, OLMo-3-7B +9.4 (22.5 vs 13.1; base valid 67%); recall@5 of one list +5.1/+3.0/+5.8/+1.8; authors' alternatives covered by one list −0.7/−0.5/−0.2 n.s., −1.3 (OLMo); by 32 samples −14.0/−8.9/−8.1/−3.1 (all sig). Within post-trained model, list − samples on alternatives: +11.0 (Qwen3-8B), +7.5 (Qwen2.5-32B), −0.4 n.s., −8.2 (OLMo); base −2.4/−8.8/−0.5 n.s./−10.0.
- Consequences differ from DDXPlus: vote@32 − single post +2.5 to +11.8 (base 7.4–7.9); agreement better than stated p for every model; agreement AUROC falls only for Qwen3-8B (0.853 → 0.788), rises for the others. Strict grading: same directions (Qwen2.5-32B pass@32 and OLMo recall lose significance). Era: no drop for ≥2023 cases.
- Paper: chil main paragraph, intro bullet, abstract (voting/agreement claims scoped to DDXPlus), discussion, appendix `sec:medcase` + `tab:medcase` (`src/gen_r11_tables.py`), claims (i) and control rows; workshop paragraph and table cells; main.tex synced. Release: case-report outputs, judges' verdicts, authors' differentials and results added (207+4 files).

## Ninth round: fresh reviews, case-report grouping fix, joint bootstrap (2026-10-27)
- **Fresh reviews (final drafts)**: CHIL clinical 5, novelty 5, statistics 5 (confidence 4); workshop weak accept.
- **Case-report grouping flaw (fixed)**: `medcase_analysis.py` grouped all correct top answers as one key ("GOLD") but wrong answers by normalized text, inflating voting gains and agreement AUROC. Now every answer is grouped by normalized text. Rerun (Tillicum 392989): voting gains base +1.6 to +2.7, post −0.7 to +3.0 (were +2.5 to +11.8); stated − agreement AUROC after post-training: Qwen3-8B +0.085 [0.020, 0.146], Qwen2.5-32B +0.062 [0.006, 0.120], others n.s. (base: agreement as good or better, OLMo −0.172); acc@50 stated > agreement for all four post-trained models. Pass@32 loss significant only for Qwen3-8B after Holm; OLMo alternatives −3.0 n.s. Conclusion reversed: case reports now follow DDXPlus in direction (smaller).
- **`src/r13_an.py` (Klone 42498597, `analysis/r13/r13.json`; review2 loader)**: stated-list entropy zero-shot −0.13 to +0.08 nats for Qwen/OLMo-3.1-32B (OLMo-3-7B −0.52), rises with examples (+0.05 to +0.25; Mistral −0.18); per-case misses of a severe true pathology −9 to −18 points, of all warranted severe −10 (OLMo-3.1-32B −2.2 n.s.); common cases (≥16 valid both): pass@16 −3.4 to −30.5 (Qwen2.5-32B n.s.), distinct −1.5 to −2.7; agreement AUROC on cases non-unanimous for both 0.72–0.77 → 0.61–0.67; acc@50 agreement > stated for every base model, stated > agreement for Qwen3-8B and OLMo-3-7B zero-shot and for every post-trained model with worked examples. Joint cluster bootstrap pooled (replaces DerSimonian–Laird): zero-shot pass@32 −11.9 [−15.4, −8.5] (Qwen − other +7.1 [2.6, 11.6]), recall@5 +11.2 (contrast +6.8 [3.6, 10.0]), TV −0.035 (contrast −0.045), agreement −0.095 [−0.125, −0.061] (contrast n.s.); with examples agreement −0.19 (contrast −0.058 sig), cond TV pooled n.s.
- **Paper**: case-report text everywhere rewritten (abstract, bullet, §4.4, discussion, appendix, claims (g)(h)(i), control row); overclaims fixed (direct-answer scope, stronger models for base view, every Qwen for recall, four of six pairs, zero-shot severe, agreement individually significant only for Qwen, thinking "better signal again", no "understate", patient-blind TV comparison dropped, memorization claim withdrawn, authors' alternatives wording, settings defined, medical overconfidence in main text); tab:meta replaced by joint bootstrap (`src/gen_r13_tables.py`), new `sec:joint` + `tab:r13pairs`; claims and control tables shrunk to fit. CHIL main body ends at the bottom of page 10 (AI statement on p. 11; required sections don't count). Workshop updated (case reports, recommendations fraction 86%, pooled 0.10).

## Tenth round: one pipeline for appendix tables, judge check, cleaned alternatives, recovery smoke test (2026-10-27)
- **One pipeline (`src/r14_an.py`, `src/r14_pheno.py`, `src/gen_r14_tables.py`; Klone 42502670, `analysis/r14/`)**: tab:pairs, tab:severity, tab:prompts, tab:stages, tab:big32, tab:fullprompt, tab:robust, tab:medical, tab:era regenerated from review2's per-case loader; every appendix interval condition-clustered (Phenopacket: by disease, 749 clusters); bold = significant after Holm within each table. Levels unchanged except ECE (now over samples with a valid top answer: Qwen3-8B-Base T=1 0.196 vs 0.214, AlphaMed-7B base-RL 0.181 vs 0.202) and tab:robust (right/wrong TV, over-naming share now per list; list-sampled column dropped, it is in tab:sampled). Changes in conclusions: AlphaMed vs parent pass@1 n.s. (+3.9 [−0.5, 8.5]) and severe-tier consideration n.s. (+4.3 [−1.2, 10.9]); base-RL vs AlphaMed pass@1 n.s.; Phenopacket AlphaMed pass@32 p=0.06, m1 significant from k=4. Statistics paragraph and tab:clustered caption no longer cite an earlier scorer.
- **Case-report judges vs silver labels (`src/medcase_silver.py`, Tillicum 393148, `analysis/medcase/silver.json`)**: positives = word-set match with the final diagnosis (10,346 top answers), negatives = no shared content word (155,522); 72–83% of top answers labeled. Sensitivity 99.95–100% (both judges); acceptance of negatives (upper bound on false positives) Mistral-Small 1.4–3.1%, Phi-4 2.6–5.2%, strict 1.4–3.0%; base vs post differences ≤0.5 points for primary and strict (all n.s.), Phi-4 alone −1.2 [−2.1, −0.3] for Qwen3-8B. Authors'-alternative judge: 94–97% of exact matches mapped to the right item, 5–7% of unrelated names mapped to some item.
- **Cleaned authors' alternatives (`src/clean_authors_ddx.py`)**: 148 of 3,168 category-only entries dropped (infection, malignancy, viral infection, metastatic disease, ...); 886 cases keep ≥1 (3.4 on average). Coverage moves ≤0.4 points; post − base on samples −14.1/−8.8/−8.3/−2.9 (n.s.); list − samples within post-trained +10.6 (Qwen3-8B), +7.1 (Qwen2.5-32B).
- **Recovery experiment, smoke test only (`--prompt-variant seeded`, `src/seeded_an.py`; Tillicum 393138–393140)**: each of 32 calls gets a stratified random u and the model picks from its own top five by cumulative probability. 8 cases: picks always present and in the list; rule followed 39% (Qwen3-8B), 40% (Qwen2.5-7B-Instruct), 66% (OLMo-3-7B-Instruct); distinct picks 1.1 / 3.3 / 6.8 vs 1.0 / 1.8 / 1.8 for ordinary samples and 5.6 / 7.0 / 7.3 if the rule were followed. Full run (5 post-trained models × 980 × 32, modes `seeded`) awaiting approval.
- **Audit fixes (subagent audit of ~900 numbers)**: full-prompt stated−agree32 interval [+0.013, +0.127]; case-report stated−agree intervals [0.022, 0.148] / [0.006, 0.117]; workshop Qwen2.5-32B case pass@32 dagger, OLMo base recall 0.221; "at most 2.0%" invalid for later OLMo stages; main.tex RL-from-base caveat synced; judge wording 99.9–100% / 5.2%.
- **Seeded pick, full run (Tillicum 393269–393273, ~6 GPU-h; Klone 42509813, `analysis/seeded/seeded.json`)**: rule followed 40/40/58/65/84% (Qwen3-8B, Qwen2.5-7B-It, Qwen2.5-32B-It, OLMo-3-7B-It, OLMo-3.1-32B-It). Distinct seeded vs ordinary: +0.0 (Qwen3-8B, n.s.), +1.1, +3.0, +3.5, +4.3; pass@32 −3.3, +7.2, +11.4, +26.8, +10.0 (all Holm-significant); vs base pass@32 −15.7 (Qwen3-8B), −2.6, +6.5, −3.3 (n.s.), +5.7. pass@1 −20.3 for both 32B models. Agreement AUROC of picks not better than ordinary agreement for any model. Exact rule following would give pass@32 0.82/0.79/0.89/0.70/0.87.
- **Closed-set case reports (`make_medcase_closed.py`, `medcase_closed_an.py`; Tillicum 393316–393323, scoring 393567, `analysis/medcase/closed.json`)**: 10 candidates per case (gold, cleaned alternatives, distractors with no shared content word); OLMo-3-7B base capped at 1,024 tokens, text after invented chat turns dropped. Post − base, all four pairs, Holm-significant: pass@32 −23.6/−20.0/−11.5/−16.6, distinct −1.0 to −1.4, alternatives by 32 samples −22 to −27 points; one list recall@5 +4.1 to +7.5, alternatives by one list +2.5 to +6.9; pass@1 +5 to +15. Within post-trained models one list covers 37–64 points more alternatives than 32 samples (base 12–34). Agreement AUROC falls in all four; post stated−agree +0.104 (Qwen3-8B), −0.047 (Qwen2.5-7B), ties otherwise.
- **Free-length DDXPlus lists (`--prompt-variant freelen`; Tillicum 393310–393315 + resumed 393564/393565; Klone 42513180, `analysis/r14/r14_freelen.json`)**: list length base → post 7.0 → 9.9 (Qwen3-8B), 10.7 → 5.2 (Qwen2.5-7B; base lists often run to the token limit), 5.8 → 4.0 (Qwen2.5-32B). Post lists contain the true pathology more often (+16.3, +8.8, +10.8) and are closer to the reference (TV −0.072, −0.067, −0.047); reference mass named +14.9, −1.9 (n.s.), +2.8; samples narrow as before (distinct −2.5 to −2.9; pass@32 −10.9, −5.7, −1.6 n.s.).
- **Judge audit sheet** (`make_judge_audit.py`): 200 middle-zone names (half base, half post, occurrence-weighted) in `audit/judge_audit.csv` for the author to grade; key kept separately.
- **Paper (2026-10-27)**: new appendix subsections `sec:freelen` (`tab:freelen`), `sec:seeded` (`tab:seeded`), `sec:closedcase` (`tab:closedcase`) from `src/gen_r15_tables.py`; claims rows (b)(d)(i) extended and new row (j); three control-matrix rows; claims and control tables split over two pages (`tab:claims2`, `tab:controlmatrix2`) because both overflowed. CHIL main text: free-length clause in §4.2, closed-set sentence in §4.4, recovery sentence and recommendation in the discussion; cuts elsewhere keep the body at 10 pages. main.tex: same appendix additions, two new findings items, limitation line. Workshop: closed-set sentence (4 pages). Release: new scripts, analysis JSONs and sampled outputs (seeded, free-length, closed-set).
- **Audit of this round's additions (subagent, ~310 items, 2026-10-28)**: all table cells, daggers, bold marks and Holm counts correct. Fixed: base list − samples range 11–34 (was 12–34); closed-set control row (g) "partly" (voting n.s.); seeded control row notes OLMo-3-7B's list improves; seeded text "none significantly below its base model" instead of "reach", list changes within 0.02/0.01, Qwen3-8B wording (follows as often as Qwen2.5-7B but rarely picks lower-ranked items: 1.25 distinct vs 5.96 under the rule); case-report error-signal claims scoped to open-ended answers (closed-set: agreement better for Qwen2.5-7B-Instruct) in the main text, discussion and claim (h).

## Eleventh round: fresh mock reviews and fixes (2026-10-28)
- **Fresh reviews (4 independent agents)**: CHIL clinical ML 5/10, statistics 5/10, LLM post-training 5/10 (all confidence 4; clinical relevance, novelty, clarity 2/5); workshop weak accept (poster). Shared criticisms: "stated list does not narrow" partly definitional; abstract overclaims; no clinician validation; incremental novelty; dense text; sampled metrics reward spread / frontier should lead.
- **Title** (author's choice): "Stated or Sampled? Post-Training Narrows the Differential a Language Model Samples, Not the Single List It States".
- **Equivalence tests (`src/tost_nulls.py`)**: greedy top-1 equivalent within 5 points in no pair (Qwen2.5-32B and OLMo-3-7B lower with reference-free examples); case-report alternatives covered by one list equivalent within 5 points in all four pairs; stated ≥ agreement non-inferior (0.03) only for Qwen3-8B and Qwen2.5-32B (open-ended) / Qwen3-8B and OLMo-3-7B-Instruct (closed set).
- **Breadth within one list vs across calls (`src/r16_breadth.py`, Klone 42537210, `analysis/r16/r16.json`; `gen_r16_tables.py`, tab:breadth)**: one list names ~5 conditions for both; post-trained +5–7 points reference mass and +7–10 points warranted conditions (not OLMo-3-7B). Union of 32 lists: −4.6 to −10.0 conditions, −10 to −13 points mass, −7 to −14 points warranted (all significant); still >2× the samples' mass (0.47–0.59 vs 0.20–0.28).
- **Temperature frontier, matched pass@1 (`src/frontier_matched.py`; 7–8B Tillicum 394579; 32B runs `frontier32` mode 16 jobs + 394768)**: OLMo-3-7B −13.0 to −31.9 points pass@32 at equal pass@1; Qwen3-8B and Qwen2.5-7B no overlap (Qwen2.5-7B frontiers meet). 32B: base at T=0.4 beats post on both readouts at every T; matched −7.4 to −15.4 (Qwen2.5-32B), −4.5 (n.s.) to −17.0 (OLMo-3.1-32B). Table tab:frontier32.
- **Case reports grouped by Mondo concept (`medcase_analysis.py --group mondo`, Tillicum 394576, `analysis/medcase/medcase_mondo.json`)**: 29% of distinct names resolve; distinct −12 to −15%; voting gains ≤3.2; agreement AUROC +0.01–0.04; post stated−agree significant only for Qwen3-8B (+0.063 [0.001, 0.128]; Qwen2.5-32B +0.020 n.s.).
- **Free-length for OLMo pairs not run**: smoke test (394554/394555) left 44–59% of OLMo base samples without a usable list (56–66% at the 1,024-token cap).
- **Text**: abstract rewritten around three findings; "narrower" defined; overclaims fixed (Qwen recall scope, greedy "no longer significant", must-not-miss losses for some models, thinking mode, "restores pass@32" except OLMo-3-7B, 32B "do not survive correction", related-work result removed, leftovers removed); §4.2 rewritten around breadth; tab:packstrict moved to appendix; rare-disease paragraph cut to one sentence. Workshop: table gains First answer and case Alternatives columns, pass@k figure added, settings explained, 4 pages.
- Tillicum spend after this round: $198.68.
- **Audit of this round (subagent)**: fixed Mondo distinct reduction (11–16%), error-signal claims scoped to non-inferiority for two of four models (Discussion, claim (h), §4.4), 'restores pass@32' qualified everywhere (OLMo-3-7B checkpoints excepted, incl. RL-Zero 0.701 vs 0.705), main item 2 OLMo-3.1 scope, main item 3 labelled five-sample agreement, workshop column renamed pass@1, frontier range '7–32 points (one 32B point n.s.)', matched-temperature limitation scoped to the two Qwen 7–8B models, TV range at T=0.7 (−0.043 to −0.062; 8–12% of the patient-blind–oracle gap), tab:breadth caption note on sample basis.

## Twelfth round: second fresh review, fixes and restructure (2026-10-28)
- **Second fresh reviews**: CHIL clinical 5 (clinical relevance 3), statistics 5, LLM post-training 4; workshop weak accept (poster). Remaining: novelty, clinical validity (patient-blind baseline, no clinician), title scope, density, "restoration is tautological", judge per-case false positives, base-model prompt confound, one reasoning model.
- **Chance table recomputed with the headline loader (`src/r18_chance.py`, Klone 42539870, `analysis/r18/r18.json`; `gen_r18_tables.py`)**: old script took a sample's first matched name as its top answer (OLMo-3-7B zero-shot −30.1 vs −28.3 in Table 1). Best prompt now defined as each model's highest-pass@32 prompt: raw −13.2 to −29.9; corrected −5.9 to −17.6, significant in 4 of 6 (Qwen3-8B −7.7, Mistral −5.9 n.s.). Reference examples corrected −7.2 to −17.6 (Qwen3-8B n.s.); another model's examples −4.9 to −14.8.
- **Union recall and drawn readouts (`r16_breadth.py` extended, Klone 42539996, `analysis/r16/r16b.json`)**: union of lists contains the true pathology 0.85–0.93 (post-trained Qwen), drawn pass@32 0.79–0.88 below it; Table 1 drawn columns recomputed with the headline loader (Qwen3-8B 0.812/4.98 vs 0.811/4.96 before); OLMo-3-7B base drawn 0.700 < sampled 0.705.
- **List TV skill vs patient-blind list (Table 2 column)**: post 15–23% vs base 3–13% (Qwen, OLMo-3.1-32B); OLMo-3-7B 0% → −5.5%.
- **Conservative case-report grading (`medcase_analysis.py --conservative`, Tillicum 394813, `analysis/medcase/medcase_conservative.json`)**: judge acceptances sharing no content word with the target dropped; base models lose slightly more; post − base alternatives by samples −12.9/−6.8/−7.4/−1.5 (was −14.1/−8.8/−8.3/−2.9); Qwen2.5-32B pass@32 −3.7 becomes significant.
- **Text**: verified errors fixed (greedy two of three pairings, pooled agreement [0.06, 0.13], frontier 6.6–32, conditional TV −0.030 to −0.079, severe "all eight", Discussion view wording, "on DDXPlus" scope, workshop figure temperature/TV range/table columns). CHIL restructured around one stated claim; §4.3/§4.4 merged with new main-text table tab:conseq; severe paragraph cut to three sentences; rare disease/medical to one sentence.
- **Audit (subagent) of this round**: 10 mismatches and 4 contradictions fixed (abstract greedy wording, chance 32B sign, "every model" for drawn pass@32 except OLMo-3-7B base, controls-row setting, "zero-shot base model's pass@32", thinking mode in voting sentence, tab:conseq dagger sources, workshop Qwen3-8B n.s., bold scope, bullet scope, pooled 0.10, conservative wording). Auditor's claim that +0.030 was in-sample checked and rejected: it is held-out (fitted on 100 labeled cases, 31-sample agreement; `r8/audit_agree32.json`).
- **Workshop revision (2026-10-28)**: Table 1 rebuilt (agreement AUROC dropped; TV of one list and best-prompt pass@32 loss raw/chance-corrected added; one significance rule: 95% interval includes zero, condition-clustered for DDXPlus, case bootstrap for case reports; double dagger = equivalent within 5 points); new Figure 2 (`src/fig_passk_diff.py`: post − base pass@k vs k for the five zero-shot pairs, condition-clustered bands, Okabe–Ito colors with markers and direct labels); definitions added (14 metrics, equivalence test, alternatives extraction, invalid-answer rates 1–40% base / ≤0.1% post); frontier, greedy, severity labels and paraphrase variants cut to clauses; citations McDuff 2025, Kirk 2024, Tian 2023; first-answer gain stated as 9–23 points; TV modest/skill sentence; title scope sentence. Still 4 pages.
- **Workshop fixes after third review (2026-10-28)**: McDuff et al. now cited as a stated-list evaluation; chance-corrected range 8–18 (6 for Mistral); Fig. 2 caption scoped (k=32 interval excludes zero except Qwen2.5-32B); TV range 0.67–0.86; sharpening citation attached to an interpretation; voting 'adds at most 1.9 points'; thinking-mode wording; case-report split 'for the Qwen models'; 'single-call answer' throughout; code and data line; Fig. 2 shortened to keep 4 pages. Third workshop review: weak accept (borderline), soundness 4, relevance 4, novelty 3, clarity 2.
- **Workshop items 1–8 and figure fix (2026-10-28)**: Figure 1 diamond (mean of five base lists) moved to its own lane so it no longer hides post-trained markers (CHIL, long version and workshop); workshop Figure 1 limited to the five T=0.7 pairs; Table 1 'Distinct' column dropped; patient-blind list, warranted, mass and mean-of-five-lists defined; greedy-decoding and logistic-combiner sentences cut; best prompt stated as chosen on test cases, with the fixed-prompt corrected loss (7–18) alongside; contribution sentence added; 14-metric description shortened. CFP check: short papers up to 4 pages (references not specified), AAAI format, single-blind, EasyChair, due Nov 20; archival in Springer SCI (camera-ready Jan 8); in-person presentation required (Montreal, Feb 22/23). 72B runs: tensor-parallel option added (--tp, custom all-reduce off, eager) after a flashinfer JIT failure; jobs pinned to one node.
- **Qwen2.5-72B pair (2026-10-28)**: Tillicum jobs 395116–395119 (4 shards × 2 H200 each; zero-shot 10–23 min, worked examples 4–7 min per shard), 980 cases × 32 samples for base and Instruct in both settings; spend $199.17 → $204.37 (about $5.20). Scored on Klone (42545455, slurm/r20_72b.sh: review2, r14_an --groups big72, r18_chance, r16_breadth; 42545933: external must-not-miss sets) → analysis/part1/r20. Earlier pairs reproduce exactly in r16/r18. Results: zero-shot pass@1 +12.2, pass@32 −2.4 [−6.9, 1.8] (corrected +0.9, n.s.), distinct 3.01→1.38, recall@5 +15.4, TV list −0.059, agreement AUROC 0.824→0.719, vote gain 4.2→0.0; worked examples pass@32 −12.3 (corrected −8.8 [−14.6, −3.6]), pass@1 +3.7 n.s., recall@5 +6.4; drawn pass@32 0.885; must-not-miss union 0.468→0.585. Paper: 72B rows in tab:big32 (section renamed 32B and 72B, new paragraph), tab:chance, tab:breadth, tab:models; one §4.1 sentence; abstract ranges 3.0–4.9 and 12–30 (five of seven corrected losses significant); breadth ranges 5–9 / 7–13; workshop Table 1 row and sentence; discussion trimmed to keep CHIL page 10. Audit (subagent): one rounding fix (TV −0.046) and stale ranges updated. 72B is outside the 112 headline tests (Holm over its own 14 per prompt).
- **Workshop clarity rewrite (2026-10-28)**: new Figure 1 (src/fig_example_case.py): one DDXPlus patient (ddx_test_120286, allergic sinusitis) read four ways — reference, Qwen3-8B-Base answers of 32 samples (allergic sinusitis 16, 4 invalid), Qwen3-8B answers (URTI 32/32), Qwen3-8B first list (all three reference conditions). New rate (src/miss_list_rate.py, analysis/part1/r21): the 32 answers never name the true pathology while the first list does for 7.2–19.9% of patients with post-trained models (Qwen3-8B 19.9, Qwen2.5-7B 14.7, Qwen2.5-72B 14.1, OLMo 3 7B 13.1, Qwen2.5-32B 11.0, OLMo 3.1 32B 7.2) and 2.9–3.5% with base models. Old Figure 2 (pass@k difference) dropped; dumbbell figure is now Figure 2. Text rewritten around one story (samples collapse, single list keeps coverage, voting/agreement lose value, case reports), then revised after a stand-alone reading test (subagent): terms defined in Setup (warranted, first answer/list, error signal, thinking mode), main/secondary/added pairs stated once, Table 1 best-prompt column renamed 'pass@32 change, best prompt' with base models' best prompt (worked examples) stated, 'hotter base model' → 'base model at a lower temperature', OLMo 3 7B lists 'do not improve', drawing-from-list method defined and reported in Results, judge checks moved to Limitations, Holm note for the 32B zero-shot losses restored. Title unchanged (user decision). Still 4 pages.
- **Workshop review fixes + single-answer arm (2026-10-28)**: fresh workshop mock review: weak accept (relevance 5, novelty 3, soundness 3, clarity 2, significance 3, confidence 4). Verified fixes applied: equal-size comparison (three most frequent of 32 answers vs top three of one list: samples win by 7–14 points for every base model; list wins by 6–14 for post-trained Qwen; OLMo about even), measurement-validity framing (base-model samples cover the most case-report alternatives, 0.33–0.38 vs ≤0.32 for a post-trained list), related work (Verbalized Sampling, Sample What You Say, Forcing Diffuse Distributions), must-not-miss conditions named, Qwen scope in abstract, chat templates, numbers for temperature sweep, thinking mode (agreement 0.732 vs 0.638, 1.76 distinct, pass@32 0.815), closed-set (22–27 / 2.5–6.9) and the 86% audit; worked-example persistence per model; pooled-list numbers; equivalence margin justified. Dropped to fit: discussion paragraph, Yue and Huang citations, chance floors, case-report voting sentence. Single-answer arm: new prompt variant 'single' (one diagnosis per call, same JSON shape); Tillicum jobs 395556–395559 (smoke) and 395560–395563 (full, 1–15 min per shard; spend 204.37 → 205.19); scored on Klone 42553017 (analysis/part1/r22). Zero-shot T=0.7: Qwen3-8B distinct 3.49→1.18, pass@1 +21.7, pass@32 −8.4 [−13.8, −3.4], agreement AUROC 0.824→0.607, vote gain 2.7→−0.4, base invalid 13.9%; Qwen2.5-7B distinct 3.84→1.43, pass@1 +15.3, pass@32 −8.4 [−13.9, −3.5], agreement 0.822→0.683, vote gain 5.6→1.4, base invalid 9.4%. Stated probability is uninformative here (one-item list renormalizes to 1). Workshop sentence added; still 4 pages. Not yet in CHIL/long version.
- **CHIL/long version: single-answer arm and equal-size comparison (2026-10-28)**: new paragraph 'One diagnosis per call' and tab:single in sec:robust (both drafts); new paragraph 'Sets of equal size' in sec:breadth (top three of one list vs three most frequent of 32 answers from tab:consequences: samples win for every base model by 7.3–13.6, list wins for every post-trained Qwen model by 5.6–13.7, OLMo Instruct +1.9/−0.6, 72B −7.4/+8.4; plus per-model samples-miss/list-has rates 2.9–3.5% base vs 7.2–19.9% post). CHIL main text: §4.1 clause (one diagnosis per call, 1.2–1.4 vs 3.5–3.8) and §4.2 sentence (list's advantage is not its length); to keep page 10: dropped the ≥16-valid-answers clause, the 32B frontier values, the discussion's must-not-miss sentence, the chat-template citation in related work, and shortened the tab:conseq caption. main.tex findings 1 and 2 updated. CHIL 61 pages, body + AI statement end on p. 10; both drafts 0 undefined references. Claims table and control matrix not updated.
- **Reference check, consistency pass, claims tables (2026-10-28)**: all 69 cited references verified by three subagents (arXiv API, Crossref, ACL Anthology, proceedings): all exist; fixed author order (jang2026cannotsample → Jang, Kim, Lee + pages; reese2026phenopacket drop misplaced Robinson; jiang2025hivemind drop Albalak per NeurIPS version; cheng2025guru truncated to agreed first six + others), venues for now-published preprints (lin2023urial ICLR 2024, liu2025prorl NeurIPS 2025, lyu2024consistency AAAI 2025, karan2025sampling ICLR 2026, agarwal2025entropy NeurIPS 2025, huang2024sharpening ICLR 2025, hamidieh2026complementing ICLR 2026, zhang2024diffuse COLM 2024), kanjee pages 78–80; @article without journal → @misc; BibTeX now warning-free. Correction of own earlier claim: the capped top-three comparison is not equal-size (post-trained top-3 answers are often fewer than three conditions); at exactly equal size (sec:further) post-trained readouts are about even (−0.5 to −5.2) while base samples win by 8.5–14.6 — text in all three papers now says the design does not favour lists and a post-trained list leads because its samples collapse ('list's advantage is not its length' removed). Cross-paper consistency (subagent, 17 items) fixed: main.tex findings distinct range (1.3–2.4 vs 3.0–4.9 at T=0.7), thinking 'at least matches', stated−agree clustered interval, Phenopacket and medical-confidence wording, stale TV/mass/severe ranges, OLMo T=1 and RL-Zero TVs, voting 14.1; CHIL conditional TV −0.025 to −0.078, tab:conseq −0.011/−0.077, 12.4 (not 12.5), 1.3–2.6; abstract 'at temperature 0.7'; workshop temperature qualifiers, list-prompt referent, zero-shot list TV 0.67–0.83, final diagnosis 3–6 (Qwen), 82% for post-trained models. Verified pass@k crossing between k=2 and 16 in all five pairs (fig_passk_diff.json added to release r21). Claims table rows (b), (d), (g) and two control-matrix rows (Qwen2.5-72B; one diagnosis per call) added. CHIL body ends on p. 10; workshop 4 pages.
- **Fresh CHIL mock review + fixes (2026-10-28)**: three reviewers (post-training, clinical informatics, statistics): 5/5/5 (soundness 4/4/3; clinical relevance, novelty, clarity 2/2/2). Re-analyses: src/r23_reanalysis.py (Klone 42560339; analysis/part1/r23/r23.json) — drawn pass@32 of the post-trained model vs the worked-example base: Qwen3-8B +0.3 (corrected +2.1), Qwen2.5-7B +3.2 (−1.0), Qwen2.5-72B +0.5 (−4.5), all n.s.; Qwen2.5-32B −1.3 (−7.8 [−13.5, −1.9]), OLMo 3.1 32B −4.1 (−7.6 [−14.6, −0.4]), OLMo 3 7B −28.3 (−22.4); coverage corrected against another patient's reference: samples' Cov@32 loss 2.0–9.1 points in five of six pairs (72B −1.7 n.s.; raw 5.1–15.6), list mass gain 5.3–7.5 (OLMo 3 7B +0.2). src/r23_medcase_eq.py (Tillicum 396796; r23_eq.json): case reports at equal size (m = min(3, distinct)): list > samples within every post-trained model (+2.7 to +6.9) and Qwen bases (+1.2 to +5.0; OLMo base −1.6); post list top three vs base three most frequent: +2.9, +0.4 (n.s.), +4.6, OLMo −2.5. Closed-set chance: uniform guesser 0.97 (32 samples) / 0.50 (one list). Text: §4.2 converge sentence reconciled with claim (a); restoration claim reworded in abstract, discussion, appendix and claim (j) (matches worked-example base for Qwen 7–8B and 72B, not 32B or OLMo 3 7B); control matrix frontier cell and column (c) header fixed; abstract greedy result moved to the first finding; §4.1 72B base 3.0 noted; 'a single list loses nothing' and the sharpening inference toned down; intro 'general evaluation and uncertainty toolkit'; closed-set chance caveat; new appendix paragraphs (coverage corrected for chance; case reports at equal size) and a §4.3 clause; discussion 'four of five can draw themselves' dropped (CHIL) / tempered (long). Workshop mirrored (toolkit wording, equal-size case-report sentence). CHIL body ends p. 10; workshop 4 pages.
- **Title change (2026-10-28, user decision)**: all three papers now titled 'Stated or Sampled? Sample-Based Evaluation Misreads the Differential Diagnoses of Post-Trained Language Models' (evaluation-methodology framing requested by two of three CHIL reviewers; the old title's single-list claim failed for OLMo 3 7B). Page limits unchanged.
