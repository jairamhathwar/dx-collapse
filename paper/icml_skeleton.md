# ICML 2027 paper skeleton (draft 0, 2026-11-16)

Deadline: expected late January 2027 (not announced). Compute ends when credits lapse at graduation (end of March 2027).

The gated-variant test on Klone picks the branch:
- **Branch M (method):** the gated self-target keeps the silent-error drop at no accuracy cost.
- **Branch A (analysis):** it doesn't; the method becomes one section reporting a trade-off.

Sections 1–4 are shared. Status tags: **[have]** result exists, **[partial]** exists but needs seeds, scale or intervals, **[need]** not run.

## Working titles
- M: *Sample What You State: Label-Free Repair of Silent Errors in Post-Trained Language Models*
- A: *Suppressed, Not Erased: What Post-Training Does to a Language Model's Uncertainty*

## One-paragraph pitch
Post-trained language models repeat the same answer across samples, including when it is wrong. We call a wrong answer that every sample repeats a *silent error*: sampling-based uncertainty methods cannot see it. Silent errors appear with post-training and with scale, on a clinical task and on open-domain question answering. Yet the same models still name the alternatives when asked for a list. (M) We train the model to sample the distribution it states, with no labels, and cut silent errors by about a quarter. (A) We map where the lost uncertainty can and cannot be recovered: by decoding, by reading the stated list, and by training.

## 1. Introduction
- Sampling-based uncertainty (self-consistency, semantic entropy) assumes samples reflect the model's uncertainty.
- Post-training breaks that assumption: samples collapse, errors become unanimous.
- The model's stated list still carries the alternatives (stated vs sampled gap; Meister 2025, Jang 2026).
- Contributions (three bullets, per branch).

## 2. Related work
Source: `lit_check_icml_2026-11-11.md`.
- Stated vs sampled distributions: Meister et al. 2025; Jang et al. 2026; Verbalized Sampling; Lake et al.
- Diversity collapse under RL and post-training: Yue et al. 2025; Wu et al.; Oh et al. 2026; Karouzos et al. 2026; Springer et al. 2026.
- Group-level and distribution-matching rewards: Arabi et al. 2026; Gao et al. 2026 (Team-GRPO); MaD-RL; GAPO; Song et al. 2025.
- Multi-answer training: Puri et al. 2026. Self-consistency training: Self-CTRL.
- Uncertainty and error detection: semantic entropy; verbalized confidence; RLCR.
- Our difference: silent errors as the target quantity; a target that is the model's own stated list (no labels, no given distribution).

## 3. Setup
- Tasks: DDXPlus (49 conditions, reference differential) **[have]**; AmbigQA closed-book (strict and lenient grading) **[have]**; optional GoEmotions **[need]**.
- Readouts: stated list vs 32 samples; single-call answer; agreement.
- Silent error: wrong single-call answer repeated by every other valid sample.
- Error-detection metrics: AUROC and errors caught at 20% referral, for agreement, sample entropy, stated probability **[have: `unc_an.py`]**.
- Models: base/post-trained pairs, 0.5B–72B **[have for DDXPlus]**; Qwen3 1.7B and 8B on AmbigQA **[have]**.

## 4. Post-training creates silent errors
- 4.1 Across pairs on DDXPlus: 19–61% of post-trained errors are silent, 1–9% for base models **[have, with intervals: r25]**.
- 4.2 With scale: collapse appears from 3B up in the Qwen2.5 ladder **[partial: one family, no intervals; small base models mostly invalid zero-shot]**.
- 4.3 On a second task: AmbigQA, 8B, silent errors 0.9% → 24.0% (lenient) **[partial: one pair, no intervals, grading not validated by hand labels]**.
- 4.4 Standard signals do not rescue them: stated probability flags 5–29% of silent errors at 20% referral **[have]**; sample entropy cannot by construction.
- 4.5 The alternatives are still stated: first list vs 32 samples **[have for DDXPlus; on AmbigQA the list barely beats the samples, which must be reported]**.
- Needed: OLMo and Mistral pairs on AmbigQA; intervals everywhere; a second family in the size ladder.

## 5M. Method: self-targeted group reward
- Target: the frozen starting model's mean stated list per prompt (label-free).
- Reward: witness-style per-sample reward from group counts (Arabi et al.), with the model's own target; uncertainty gate (scale by 1 − target's top probability).
- Estimator: plain GRPO centering now; Team-GRPO correction **[need]**.
- Results on Qwen3-1.7B, DDXPlus **[have, 3 seeds, no intervals]**:
  - silent errors 63% → 46% (with accuracy reward), 43–46% (label-free alone);
  - agreement AUROC 0.63–0.64 → 0.69–0.70;
  - cost: pass@1 −2.5 points with the accuracy reward; four fixes (weights, sharpening) traded gain for cost.
- Gated variant **[running on Klone: decides the branch]**.
- Needed for M: 8B on DDXPlus; 8B on AmbigQA; baselines (reference-target witness, MaD-RL, GAPO-style uniform, Song penalty, old `gm`); decoding baselines (temperature, min-p); 3 seeds at 1.7B, 2 at 8B.

## 5A. Where the uncertainty can be recovered
- By reading the list: stated list vs samples across pairs **[have]**.
- By decoding: min-p and temperature recover base pass@32 at 14–72B, not consistently below; family-dependent **[partial]**.
- By training: self-targeted reward cuts silent errors at an accuracy cost **[have, as a trade-off]**.
- By reasoning: thinking mode restores pass@32 for Qwen3-8B **[have, one model]**.
- Needed for A: decoding grid with equal tuning across sizes and two tasks; token-level analysis of where the correct answer sits under the post-trained model; a second family.

## 6. Discussion and limitations
- Synthetic clinical data; closed answer sets; open models up to 72B; one training recipe.
- Lenient QA grading not validated against human labels.
- Self-target inherits the stated list's errors.

## Experiment queue (in order)
1. Gated variant, 3 seeds, 1.7B (Klone, free) — running.
2. Intervals for the pilot tables (CPU).
3. Branch M: Team-GRPO estimator; reference-target seeds; 8B DDXPlus; 8B AmbigQA.
   Branch A: AmbigQA for all pairs; decoding grid; second-family ladder.
4. Hold back about $30–40 for rebuttal-stage runs before the credits lapse.

## Budget
$252 of Tillicum credit as of 2026-11-16. 1.7B runs go to Klone. An 8B training run costs about $17; a full-pair sampling run about $1–3.
