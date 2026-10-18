# Novelty check (2026-10-17)

Verification key: [V] means a primary source (arXiv abstract or HTML) was checked; [S] means it was seen only in search snippets, so verify before citing.

## Closest prior work: Puri et al. 2026 [V]
*Reaching Beyond the Mode: RL for Distributional Reasoning in Language Models.* Puri, Damani, Shenfeld, Ghassemi, Andreas, Kim. arXiv 2603.24844 (Mar 2026). Reported as an ICML 2026 poster under the title "Escaping the Mode"; check the venue.

**What they did**
- Setup: GRPO without std normalization, BNPO loss, Qwen3-8B, 32 samples at T=0.7, effective batch 1536.
- Datasets: DDXPlus (25k training examples, K=3 answers), HotPotQA-Modified, MBPP.
- Multi-Answer RLVR reward: R = Σ 1[a_i ∈ Y*]. Y* is the *set* of correct diagnoses, graded by exact string match.
- Multi-Answer RLCR reward: R = RLVR_multi − (1/K) Σ (q_i − 1[a_i ∈ Y*])².
- A uniqueness/format gate zeroes the reward when answers repeat.

**What they found on DDXPlus**
| | Coverage | Top-1 accuracy |
|---|---|---|
| RLVR, single answer | 0.62 | 0.50 |
| RLVR, multi-answer | 0.79 | 0.42 |

- Calibration: set ECE 0.02 for Multi-RLCR; top-1 ECE 0.34 for single-answer RLVR.
- Diversity: about 8 vs 4 unique answers.

**What they did not do**
- No use of DDXPlus's *probabilities*: no entropy, TV or mass-coverage metrics.
- No comparison against publicly released post-trained model families.
- No rare-disease evaluation, no pass@k sharpening analysis.

**Implications for us**
- They are the main baseline for the training section.
- Their per-answer confidences mean "probability this answer is in the set", so `set_brier` rewards reporting about 0.9 for every member. Our rewards treat the answer as a distribution over diagnoses, which is a different objective (see `rewards.py` tests).

## Risk per contribution
| Contribution | Risk | Note |
|---|---|---|
| A. Collapse vs the soft reference differential, same-base model families | Medium | Puri et al. show RLVR collapse on DDXPlus. The soft-distribution metrics and the controlled public families are new. |
| B. Medical RL has no sharpening tax, yet the distribution stays collapsed | Low–Medium | No text-only medical pass@k study exists. Opposes the medical-VLM (2603.01301) and agentic (2610.01509) sharpening findings. |
| C. Medical post-training raises overconfidence on rare disease | Low (incremental) | The general effect is known (Leng 2410.09724, Mei 2506.18183, RLCR 2507.16806). This setting is unpublished. |
| D. GRPO with a differential-level reward | High | Puri et al. did set coverage plus set-Brier. Still open: proper scoring against the *soft* distribution, sampling-level distribution matching, and rare-disease transfer. |

## Positioning
1. Measurement-first paper (the chosen option). Headline: pass@k hides distributional collapse.
2. Use Puri et al.'s Multi-RLVR and set-Brier rewards as baselines. Compare them with distribution rewards on TV, entropy and mass coverage, and on transfer to rare-disease overconfidence.
3. Controls:
   - Qwen2.5-7B base, so the comparison is not only instruct→RL.
   - Qwen3-8B-Base vs Qwen3-8B.
   - Temperatures 0.7 and 1.0.
4. Describe DDXPlus's differential as a reference posterior from a rule-based engine; the true pathology is ranked first for only about 70% of patients.

## Must-cite
- Diagnosis datasets and LLM diagnosis: DDXPlus 2205.09148; CASANDE 2210.07198; H-DDx 2510.03700.
- RL sharpening, pass@k and support: Yue et al. 2504.13837; Invisible Leash 2507.14843; Sharpening Tax 2610.01509; Guru 2506.14965; medical-VLM RL 2603.01301.
- Calibration under RL: RLCR 2507.16806; Rewarding Doubt 2503.02623; Leng et al. 2410.09724; Mei et al. 2506.18183.
- Diversity loss from post-training: Kirk et al. 2310.06452; West & Potts 2505.00047; Verbalized Sampling 2510.01171; Artificial Hivemind 2510.22954.
- Models studied: AlphaMed 2505.17952; HuatuoGPT-o1 2412.18925; m1 (arXiv ID to verify).
- Rare disease: Reese et al. (Eur J Hum Genet; medRxiv 10.1101/2024.07.22.24310816); Jiang et al. 2608.14683 and 2609.02473.
