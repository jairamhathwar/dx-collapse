# Paper A plan: measurement-first (decided 2026-10-17)

**Working title:** *Pass@k Hides Differential Collapse: A Controlled Study of Post-Training in Medical Diagnosis*

**Targets**
- ICML 2027. The abstract is estimated around Jan 16 and the full paper around Jan 22; confirm both on icml.cc.
- Fallbacks: CHIL 2027 (Jan–Feb), ICLR 2027 workshops, AMIA Annual (around mid-March), then a master's thesis chapter.

**Constraints:** independent project with no clinicians and no credentialed data. Compute is UW Hyak: STF account plus the checkpoint pool. Detailed run records live in [EXPERIMENTS.md](EXPERIMENTS.md); prior work and positioning in [NOVELTY.md](NOVELTY.md).

---

## 1. Thesis
Post-training makes models more accurate at the top of the differential while collapsing the rest of it. Single-answer metrics, including pass@k, do not show this. Evidence so far:
- **Medical RL (AlphaMed) improves pass@k at every k on DDXPlus, so there is no sharpening tax.** Yet the sampled differential is far narrower than the reference posterior:
  - about 1.8 distinct answers per 32 samples;
  - answer entropy about 0.3, against 1.87 for the reference;
  - about 25% of the reference's probability mass covered.
- **On rare disease (Phenopackets), medical post-training raises stated confidence while accuracy stays under 2%.** Mean stated confidence: base 0.51, AlphaMed 0.69.

## 2. Contributions
1. **Measurement framework.** Sample-level distributional metrics measured against a reference posterior: entropy, total-variation (TV) distance, probability-mass coverage, plus calibration. These sit alongside pass@k, on two complementary tasks:
   - DDXPlus: common conditions, closed set, with a reference differential;
   - Phenopackets: rare disease, open-ended, graded against the Mondo ontology.
2. **Controlled comparison of post-training methods from one parent.**
   - Qwen2.5-7B-Instruct and its medical descendants: AlphaMed (RL only), HuatuoGPT-o1 (SFT then PPO), m1 (SFT distillation only).
   - Plus general post-training controls: Qwen2.5-7B base vs Instruct, and Qwen3-8B-Base vs Qwen3-8B (with and without thinking).
3. **Findings.** These are hypotheses still being tested, with the current evidence above:
   - Medical RL does not sharpen, measured by pass@k.
   - Distributional collapse comes from general post-training.
   - Medical post-training increases overconfidence where the model is out of its depth.
4. **Small-scale intervention.** GRPO on Qwen3-1.7B or 4B with LoRA, comparing four reward families:
   - standard RLVR (top-1 accuracy);
   - **set-based rewards** (Puri et al. 2026 Multi-RLVR and set-Brier);
   - **distribution-based rewards** (proper scoring against the stated differential; overlap with the reference posterior);
   - combinations of these.

   Each is evaluated on the paper's distributional metrics and on **transfer to rare-disease overconfidence**.

## 3. Experiments and status
| # | Experiment | Status |
|---|---|---|
| E1 | Phenopackets go set (500 cases): Qwen2.5 trio + base, T=0.7 | done; m1 rerun at 8k tokens queued |
| E2 | DDXPlus (980 cases): Qwen2.5 trio + base, T=0.7 | base and AlphaMed done; HuatuoGPT and m1 running |
| E3 | DDXPlus at T=1.0: base, AlphaMed, Qwen3 pair | queued |
| E4 | Qwen3-8B-Base vs Qwen3-8B (and thinking mode) on both datasets | running |
| E5 | Qwen2.5-7B base control on both datasets and both temperatures | queued |
| E6 | Phenopackets full eval set (1,401 cases) for the final numbers | todo, after the go-set analysis |
| E7 | GRPO intervention: acc / set-based (Puri) / distribution-based / combined; Qwen3-1.7B, then 4B | code done, dry run queued |
| E8 | Intervention transfer: evaluate E7 checkpoints on Phenopackets calibration | todo |
| E9 | Optional: Llama-3.1-8B family (AlphaMed-8B, HuatuoGPT-o1-8B) | needs Meta license acceptance |

## 4. Timeline to ICML (~Jan 22)
| Weeks | Dates | Work |
|---|---|---|
| 1–2 | Oct 17–31 | Finish E1–E5. Analysis figures: pass@k vs distribution, calibration reliability diagrams, per-condition collapse. |
| 3–4 | Nov 1–14 | E7 at 1.7B, all reward families × 2 seeds. Run E6. |
| 5–6 | Nov 15–28 | E7 at 4B (best two reward families) and E8. Write first draft of the measurement sections. |
| 7–8 | Nov 29–Dec 12 | Ablations: reward weights, temperature, K. Seeds. Optional E9. |
| 9–10 | Dec 13–26 | Full draft. Holiday slack. |
| 11–13 | Dec 27–Jan 16 | Revise, finalize figures, register the abstract. |
| 14 | Jan 17–22 | Submit; post to arXiv. |

## 5. Risks and mitigations
- **The DDXPlus reference differential comes from a rule-based engine**, and the true pathology is ranked first for only about 70% of patients. Describe it as a reference posterior, and report the gold-only metrics alongside.
- **Collapse may come from DDXPlus itself rather than from post-training.** E3, E4 and E5 settle this: the base models at T=1.0 are the key test.
- **Contamination on Phenopackets.** Report results split by publication era.
- **Puri et al. overlap on the intervention.** Position it as a comparison of set-based vs distribution-based rewards, and keep it secondary to the measurement findings.
- **Compute on shared STF/checkpoint GPUs.** Use resumable job arrays with throttled concurrency; the intervention uses small models with LoRA.
- **Dropped Hyak connections.** All runs and scoring are queued jobs with dependencies, so nothing depends on the connection staying up.
