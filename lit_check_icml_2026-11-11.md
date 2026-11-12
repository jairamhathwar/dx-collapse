# Literature check for an ICML 2027 method paper (2026-11-11)

Three parallel searches covered:
- diversity-preserving RL;
- stated vs sampled distributions, and training models to sample what they state;
- decoding and model size.

I personally opened the closest papers: Gao et al., Arabi et al., MaD-RL, Self-CTRL, Karouzos et al., and the scale section of Guru. Everything else was opened by a search agent (arXiv abstract or HTML) unless marked "snippet".

## Verdict by direction

### A. Our group reward with a reference target (`gm`): already published

Five papers already reward each GRPO sample according to how its answer moves the group's answer counts toward a target distribution:

| Paper | What it does | Target |
|---|---|---|
| Gao, Shi, Hu, Harrison, *When Rewards Depend on the Group, GRPO Plays a Tournament*, arXiv 2611.02404 (2 Nov 2026) | Proves that GRPO/RLOO centering biases any group-dependent reward (diversity, coverage, distribution matching) toward competing against groupmates. Proposes Team-GRPO, an unbiased version that needs only two samples. On GlobalOpinionQA, GRPO collapses onto the majority answer and Team-GRPO cuts error from 0.20 to 0.08. | human opinion data |
| Arabi, Smith, Yadav, *Sample What You Say*, arXiv 2609.34929 (28 Sep 2026) | Per-rollout "witness advantage" (from maximum mean discrepancy, using leave-one-out group counts). Shows that a single group-level reward is zeroed out by GRPO's centering. Qwen2.5-1.5B; synthetic distributions, urns, opinions. | written in the prompt; the model's own stated distribution appears only as a failing baseline |
| Kulkarni et al., *MaD-RL*, arXiv 2609.31644 (12 Sep 2026) | Correctness reward plus a reward matching the group's answer frequencies to a target (L2, KL, JS), added or gated. Math and code tasks. | specified (uniform or peaked) |
| Anschel et al., *GAPO*, arXiv 2511.12596 (EMNLP 2025) | Frequency-aware group reward pushing toward a uniform spread over valid answers | uniform |
| Song, Kempe, Munos, *Outcome-based Exploration*, arXiv 2509.06941 | Penalty for answers repeated within a batch, plus a count-based bonus | none |

Our clipped total-variation form is only a minor variant. More importantly, **Gao et al.'s critique applies to how we trained it** (plain GRPO centering). Our gains (+0.10–0.14 distinct answers, 2 seeds) would read as incremental. **No-go as an ICML contribution.**

### B. The measurement finding (one stated list beats many samples after post-training): published in general form

- Meister et al., NAACL 2025 (arXiv 2411.05403): models describe distributions better than they can sample them.
- Jang et al., Findings of EMNLP 2026 (arXiv 2607.25292): base vs instruction-tuned models, tracked across post-training stages; a single "describe" call beats sampling.
- Verbalized Sampling (arXiv 2510.01171): a stage ablation on Tülu.

Our additions (a correctness task, compute-matched readouts, clinical data, case reports) suit CHIL but are thin for ICML. Puri et al. (ICML 2026) already use DDXPlus.

### C. Training the model to sample its *own* stated distribution, without labels: open, with high risk of being scooped

- I found no paper that uses the model's own verbalized list as the target for its sampling.
- Arabi et al. use targets written in the prompt and name free-form extensions as their next step.
- Self-CTRL (Pres, Ruis, Ghebreselassie, Li, Andreas, arXiv 2606.18327) trains behavior to match self-explanations, but only for a scalar coin bias and for refusals.
- TTCL and related work go the other way: they train stated confidence toward sample frequency.
- The Andreas group has both Self-CTRL and the multi-answer RL work, so a concurrent paper is plausible.

What a paper would need:
- **Target:** a frozen or stop-gradient stated list from the initial model, so the list can't collapse to match the samples.
- **Estimator:** Team-GRPO or witness-style.
- **Open-ended answers:** normalize answers so open-ended labels can be compared, which Arabi lists as an unsolved limitation.
- **Evidence:** show gains in correctness and pass@k, not just distance to the target, on at least two tasks.
- **Baselines:** witness advantage, MaD-RL, Team-GRPO, Puri's multi-answer RL, Verbalized Sampling, temperature, three seeds each.

### D. Diversity is suppressed, not lost, at large scale (decoding recovers base pass@k at 32–72B but not at 7–8B): open, lower scoop risk

- Not reported directly anywhere. Prior decoding tests are single-scale or small:
  - Yue et al.: temperature matching, 7B only.
  - Zhou & Li: min-p sweep, 3B.
  - Karouzos et al.: "in the weights", Olmo 3; I checked the paper, and its test was only suppressing chain of thought.
- Closest prior:
  - Guru (arXiv 2506.14965) §4.3: RL pass@k crosses below the base model at k=64 for 7B but never for 32B. That concerns RL itself, not recovery by decoding.
  - Li et al. (arXiv 2506.07295, snippet): larger models tolerate high temperature better, a likely mechanism.
- Tensions to address:
  - Springer et al. (ICML 2026): mode collapse worsens with scale. This fits our finding that distinct answers stay low.
  - Schaeffer et al. (arXiv 2506.13681, snippet): min-p shows no edge when tuning budgets are equal.
- What a paper would need:
  - a size ladder (Qwen2.5 0.5–72B, Qwen3 4–32B, OLMo-3 7B vs 32B lineages);
  - at least two tasks (DDXPlus plus MATH or code);
  - a decoding grid with equal tuning (temperature × min-p / top-nσ / top-p), with k up to 256;
  - token-level mechanism checks (rank and log-probability of base-correct answers under the post-trained model);
  - "lost" tests at small scale (prefix forcing, logit mixing with the base model).
- **Risk:** reviewers may see it as an analysis paper or a decoding curiosity.

## Recommendation

- **Drop A as the ICML contribution.**
- **C and D are the two openings**, and they combine naturally into one story: "post-training suppresses diversity the model still states; decoding recovers it only at scale; self-targeted RL recovers it at small scale."
- Pilot both before committing (about two weeks, about $60–90 of Tillicum credit):
  1. **D:** the min-p setting on a Qwen2.5 size ladder (0.5–72B) on DDXPlus, plus MATH-500 at 7B and 32B.
  2. **C:** a self-targeted reward on Qwen3-1.7B and 8B, reusing the existing GRPO code, with Team-GRPO estimation and the frozen initial model's stated list as the target.
- **Go/no-go at week 3:** go if C clearly beats `acc` and the witness advantage on pass@k at equal pass@1, or if D shows a clean size trend on two tasks. Otherwise submit the existing paper to CHIL (February 3).

## Main sources

- [Gao et al. 2611.02404](https://arxiv.org/abs/2611.02404)
- [Arabi et al. 2609.34929](https://arxiv.org/abs/2609.34929)
- [MaD-RL 2609.31644](https://arxiv.org/abs/2609.31644)
- [GAPO 2511.12596](https://arxiv.org/abs/2511.12596)
- [Song et al. 2509.06941](https://arxiv.org/abs/2509.06941)
- [Self-CTRL 2606.18327](https://arxiv.org/abs/2606.18327)
- [Jang et al. 2607.25292](https://arxiv.org/abs/2607.25292)
- [Meister et al. 2411.05403](https://arxiv.org/abs/2411.05403)
- [Verbalized Sampling 2510.01171](https://arxiv.org/abs/2510.01171)
- [Puri et al. 2603.24844](https://arxiv.org/abs/2603.24844)
- [Karouzos et al. 2604.16027](https://arxiv.org/abs/2604.16027)
- [Guru 2506.14965](https://arxiv.org/abs/2506.14965)
- [Yue et al. 2504.13837](https://arxiv.org/abs/2504.13837)
- [Min-p 2407.01082](https://arxiv.org/abs/2407.01082)
- [Schaeffer et al. 2506.13681](https://arxiv.org/abs/2506.13681)
- [Springer et al. 2605.09995](https://arxiv.org/abs/2605.09995)
- [Li et al. 2506.07295](https://arxiv.org/abs/2506.07295)
- [Zhou & Li 2608.29188](https://arxiv.org/abs/2608.29188)
