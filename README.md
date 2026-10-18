# dx-collapse

Master's thesis code: **Pass@k hides differential collapse: a controlled study of post-training in medical diagnosis.**

We sample LLMs 32 times per case and measure pass@k, diversity, coverage of a reference differential, and calibration on
DDXPlus (49 conditions, reference differential per patient) and Phenopacket Store (rare disease, Mondo-graded), comparing
base models with their post-trained descendants. A small GRPO study compares set-based and distribution-based rewards.

- `PLAN.md`: thesis plan, contributions, timeline
- `EXPERIMENTS.md`: datasets, models, runs, findings, pipeline fixes
- `NOVELTY.md`: prior work and positioning
- `src/`: data prep (`prep_*.py`), sampling (`sample_diagnoses.py`), grading (`score_*.py`, `disease_normalizer.py`,
  `ddx_names.py`, `parsing.py`), comparisons (`compare_runs.py`), training (`train_grpo.py`, `rewards.py`), tests (`test_*.py`)
- `slurm/`: Hyak (Klone) job scripts
- `results/`: small reports and figures (raw samples and data stay on Hyak scratch)

Runs on UW Hyak. Data: Phenopacket Store 0.1.27, DDXPlus (CC BY 4.0), Mondo/HPO; all public.
