# Session handover (written 2026-10-20 ~09:45 PDT; updated ~16:30 PDT)

## UPDATE 2026-10-20 ~16:30. Read this first; it supersedes sections 2 and 6 where they conflict.
- **The group reward works on Qwen3-1.7B.** Full table in EXPERIMENTS.md, "E7 v2 full results".
  - Plain `gm` removes the slight sharpening that `acc` and `dist` cause.
  - Boosted `gm3t12` (gm weight 3, train T=1.2), against `acc`: pass@1 +1.6, distinct +0.10, entropy +0.034, cov@32 +1.5 points, TV −0.014. Two seeds; all CIs exclude 0.
  - Diversity ends *above* the untrained model.
  - All 1.7B Tillicum evaluations are copied to Klone `runs/e7v2_{ddx,go}/` and scored. Use `src/compare_recipes.py` for paired comparisons.
- **In flight (Tillicum), E9: GRPO from Qwen3-8B-Base.**
  - Training: `b8-acc-s0` (job 374700) and `b8-gm-s0` (job 374703), started about 15:45 with a 20 h limit, at about 320 s/step. Expected finish around 10:00 on 10-21.
  - Evaluation runs automatically on full H200s: DDX jobs 374701 and 374704, Phenopacket jobs 374702 and 374705. Outputs go to `runs/eval_ddx/b8-*` and `runs/eval_go/b8-*`.
  - **Next:** copy the outputs to Klone, `runs/e9_ddx/<tag>/`. Symlink the untrained `ddx_qwen3/qwen3-8b-base` and official `ddx_qwen3/qwen3-8b` samples there (T=0.7, same settings). Score with `BASE=qwen3-8b-base`, then compare with `compare_recipes.py`.
  - If gm preserves the base model's breadth while acc collapses it, run seed 1 for both (about $34; the user approved option A).
- **Budget:** $4.54 of $250 used as of 15:45. The E9 pair is expected to add about $36–40.
- **Train script change:** `train.sbatch` (Tillicum) now sets a per-job `MASTER_PORT`. Two jobs on one node had clashed on port 29500.

This note is for the next Claude Code session on Jairam's master's thesis. It is self-contained. Read it,
then [PLAN.md](PLAN.md), [EXPERIMENTS.md](EXPERIMENTS.md) (full run log and findings) and [NOVELTY.md](NOVELTY.md).

## 1. The project in one paragraph
**Working title:** *Pass@k Hides Differential Collapse: A Controlled Study of Post-Training in Medical Diagnosis.*
Target: ICML 2027 (about Jan 22, 2027); fallbacks are CHIL and ICLR workshops. It is also Jairam's master's thesis.
The paper is **measurement-first** (decided with the user). We sample models 32 times per case on two public datasets: DDXPlus
(49 conditions, with a reference differential) and Phenopacket Store (rare disease, graded with Mondo). We measure pass@k,
diversity, coverage of the reference differential and calibration, comparing base and post-trained models built on the same
pretrained weights.
A secondary GRPO section tests rewards that fight the collapse; Puri et al. 2026 (arXiv 2603.24844) is the main baseline.

**Independence rule.** This project is independent of Deloitte, the user's employer. Never use Deloitte resources: the
Elevance DGX cluster, the Deloitte EC2 instance, or the `jhathwar_deloitte` GitHub CLI login on this laptop. Commits go to
the user's personal GitHub through a deploy key on Klone, authored as `jairamhathwar <jairamhathwar@users.noreply.github.com>`.

## 2. Key results so far (details and CIs in EXPERIMENTS.md and results/paper_numbers.json)
- **General post-training trades coverage for top-1 accuracy.**
  - Qwen3-8B vs its base at T=1: pass@1 +27.5, pass@32 −14.6 (curves cross at k≈4–8).
  - Qwen2.5-7B vs its base: pass@1 +22.2, pass@32 −10.0.
- **The differential collapses.** Post-trained models give 1.3–2.6 distinct top-1 answers per 32 samples (base models 3.8–8.3).
  Their answer entropy is 0.11–0.49 (base 0.78–1.59; reference 1.87). Their top-1 calibration *improves*.
- **Severe conditions drop out too.** Among severity-1–2 conditions, the share that is ever named as a top answer falls about 18
  points in both families.
  - Qwen2.5-7B-Instruct practically never names unstable angina, stable angina or myocarditis as its top answer.
  - It also concentrates on "attractor" answers: NSTEMI is its top answer in 13% of samples, against 4% in the reference.
- **Medical RL** (AlphaMed vs Qwen2.5-7B-Instruct) improves pass@k at every k (+3.9 to +8.4) and does not add to the collapse.
- **Rare disease.** Medical post-training raises stated confidence from 0.51 to 0.66–0.69 while accuracy stays below 2%.
- **GRPO on Qwen3-1.7B.**
  - v1 settings (lr 2e-5, no KL) collapsed to a constant answer in several runs.
  - v2 settings (lr 5e-6, KL 0.04, no std normalization, 16 generations, 200 steps) are stable: about +8.5 pass@1. But the
    per-sample `acc` and `dist` rewards are indistinguishable, and neither changes the spread of sampled answers.
  - We then added a **group-level reward `group_match`**, which rewards the spread of top-1 answers across a GRPO group.
    In training it is sparse, because groups are nearly unanimous.
  - A boosted variant (group-match weight 3, training temperature 1.2) is training now.

## 3. Where everything lives
| Location | What |
|---|---|
| GitHub `jairamhathwar/dx-collapse` (private) | Code, docs, paper draft, small results. **The source of truth.** Pushed from Klone. |
| Laptop session folder `.../scratch-2026-10-16-9a3532/thesis/` | Working copy the previous session edited, then rsynced to the clusters. The app may delete it with the session. Prefer a fresh clone of the repo. |
| Klone (Hyak) `/gscratch/scrubbed/jairamh/thesis/` | `code/` (git checkout, the repo), `envs/thesis` (conda env), `data/`, `runs/`, `logs/`. **Scrubbed storage purges files with old timestamps**; see the scrontab below. |
| Tillicum `/gpfs/projects/stf/jairamh/thesis/` | `code/` (rsynced, not a git checkout), `envs/thesis`, `data/`, `runs/`, `logs/`. Persistent. Caches are in `/gpfs/scrubbed/jairamh/thesis`. |
| `~/.thesis_env` on each cluster | Sets `$THESIS` and the cache paths. `source ~/.thesis_env` before anything. |

Paper: `paper/main.tex`, `paper/refs.bib`, `paper/figures/`. It compiles with Tectonic on Klone at
`$THESIS/envs/tex/bin/tectonic -X compile main.tex`. Red `\todo{}`s mark pending items; `refs.bib` entries marked VERIFY need checking.

## 4. Connecting (password + Duo; Claude must never handle passwords)
The **user** runs these in a terminal **on this Mac** (the app's Terminal panel or macOS Terminal), types the password and approves
Duo. Claude then reuses the shared socket non-interactively with `ssh -o ControlPath=~/.ssh/cm-<name> -o BatchMode=yes ...`.
```
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-hyak -o ControlPersist=10h -o ServerAliveInterval=60 jairamh@klone.hyak.uw.edu
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-tillicum -o ControlPersist=10h -o ServerAliveInterval=60 jairamh@tillicum.hyak.uw.edu
```
- **Check first:** `ls ~/.ssh/cm-*` and a test command. Connections drop whenever the laptop sleeps. Logins from another device do
  not create the socket on this Mac.
- **Security:** the user pasted their NetID password into chat on 2026-10-20. It was not used. They were advised to change it at
  identity.uw.edu. Remind them if it hasn't been done.

## 5. Compute and budget
- **Klone (free).**
  - Account `stf`. Partitions: `ckpt-all` (preemptible idle pool, often saturated), `gpu-l40s` and `gpu-l40` (STF priority,
    shared by students), `compute` (STF CPU, often at its group limit).
  - Job scripts are in `slurm/` (`sample.sbatch`, `train.sbatch`, `score_ddx.sbatch`, `score_go.sh`).
- **Tillicum (credit).**
  - RCC student credit **$250**, used **$0.15**, expires 2027-04-19, enforced. Check with `hyakusage`.
  - H200 costs $0.90/GPU-h. A MIG `1g.18gb` slice costs about 1/7 of that (2 CPUs per GPU maximum).
  - `debug` QOS: 1 h, nearly free.
  - Scripts are in `slurm/tillicum/`: `train.sbatch` (H200), `sample.sbatch` (MIG), `submit_train_eval.sh`, `submit_merge_eval.sh`, `build_env.sbatch`.
  - GRPO speed: about 25–30 s/step on an H200, against about 150 s on a Klone A40. A 200-step run takes about 1.5 h, roughly $1.30.
  - Keep a budget log in EXPERIMENTS.md, and stop launching at about $225.

## 6. In flight right now (job IDs are Tillicum unless noted)
| Job | What | Next step |
|---|---|---|
| 373815, 373818 | Train `gm3t12-s0` and `gm3t12-s1` (acc + group_match, weights 1,1,3, train T=1.2), about step 135 of 200 | Evals 373816/373819 (DDX, moved to gpu-h200) and 373817/373820 (Phenopackets, MIG) run automatically |
| 373840, 373843, 373846 (arrays 0–1) | DDXPlus eval of Klone-trained `dist-s0`, `gm-s0`, `gm-s1` (moved to gpu-h200) | Outputs go to `runs/eval_ddx/<tag>/` |
| 373838, 373841, 373844, 373847 | Phenopacket evals of `acc-s0`, `dist-s0`, `gm-s0`, `gm-s1` (MIG) | Outputs go to `runs/eval_go/<tag>/` |
| done | `runs/eval_ddx/acc-s0` (980 cases); merged models for the four Klone runs | |
| Klone, ~20 pending | Older measurement jobs: HuatuoGPT and m1 on DDXPlus, Qwen2.5 base at T=0.7 and Phenopackets, Qwen3 Phenopacket runs, m1 at 8k on Phenopackets, AlphaMed at T=1.0, v1 `acc-s1` evals. Several have auto-scoring jobs chained. | Run when `ckpt-all` frees up |
| Klone scrontab | Daily 03:30 `touch` of everything under `$THESIS`, on `ckpt-all`, to stop the scrubbed purge (`slurm/scrontab.txt`) | Check `logs/touch-thesis-*.out` |

**Not yet done:** scoring of the Tillicum evaluations.
1. Copy `runs/eval_ddx/<tag>/samples.s*.jsonl` and `runs/eval_go/<tag>/...` from Tillicum, through the laptop, into Klone's
   `runs/e7v2_ddx/<tag>/` and `runs/e7v2_go/<tag>/`. The untrained baseline `qwen3-1.7b` is already there.
2. On Klone, run `RUNS=$THESIS/runs/e7v2_ddx BASE=qwen3-1.7b bash code/slurm/score_ddx.sbatch` and the `score_go.sh` equivalent,
   or `runs/score_e7v2_now.sh`.
3. Then run `python code/src/summarize_e7.py --runs $THESIS/runs --prefix e7v2 --dataset ddx` (and `go`).
4. Note that `summarize_e7.py` ORDER doesn't yet list gm or gm3t12; they are appended at the end anyway.

## 7. Suggested next steps (in order)
1. **Score the group-reward runs** (section 6). Success means more distinct answers, higher entropy and higher reference
   coverage@32 than `acc` and `dist`, with pass@1 kept at about 0.46.
2. **Depending on the result:**
   - If boosted `gm3t12` works: run seeds and a weight/temperature sweep on Tillicum.
   - If it doesn't: consider a denser group reward, for example r = min(1, p_ref(a) / (k_a / n)), and longer training.
   - Either way, run the Puri et al. baselines (`multi_rlvr`, `multi_rlvr,set_brier`) under the v2 settings for the paper.
3. **Finish the measurement results** still pending on Klone: HuatuoGPT and m1 on DDXPlus, the full Qwen2.5 base pair (811 of
   980 cases so far), and the Phenopacket publication-era table. Consider moving them to Tillicum MIG slices.
4. **Paper:** fill the TODOs, redraw the figures paper-style, verify the references, and write the training section once E7 settles.
5. **Optional:** the Llama-3.1-8B family (needs the user to accept Meta's license and run `huggingface-cli login` themselves),
   and a 4B GRPO on Tillicum.

## 8. Gotchas learned the hard way
- **Never put an apostrophe in commit messages or heredocs** inside a single-quoted `ssh '...'` command. It breaks the quoting
  (this happened twice).
- **`pkill -f <pattern>` can match and kill your own ssh shell.** Kill by PID.
- **Klone forbids multi-partition jobs.** Don't `scontrol update` jobs into several partitions; it once left jobs with 15-minute
  limits. On Klone, moving `ckpt-all` jobs to other partitions fails on QOS, so cancel and resubmit instead.
- **Slurm on Klone is sometimes slow or times out.** Wrap `squeue`/`sbatch` in `timeout`. A "socket timed out" from sbatch may
  still have submitted the job.
- **Don't run heavy work on login nodes.** Use `srun` or `sbatch` (`-A stf -p compute`, or `ckpt-all` for CPU jobs).
- **Scrubbed purge on Klone:** `unzip` and conda keep old file timestamps, so files can vanish within a day (DDXPlus files,
  conda package caches). Use `unzip -DD` and touch files. The daily scrontab handles this now.
- **vLLM:**
  - It needs `CUDA_HOME` pointed at the env's bundled nvcc.
  - Set `VLLM_USE_FLASHINFER_SAMPLER=0`; the FlashInfer JIT doesn't match torch 13.0.
  - On Tillicum, set `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and every vLLM entry script needs an `if __name__ == "__main__"` guard.
- **Base models need `stop_token_ids` for end-of-turn** (handled in `sample_diagnoses.py`). Parsing is lenient (`parsing.py`);
  DDXPlus names go through `ddx_names.Matcher`. Note that DDXPlus spells it "Larygospasm".
- **Always run a CPU smoke test before GPU training** (`train_grpo.py --cpu-smoke`, about 5 min on CPU).
- **The user follows from other devices.** Send files with SendUserFile, and keep replies concise.

## 9. User preferences observed
- Wants efficient progress and asks "how's it looking" often. Give short status tables.
- Prefers low GPU spend and Hyak/Tillicum over paid cloud.
- Wanted the Tillicum application answers kept short.
- Decided: measurement-first framing, adding the Qwen3 pair, temperature-1.0 checks, the per-condition analysis, the group-level reward.
