# Session handover: dx-collapse (Jairam's master's thesis / ICML 2027 paper)

Written 2026-10-23 ~12:20 PDT by the outgoing Claude session. This file **replaces** the 2026-10-20 handover.
Read it first, then `EXPERIMENTS.md` (full run log with every number and CI), `PLAN.md`, `NOVELTY.md`, and `paper/main.tex`.

Notation: **[V]** = verified in this session from files or command output; **[A]** = assumption or estimate; **[STALE?]** = may be out of date by the time you read this.

---

## 0. Do these first (in order)

1. **Check the cluster sessions.** Run `ssh -O check -o ControlPath=~/.ssh/cm-hyak x` and the same for `cm-tillicum`. If either is down, ask the user to log in (section 4). Never handle the password yourself.
2. **Check the OLMo 3 runs on Tillicum.** These are the only jobs in flight (section 6).
   - The **base model may hit the 4 h time limit** at about 14:06 PDT on 10-23 [A]. At 2h07 it had 512/980 DDXPlus cases (each temperature) and 512/1401 Phenopacket cases [V].
   - Sampling is resumable. Resubmit any unfinished base run with the command in section 6.
3. When all OLMo samples are complete, **copy, score and analyze** them (section 6, steps 2–4). Then **write the OLMo stage-wise results into the paper** (section 7).
4. **Commits are up to date.** Everything above, including this file, is pushed as **`d935238`** [V]. The Tillicum `code/` copy is synced too.

---

## 1. Project in one paragraph

**Working title:** *Pass@k Hides Differential Collapse: A Controlled Study of Post-Training in Medical Diagnosis.*

- **Target:** ICML 2027 (deadline about Jan 22, 2027 [A]); fallbacks are CHIL and ICLR workshops. It is also the user's UW master's thesis.
- **Framing (user's decision):** measurement-first, optimized for what top AI labs care about.
- **Method:** sample models 32× per case on DDXPlus (49 conditions, with a reference differential per patient) and on Phenopacket Store rare-disease cases (graded with the Mondo ontology). Compare base vs post-trained models built from the **same pretrained weights**, on pass@k, answer diversity, reference coverage, severity-tier "consideration" and calibration.
- **Secondary section:** GRPO training with a novel **group-level reward** (`group_match`) that fights the collapse. The main baseline is Puri et al. 2026, "Escaping/Reaching Beyond the Mode", arXiv 2603.24844.

## 2. Hard constraints and user preferences (preserve these)

- **Independence from Deloitte (the user's employer).** Never use Deloitte resources:
  - the Elevance DGX cluster (`/projects/elevance-health/`);
  - the Deloitte EC2 instance;
  - the `gh` CLI on this laptop, which is logged in as `jhathwar_deloitte`;
  - the AWS MCP server or the Deloitte SSH socket `cm-jhathwar@login.dcaic.deloitte.com`.
  - All compute is UW Hyak: **Klone** (free) and **Tillicum** (student credit).
- **Claude never enters or handles passwords or tokens.**
  - The user logs into the clusters themselves (password + Duo push) on this Mac (US-MW3L6HC60X). Claude reuses the ControlMaster sockets with `-o BatchMode=yes`.
  - The user once pasted their NetID password into chat (2026-10-20). It was not used, and they were told to change it at identity.uw.edu. Remind them if it is unclear whether they did.
- **Git.**
  - Repo: personal GitHub `jairamhathwar/dx-collapse` (private). Push **from Klone** with the deploy key: `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_thesis_repo" git push origin HEAD`.
  - Author: `jairamhathwar <jairamhathwar@users.noreply.github.com>`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - **No apostrophes** in commit messages or heredocs inside single-quoted `ssh '...'` commands; they break the quoting.
- **Budget:** low GPU spend preferred. Tillicum credit is $250 and expires 2027-04-19. Stop launching new work at about $225. **Spend: $65.68 at 12:13 PDT 10-23 [V].**
- **Communication style:** the user often asks "how's it looking" and wants short status tables, honest caveats, and concise replies. They follow from other devices, so send PDFs and reports with SendUserFile.
- **Approvals given (do not re-ask):** measurement-first framing; Qwen3 and Qwen2.5 pairs; T=1 checks; DDXPlus; per-condition analysis; group reward; GRPO from base at 8B (two seeds); Puri baselines; the full 1,401-case Phenopacket set; prompt variants; **switching the second model family from Llama to OLMo 3** (approved 2026-10-23).

## 3. Where everything lives

| Location | What |
|---|---|
| GitHub `jairamhathwar/dx-collapse` | Source of truth: code, docs, paper. Last pushed commit **`d935238`** [V]. |
| Klone `/gscratch/scrubbed/jairamh/thesis/` | `code/` (the git checkout), `envs/thesis` (conda: vLLM 0.31.0, torch 2.13 cu130, TRL 1.16), `envs/tex` (Tectonic), `data/`, `runs/`, `logs/`. **Scrubbed storage purges files with old timestamps.** A daily scrontab `touch` job (03:30, ckpt-all) prevents this; check that `touch-thesis` is still queued. |
| Tillicum `/gpfs/projects/stf/jairamh/thesis/` | `code/` (rsynced, **not** git; sync it from the laptop or Klone before running), `envs/thesis`, `data/`, `runs/`, `logs/`. Caches are in `/gpfs/scrubbed/jairamh/thesis`. |
| `~/.thesis_env` on each cluster | Sets `$THESIS`, `HF_HOME` and caches. `source ~/.thesis_env` before anything. |
| Laptop `.../scratch-2026-10-16-9a3532/thesis/` | Working copy the outgoing session edited, then rsynced to the clusters. It is an app scratch folder and **may be deleted with the session**. A fresh session should clone the repo or work from Klone's `code/`. |

**Paper:** `paper/main.tex` + `refs.bib` + `figures/`. Compile on Klone: `cd $THESIS/code/paper && $THESIS/envs/tex/bin/tectonic -X compile main.tex`.

**Moving data:** the clusters can't talk to each other directly. Relay **Tillicum → laptop scratchpad → Klone** with `rsync -rltz --no-perms -e "ssh -o BatchMode=yes -S ~/.ssh/cm-<name>"`. Without `--no-perms`, rsync prints harmless `fchmodat` errors.

## 4. Connecting (the user does this)

```bash
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-hyak -o ControlPersist=10h -o ServerAliveInterval=60 -o ServerAliveCountMax=10 jairamh@klone.hyak.uw.edu
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-tillicum -o ControlPersist=10h -o ServerAliveInterval=60 -o ServerAliveCountMax=10 jairamh@tillicum.hyak.uw.edu
```

- Claude then runs `ssh -o BatchMode=yes -S ~/.ssh/cm-hyak klone '...'` (the hostname after `-S` is arbitrary).
- Sessions die after 10 h or when the laptop sleeps.
- Login attempts have repeatedly failed when the Duo push wasn't approved quickly, or on a timeout at the password prompt. Tell the user to approve Duo immediately.
- The user can open login tabs with the terminal tool (`run_in_terminal`); the sockets appear in `~/.ssh/cm-*`.

## 5. Results so far (all in the paper; numbers from files; CIs are 95% paired case bootstrap)

**DDXPlus, 980 test cases × 32 samples [V]:**
- **General post-training trades coverage for top-1 accuracy** when models answer directly.
  - Qwen3-8B vs Base: T=1, +27.5 at pass@1 and −14.6 at pass@32; T=0.7, +17.0 and −12.2.
  - Qwen2.5-7B: T=0.7, +12.6 and −9.7; T=1, +22.5 and −10.3.
  - Distinct top-1 answers: 1.3–2.6 vs 3.8–8.3. Answer entropy 0.11–0.49 vs 0.78–1.59 (reference 1.87).
- **Severity-tier "Considered@32"** (980 cases):
  - General post-training: −17 to −18 points on the severe tier at T=1, and −10 to −12 at T=0.7.
  - Medical descendants *raise* it: AlphaMed +4.3, HuatuoGPT-o1 +8.4, m1 +13.5.
  - Our 8B GRPO: acc −2.8; gm −1.5, not significant. Official Qwen3-8B: −9.8.
- **Robustness (`sec:robust`, `tab:prompts`):**
  - The plain prompt replicates the collapse (k=32: −7.1 at T=0.7, −8.7 at T=1).
  - **The CoT prompt removes the large-k loss** (+4.7 at T=0.7, 0.0 at T=1) but not the diversity loss (2.1 vs 3.9 distinct). The abstract is qualified accordingly.
  - Format: base models give invalid answers on up to 25% of samples (post-trained ≤0.3%).
    - Valid-only pass@1 gains are +12 to +19 points, a quarter to a third smaller than the raw gains.
    - On the subset where all samples are valid, the k=32 loss holds: −15.2 (Qwen3, 105 cases) and −10.2 (Qwen2.5, 748 cases).
- **Medical family** (Qwen2.5-7B-Instruct parent, T=0.7):

  | Model | pass@1 | pass@32 | Distinct |
  |---|---|---|---|
  | Parent | 0.478 | 0.598 | 1.82 |
  | AlphaMed | 0.516 | 0.682 | 1.87 |
  | HuatuoGPT-o1 | 0.536 | 0.735 | 2.49 |
  | m1 (8k tokens) | 0.486 | 0.779 | 3.29 |

  None narrows the differential further.
- **AlphaMed-7B-base-rl** (medical RL initialized from the Qwen2.5-7B *base*): 0.495 / 0.790 / 3.14 distinct. Severe-tier change −0.7 (not significant).
  - **Provenance [V]:** the AlphaMed paper (arXiv 2505.17952, only v1) documents only instruct-initialized models. The base-rl checkpoint's `config.json` `_name_or_path` points to `Qwen2.5-7B`.
  - The paper therefore treats it as *corroborating, not controlled* evidence.
- **Llama-3.1-8B-Instruct medical descendants** (no parent run): HuatuoGPT-o1-8B 0.554 / 0.810; AlphaMed-8B 0.495 / 0.791; UltraMedical-8B 0.484 / 0.702. Logged in EXPERIMENTS.md, not in the paper body.

**Phenopacket Store, full set: 1,401 cases / 749 diseases [V]:**
- Everything is near the floor: pass@1 1.4–2.2%, pass@32 3.7–7.0%.
- **Medical RL raises confidence** from 0.51 to 0.67–0.71 (ECE 0.49 → 0.64–0.70) while accuracy gains stay under 1.5 points.
- Qwen3-8B vs base: +0.7 at k=1, **−2.2 [−3.1, −1.3] at k=32 (significant)**.
- **Publication era, pass@32:** ≤2022 (n=901) 5–10%; 2023–24 (n=348) ≤1.7%; 2025+ (n=152) 2.6–4.6% (`tab:era`). This suggests memorization.

**GRPO (`sec:training`) [V]:**
- **1.7B, two seeds:** `gm3t12` beats `acc` on pass@1 +1.6, distinct +0.10, cov@32 +1.5.
  - The Puri Multi-RLVR/RLCR baselines barely move pass@1 (0.385 vs untrained 0.378) but raise single-list recall (0.317).
  - gm3t12 beats them on every sampling metric (+8.9 at pass@1).
- **8B from Qwen3-8B-Base, two seeds:**
  - acc: 0.537 / 0.759 / 2.72 distinct.
  - gm: 0.539 / 0.770 / 2.86 distinct.
  - Official Qwen3-8B: 0.558 / 0.607 / 1.27.
  - gm vs acc: distinct +0.14, cov@32 +1.1, pass@32 +1.2 (all significant); pass@1 not significant.

## 6. In flight: OLMo 3 7B post-training stages (Tillicum, launched ~10:06 PDT 10-23)

**Why OLMo:** the user's Meta Llama access request was still pending after more than 2 days, and OLMo 3 is ungated (Apache-2.0). It publishes every stage from one base: `allenai/Olmo-3-1025-7B` → `-Instruct-SFT` → `-Instruct-DPO` → `-Instruct` (RLVR). It also publishes **RL-Zero** checkpoints trained with RL directly from the base, which is the documented, controlled version of the AlphaMed base-RL result. This lets the paper attribute the collapse to SFT, DPO or RL.

**Launch command:** `bash code/slurm/tillicum/submit_llama.sh olmo` (from `$THESIS` on Tillicum).
- Settings: DDXPlus T=0.7 → `runs/meas_ddx/<tag>`, T=1 → `runs/meas_ddx_t1/<tag>`, Phenopackets full 1,401 → `runs/meas_full/<tag>`. Max tokens 4096, model length 6144, k=32, 2 shards each.
- The base model uses `--chat-template code/templates/olmo3_sft.jinja` (the SFT checkpoint's ChatML template), because the base ships none. The flag is new in `sample_diagnoses.py`.

**Status at 12:13 PDT [V]:**

| Tag | ddx T=0.7 | ddx T=1 | full Phenopackets |
|---|---|---|---|
| olmo3-sft, -dpo, -instruct, -rlzero-general | 980 ✓ | 980 ✓ | 1401 ✓ |
| olmo3-base | 512/980 (running; jobs 383434) | 640/980 (383435) | 512/1401 (383436) |
| olmo3-rlzero-mix | **failed** | failed | failed |

- **RL-Zero-Mix failed:** its config has model type `olmo2-retrofit`, which the installed Transformers/vLLM doesn't recognize. Recommendation: skip it (RL-Zero-General works), and note this in EXPERIMENTS.md. Don't upgrade the environment for this.
- **Base is slow.** It rambles to the token limit. The 4 h limit (`submit_llama.sh` uses `--time=04:00:00`) will probably cut off at least the Phenopacket base run [A].
- **To resume a base run** (the sampler skips cases already in the output file), run on Tillicum:

```bash
source ~/.thesis_env; cd $THESIS; sbatch -p gpu-h200 --time=10:00:00 --array=0-1 --job-name=m-meas_full-olmo3-base code/slurm/tillicum/sample.sbatch --model allenai/Olmo-3-1025-7B --chat-template code/templates/olmo3_sft.jinja --out "runs/meas_full/olmo3-base/samples.s{shard}.jsonl" --cases data/phenopackets/cases.jsonl --case-ids data/phenopackets/splits/eval_full.txt --k 32 --shard -1 --num-shards 2 --max-tokens 4096 --max-model-len 6144
```

  For the DDXPlus base runs, use `--task ddxplus --conditions data/ddxplus/conditions.json --cases data/ddxplus/cases_test.jsonl` with `--out runs/meas_ddx{,_t1}/olmo3-base/...` and add `--temperature 1.0 --top-p 1.0` for T=1.
- **Smoke test [V]:** OLMo 3 loads in vLLM (`Olmo3ForCausalLM`). Instruct output is clean JSON. Base output is messy: refusals, plain lists, rambling. Expect a high invalid rate; `src/format_robust.py` handles this.

**After sampling finishes:**
1. **Relay.** Use `slurm/relay_olmo_laptop.sh <scratchpad>` (it was written for job IDs 383434–383451; edit the wait regex if you resubmitted), or relay by hand. Mapping: Tillicum `meas_ddx` → Klone `runs/ddx_olmo`, `meas_ddx_t1` → `runs/ddx_olmo_t1`, `meas_full` → `runs/go_full_olmo`. Copy only `olmo3-*/samples.s*.jsonl`.
2. **Score and analyze on Klone:** `sbatch -A stf -p ckpt-all -c 16 --mem=96G --time=06:00:00 -J olmo-score -o $THESIS/logs/%x-%j.out $THESIS/code/slurm/olmo_klone.sh`. It scores all three dirs (BASE=olmo3-base) and prints:
   - the pass@k reports;
   - `format_robust` for each stage vs base;
   - stage-to-stage paired comparisons (SFT vs base, DPO vs SFT, Instruct vs DPO);
   - severity tiers (`runs/analysis/pc_olmo`).
   - It was written but **not yet run or tested**. Check it for path bugs first (it uses `bash slurm/score_ddx.sbatch` with `RUNS`/`BASE` env vars).
3. Add `ddx_olmo`, `ddx_olmo_t1` and `go_full_olmo` to `COMPARISONS` in `src/collect_tables.py`, then regenerate `results/paper_numbers.json`.

## 7. Paper: state and next edits

- **Sections:** Intro, Related work, Setup, Results (`sec:general`, `sec:collapse`, `sec:robust`, `sec:medical`, `sec:rare`), Training (`sec:training`), Limitations.
- **Figures:** vector PDFs from `src/paper_figures.py` with bootstrap bands: `fig_passk_pairs`, `fig_passk_medical`, `fig_passk_grpo8b`, `fig_percond`.
- **One `\todo` left [V],** in Limitations: "Llama-3.1-8B base/instruct pair (pending access)". **Replace it with the OLMo 3 results.**
- **Planned edits once OLMo is scored:**
  1. Add a subsection, e.g. "Which post-training stage collapses the differential?", with a table: base / SFT / DPO / Instruct (RLVR) / RL-Zero. Columns: pass@1, pass@32, distinct, entropy, cov@32, severe-tier Considered@32, Phenopacket confidence.
  2. If RL-Zero keeps base diversity, it is the controlled counterpart of AlphaMed base-RL and our GRPO-from-base. Strengthen the claim "collapse comes from instruction tuning, not RL" accordingly. If it doesn't, soften the claim.
  3. Update the abstract, Setup (ii) and the findings list to cover two families (Qwen and OLMo).
  4. Move the Llama medical descendants to an appendix, or drop them.
  5. Add an OLMo 3 citation to `refs.bib`. **Verify the citation details; don't invent them.**
- **Remaining polish:** related-work pass; check that `refs.bib` entries are all verified (they were at commit 7f50339); keep figures in sync.

## 8. Key code (all in `src/`)

**Sampling and scoring:**
- `sample_diagnoses.py`: vLLM sampling. Main flags:
  - `--task ddxplus|phenopacket`, `--shard -1` (uses the Slurm array index), `--out ...{shard}...`;
  - `--prompt-variant default|plain|cot`;
  - `--chat-template FILE`.
  - It stops on `<|im_end|>`, `<|eot_id|>` and `<end_of_turn>`.
- `parsing.py` (`parse_dx`, now RecursionError-safe), `ddx_names.py` (`Matcher`), `disease_normalizer.py` (Mondo grading).
- `score_ddxplus.py` (now writes `n_parsed`; "parsed" means a valid on-list top-1), `score_samples.py` (Phenopackets), `compare_runs.py` (reports with paired CIs).

**Analysis:**
- `compare_recipes.py` (paired bootstrap, averaging over seeds), `format_robust.py`, `per_condition.py`, `era_table.py`, `collect_tables.py`, `paper_figures.py`, `summarize_e7.py`.

**Training:**
- `rewards.py` (including `group_match_rewards` and the Puri baselines `multi_rlvr` / `set_brier`), `train_grpo.py`.

**Slurm scripts:**
- Klone (`slurm/`): `sample.sbatch`, `score_ddx.sbatch` (env `RUNS`, `BASE`), `score_go.sh`, `score_new.sbatch` (env `SPEC="dir:base:tag1+tag2 ..."`; **tags are joined with `+` because `sbatch --export` splits on commas**), `compare_seed1.sh`, `fullpheno_klone.sh`, `olmo_klone.sh`, `scrontab.txt`.
- Tillicum (`slurm/tillicum/`): `sample.sbatch` (pass `-p gpu-h200` for full GPUs; 7–8B models don't fit MIG slices), `train.sbatch`, `submit_train_eval.sh`, and `submit_llama.sh` with modes `descendants | pair | prompts | fullpheno | olmo`.

## 9. Gotchas learned the hard way

- **sbatch:** `--export=ALL,X=a,b` splits on commas; join lists with `+`. An sbatch "socket timed out" may still have submitted the job, so check `squeue` before resubmitting.
- **Klone queues:** forbids multi-partition jobs. `gpu-l40s` CPU jobs can hit `AssocGrpCpuLimit` (use `ckpt-all`). `ckpt-all` can be slow to start.
- **Scorer selection:** `score_new.sbatch` picks the scorer by matching `*ddx*` in the directory name; keep DDXPlus dirs named with "ddx".
- **`score_ddx.sbatch` rebuilds `samples.jsonl` from `samples.s*.jsonl`.** Make sure shard files exist before rescoring.
- **Don't** run heavy work on Klone login nodes. Even importing the vLLM registry there hangs because of GPFS.
- **Don't** `pkill -f` patterns that match your own ssh command.
- **vLLM env:** `VLLM_USE_FLASHINFER_SAMPLER=0` and `CUDA_HOME` set to the env's `nvidia/cu13`. On Tillicum also `VLLM_WORKER_MULTIPROC_METHOD=spawn` and `__main__` guards. These are already in the sbatch scripts.
- **arXiv is blocked in the in-app browser by policy.** WebFetch works.
- **Background watchers** die when the session ends; this happened to the OLMo watcher. Re-check state rather than trusting old task output.

## 10. Unresolved or open questions

- **OLMo base completion and cost:** $65.68 spent at 12:13, up from $50.78 before OLMo, and the base runs are still going [V]. Expect roughly $70–80 total after resubmits [A].
- **Llama access:** the Meta request was still pending as of 10-23. If it's approved later, `submit_llama.sh pair` is ready; it needs the user to run `huggingface-cli login` *on Tillicum*.
- **Whether the user changed their NetID password** after the 10-20 paste: unknown.
- **Llama medical descendants:** move to an appendix or drop? User not yet asked; recommended: appendix.
