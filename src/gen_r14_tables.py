"""Appendix tables from one pipeline (r14_an.py, r14_pheno.py): tab:pairs, tab:severity, tab:prompts, tab:stages,
tab:big32, tab:fullprompt, tab:robust, tab:medical, tab:era. Writes one .tex file per table into OUT.

    python gen_r14_tables.py R14_DIR OUT_DIR
"""
import glob
import json
import sys
from pathlib import Path

D, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
M, P, G = {}, {}, {}
for f in glob.glob(f"{D}/r14_*.json"):
    if "pheno" in f:
        continue
    d = json.load(open(f))
    M.update(d["models"]); P.update(d["pairs"]); G.update(d.get("auroc_gap", {}))
PH = json.load(open(f"{D}/r14_pheno.json"))
LOWER = {"tv_samples", "tv_list", "ece", "tv_complete", "tv_right", "tv_wrong", "excess_list", "excess_samples", "sev_low_share"}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def f3(x):
    return f"{x:.3f}"


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


def sig_bold(rows, metrics):
    """rows: [(pair_key)]; returns {(pair_key, metric): 'base'|'post'|None} after Holm over all tests of the table"""
    tests = [(k, m) for k in rows for m in metrics if m in P[k]]
    adj = holm([P[k][m]["p"] for k, m in tests])
    out = {}
    for (k, m), p in zip(tests, adj):
        d = P[k][m]["diff"]
        better_post = (d < 0) if m in LOWER else (d > 0)
        out[(k, m)] = ("post" if better_post else "base") if p < 0.05 else None
    return out


def write(name, text):
    (OUT / f"{name}.tex").write_text(text + "\n")


def ci(v, sc=100, fmt="{:+.1f}"):
    lo, hi = v["cluster_ci"]
    return f"${fmt.format(sc * v['diff'])}$ [${fmt.format(sc * lo)},{fmt.format(sc * hi)}$]"


# ---------------- tab:pairs
rows = [("ddx_qwen3_t1", "qwen3-8b-base", "qwen3-8b", "Qwen3-8B, $T{=}1$", "post-trained"),
        ("pair25_t1", "qwen25-7b-base", "qwen25-7b-it", "Qwen2.5-7B, $T{=}1$", "instruct"),
        ("pair_q3", "qwen3-8b-base", "qwen3-8b", "Qwen3-8B, $T{=}0.7$", "post-trained")]
cols = ["pass1", "pass32", "distinct", "entropy", "tv_samples", "cov32"]
keys = [f"{d}:{b}>{p}" for d, b, p, _, _ in rows]
sb = sig_bold(keys, cols)
body = []
for (d, b, p, lab, plab), k in zip(rows, keys):
    for who, t, nm in [("base", b, "base"), ("post", p, plab)]:
        m = M[f"{d}/{t}"]
        cells = []
        for c in cols:
            v = f"{m[c]:.2f}" if c in ("distinct", "entropy") else f3(m[c])
            cells.append(bold(v, sb.get((k, c)) == who))
        other = M[f"{d}/{p if who == 'base' else b}"]["ece"]
        cells.append(bold(f3(m["ece"]), m["ece"] < other))
        body.append((f"\\multirow{{2}}{{*}}{{{lab}}}" if who == "base" else "") + f" & {nm} & " + " & ".join(cells) + " \\\\")
    body.append("\\midrule")
write("tab_pairs", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llccccccc}
\toprule
Pair & Model & $\passk{1}$ & $\passk{32}$ & Distinct & Entropy & TV$\downarrow$ & Cov@32 & ECE$\downarrow$ \\
\midrule
""" + "\n".join(body[:-1]) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{DDXPlus, 32 samples per case (980 cases). The reference differential has mean entropy 1.87. Distinct, entropy and TV refer to the top-1 answers across samples; Cov@32 is the share of reference probability mass covered by those answers. ECE is for the stated top-1 probability, over samples whose top answer is one of the 49 conditions. Bold: significantly better model (condition-clustered bootstrap, Holm across the table's 18 tests); for ECE, which is not tested, the lower value.}
\label{tab:pairs}
\end{table}""")

# ---------------- tab:severity
srows = [("ddx_qwen3_t1:qwen3-8b-base>qwen3-8b", "Qwen3-8B, $T{=}1$"), ("pair25_t1:qwen25-7b-base>qwen25-7b-it", "Qwen2.5-7B, $T{=}1$"),
         ("pair_q3:qwen3-8b-base>qwen3-8b", "Qwen3-8B, $T{=}0.7$"), ("pair25_t07:qwen25-7b-base>qwen25-7b-it", "Qwen2.5-7B, $T{=}0.7$"), None,
         ("ddx:qwen25-7b-it>alphamed-7b", "Instruct $\\rightarrow$ AlphaMed, $T{=}0.7$"), ("ddx:qwen25-7b-it>huatuo-o1-7b", "Instruct $\\rightarrow$ HuatuoGPT-o1, $T{=}0.7$"),
         ("ddx:qwen25-7b-it>m1-7b", "Instruct $\\rightarrow$ m1, $T{=}0.7$"), ("ddx_base_rl:qwen25-7b-base>alphamed-7b-base-rl", "Base $\\rightarrow$ AlphaMed (base-RL), $T{=}0.7$"), None,
         ("e9_ddx:qwen3-8b-base>b8-acc-s0", "Base $\\rightarrow$ \\texttt{acc} GRPO (8B), $T{=}0.7$"), ("e9_ddx:qwen3-8b-base>b8-gm-s0", "Base $\\rightarrow$ \\texttt{gm} GRPO (8B), $T{=}0.7$")]
body = []
for r in srows:
    if r is None:
        body.append("\\midrule")
        continue
    k, lab = r
    d, bp = k.split(":")
    b, p = bp.split(">")
    lv = [f"{M[f'{d}/{b}'][f'cons_{t}']:.2f} $\\rightarrow$ {M[f'{d}/{p}'][f'cons_{t}']:.2f}" for t in ["severe", "moderate", "mild"]]
    df = [f"\\footnotesize{{{ci(P[k][f'cons_{t}'])}}}" for t in ["severe", "moderate", "mild"]]
    body.append(f"{lab} & " + " & ".join(lv) + " \\\\")
    body.append(" & " + " & ".join(df) + " \\\\")
write("tab_severity", r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccc}
\toprule
 & \multicolumn{3}{c}{Considered@32 (base $\rightarrow$ post-trained)} \\
Comparison & Severe (1--2) & Moderate (3) & Mild (4--5) \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Share of (case, condition) pairs with reference probability $\geq 0.1$ in which the condition is a top-1 answer in at least one of 32 samples, by DDXPlus severity. Second rows give differences in points with 95\% condition-clustered intervals. All rows use the 980 test cases; GRPO rows are seed 0. General post-training lowers consideration in every tier, including the severity-1--2 tier; medical post-training does not lower it (HuatuoGPT-o1 and m1 raise it in every tier); our GRPO from the base model loses at most 4.3 points.}
\label{tab:severity}
\end{table}""")

# ---------------- tab:prompts
prow = [("pair_q3", "default", "0.7"), ("ddx_qwen3_t1", "default", "1.0"), ("pv_plain_t07", "plain", "0.7"), ("pv_plain_t1", "plain", "1.0"),
        ("pv_cot_t07", "CoT", "0.7"), ("pv_cot_t1", "CoT", "1.0")]
keys = [f"{d}:qwen3-8b-base>qwen3-8b" for d, _, _ in prow]
sb = sig_bold(keys, ["pass1", "pass32", "distinct"])
body = []
for (d, nm, t), k in zip(prow, keys):
    b, p = M[f"{d}/qwen3-8b-base"], M[f"{d}/qwen3-8b"]
    cells = []
    for c, fm in [("pass1", f3), ("pass32", f3), ("distinct", lambda x: f"{x:.2f}")]:
        cells += [bold(fm(b[c]), sb.get((k, c)) == "base"), bold(fm(p[c]), sb.get((k, c)) == "post")]
    body.append(f"{nm} & {t} & " + " & ".join(cells) + " \\\\")
write("tab_prompts", r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{2}{c}{pass@1} & \multicolumn{2}{c}{pass@32} & \multicolumn{2}{c}{Distinct} \\
Prompt & $T$ & Base & Post & Base & Post & Base & Post \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Qwen3-8B-Base vs.\ Qwen3-8B on DDXPlus (980 cases, 32 samples) under three prompts. Bold marks the significantly better model (condition-clustered bootstrap, Holm across the table's 18 tests). Without reasoning, the base model wins at $k{=}32$; with CoT, the post-trained model matches or beats it, but its samples stay far less diverse.}
\label{tab:prompts}
\end{table}""")

# ---------------- tab:stages
STG = [("olmo3-base", "Base"), ("olmo3-sft", "SFT"), ("olmo3-dpo", "SFT+DPO"), ("olmo3-instruct", "SFT+DPO+RLVR (Instruct)"), ("olmo3-rlzero-general", "RL-Zero (RL from base)")]
cols = [("pass1", f3), ("pass32", f3), ("distinct", lambda x: f"{x:.2f}"), ("entropy", lambda x: f"{x:.2f}"), ("cov32", f3), ("cons_severe", f3)]
body = []
for d, T in [("ddx_olmo", "0.7"), ("ddx_olmo_t1", "1.0")]:
    best = {c: max(M[f"{d}/{t}"][c] for t, _ in STG) for c, _ in cols}
    for i, (t, nm) in enumerate(STG):
        m = M[f"{d}/{t}"]
        body.append(f"{T if i == 0 else ''} & {nm} & " + " & ".join(bold(fm(m[c]), m[c] == best[c]) for c, fm in cols) + " \\\\")
    body.append("\\midrule")
write("tab_stages", r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccccc}
\toprule
$T$ & Stage & pass@1 & pass@32 & Distinct & Entropy & Cov@32 & Severe C@32 \\
\midrule
""" + "\n".join(body[:-1]) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{OLMo~3 7B post-training stages on DDXPlus (980 cases, 32 samples). Severe C@32 is severe-tier Considered@32 as in Table~\ref{tab:severity}. Bold marks the best value in each block. Diversity falls at every stage of the instruct pipeline, and the RLVR stage loses the most coverage; RL from the base keeps the base model's pass@32 at $T{=}1$.}
\label{tab:stages}
\end{table}""")

# ---------------- tab:big32
fam = [[("qwen25-32b-base", "Qwen2.5-32B (base)"), ("qwen25-32b-it", "Qwen2.5-32B-Instruct")],
       [("olmo31-32b-base", "OLMo-3-1125-32B (base)"), ("olmo31-32b-sft", "\\quad SFT"), ("olmo31-32b-dpo", "\\quad SFT+DPO"),
        ("olmo31-32b-instruct", "\\quad SFT+DPO+RLVR (Instruct)")]]
cols = [("pass1", f3, max), ("pass32", f3, max), ("distinct", lambda x: f"{x:.2f}", max), ("cov32", f3, max),
        ("cons_severe", lambda x: f"{x:.2f}", max), ("tv_list", f3, min), ("tv_samples", f3, min)]
body = []
for F in fam:
    ms = [M[f"big32_ddx/{t}"] for t, _ in F]
    best = {c: agg(m[c] for m in ms) for c, _, agg in cols}
    for (t, nm), m in zip(F, ms):
        body.append(f"{nm} & " + " & ".join(bold(fm(m[c]), m[c] == best[c]) for c, fm, _ in cols) + f" & ${100 * m['vote_gain']:+.1f}$ \\\\")
    body.append("\\midrule")
write("tab_big32", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccccc}
\toprule
Model & pass@1 & pass@32 & Distinct & Cov@32 & Severe C@32 & TV one list$\downarrow$ & TV samples$\downarrow$ & Vote@32 gain \\
\midrule
""" + "\n".join(body[:-1]) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{32B replication on DDXPlus ($T{=}0.7$, 980 cases, 32 samples). TV columns: distance to the reference differential of one stated list and of the sampled top-1 answers (\S\ref{sec:stated}). Vote@32 gain: accuracy of the majority answer over 32 samples minus single-call accuracy, in points (\S\ref{sec:consequences}). Bold: best value per family.}
\label{tab:big32}
\end{table}""")

# ---------------- tab:fullprompt
FP = [("pv_full_t07", "qwen3-8b-base", "Qwen3-8B-Base"), ("pv_full_t07", "qwen3-8b", "Qwen3-8B"), ("pv_full_t07", "qwen25-7b-base", "Qwen2.5-7B"),
      ("pv_full_t07", "qwen25-7b-it", "Qwen2.5-7B-Instruct"), ("@pv_full_t07", "olmo3-base", "OLMo-3-1025-7B"), ("@pv_full_t07", "olmo3-instruct", "OLMo-3-7B-Instruct")]
body = []
for i in range(0, 6, 2):
    pair = [M[f"{d}/{t}"] for d, t, _ in FP[i:i + 2]]
    for j, (d, t, nm) in enumerate(FP[i:i + 2]):
        m, o = pair[j], pair[1 - j]
        sig = {s: m[f"auroc_{s}"] for s in ["stated_p", "agree5", "agree32"]}
        bs = max(sig.values())
        body.append(f"{nm} & {bold(f3(m['tv_list']), m['tv_list'] < o['tv_list'])} & {bold(f3(m['tv_samples']), m['tv_samples'] < o['tv_samples'])} & "
                    + " & ".join(bold(f3(v), v == bs) for v in sig.values()) + f" & ${100 * m['top3_list_minus_samples']:+.1f}$ \\\\")
write("tab_fullprompt", r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{4pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{TV to reference$\downarrow$} & \multicolumn{3}{c}{AUROC, single-call answer correct} & Top-3: one \\
Model & One distribution & Samples & Stated $p$ & Agree@5 & Agree@32 & dist.\ $-$ 32 samples \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Full-distribution prompt on DDXPlus ($T{=}0.7$, 980 cases): a probability on every condition, top answer = highest stated probability. TV: one sample's stated distribution vs.\ the sampled top-1 answers (bold: better model of each pair). AUROC: bold marks each model's best signal.}
\label{tab:fullprompt}
\end{table}""")

# ---------------- tab:robust
RB = [[("pair_q3/qwen3-8b-base", "Qwen3-8B-Base"), ("pair_q3/qwen3-8b", "Qwen3-8B")],
      [("pair25_t07/qwen25-7b-base", "Qwen2.5-7B"), ("pair25_t07/qwen25-7b-it", "Qwen2.5-7B-Instruct")],
      [("big32_ddx/qwen25-32b-base", "Qwen2.5-32B"), ("big32_ddx/qwen25-32b-it", "Qwen2.5-32B-Instruct")], None,
      [("ddx_olmo/olmo3-base", "OLMo-3-1025-7B"), ("ddx_olmo/olmo3-instruct", "OLMo-3-7B-Instruct")],
      [("big32_ddx/olmo31-32b-base", "OLMo-3-1125-32B"), ("big32_ddx/olmo31-32b-instruct", "OLMo-3.1-32B-Instruct")]]
cols = ["tv_complete", "tv_right", "tv_wrong", "excess_list", "excess_samples", "sev_low_share"]
body = []
for pr in RB:
    if pr is None:
        body.append("\\midrule")
        continue
    a, b = M[pr[0][0]], M[pr[1][0]]
    for (k, nm), m, o in [(pr[0], a, b), (pr[1], b, a)]:
        cells = [f"{m['complete']:.2f}"]
        for c in cols:
            better = m[c] < o[c] if c in LOWER else m[c] > o[c]
            cells.append(bold(f3(m[c]), better))
        body.append(f"{nm} & " + " & ".join(cells) + " \\\\")
write("tab_robust", r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccc}
\toprule
 & Complete & \multicolumn{3}{c}{TV of one list$\downarrow$} & \multicolumn{2}{c}{Excess TV over floor$\downarrow$} & Severe named \\
Model ($T{=}0.7$) & lists & complete only & top-1 right & top-1 wrong & One list & Samples & ref.\ $<0.1$ \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Robustness of the stated-list result (DDXPlus, $T{=}0.7$, 980 cases, valid samples). Complete lists: share naming five on-list conditions with valid probabilities. TV of one list on complete lists only, and on samples whose top answer is right or wrong. Excess TV: TV minus the TV of the matching oracle (the reference's top five for one list; the expected TV of as many reference draws as the model has valid samples, for the samples). Severe named, ref.\ $<0.1$: among lists naming at least one severe condition, the share of those conditions that the reference gives less than 0.1 (over-naming). Bold: better model of each pair. Excess TV is averaged over samples against each case's floor, so it is not exactly the difference of the mean TV and the mean floor. pass@32 with top answers drawn from the stated lists is in Table~\ref{tab:sampled}.}
\label{tab:robust}
\end{table}""")

# ---------------- tab:medical
MD = [[("ddx/qwen25-7b-it", "Qwen2.5-7B-Instruct", "(parent)"), ("ddx/alphamed-7b", "AlphaMed-7B", "rule-based RL"),
       ("ddx/huatuo-o1-7b", "HuatuoGPT-o1-7B", "SFT + PPO"), ("ddx/m1-7b", "m1-7B-23K", "SFT distillation")],
      [("ddx_base_rl/qwen25-7b-base", "Qwen2.5-7B (base)", "(parent)"), ("ddx_base_rl/alphamed-7b-base-rl", "AlphaMed-7B (base-RL)", "rule-based RL")]]
cols = [("pass1", f3, max), ("pass32", f3, max), ("distinct", lambda x: f"{x:.2f}", max), ("entropy", lambda x: f"{x:.2f}", max), ("cov32", f3, max), ("ece", f3, min)]
allm = [M[k] for blk in MD for k, _, _ in blk]
best = {c: agg(m[c] for m in allm) for c, _, agg in cols}
body = []
for blk in MD:
    for k, nm, how in blk:
        m = M[k]
        body.append(f"{nm} & {how} & " + " & ".join(bold(fm(m[c]), m[c] == best[c]) for c, fm, _ in cols) + " \\\\")
    body.append("\\midrule")
write("tab_medical", r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccccc}
\toprule
Model & Post-training & pass@1 & pass@32 & Distinct & Entropy & Cov@32 & ECE \\
\midrule
""" + "\n".join(body[:-1]) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Medical models on DDXPlus (980 cases, 32 samples, $T{=}0.7$). Upper block: descendants of Qwen2.5-7B-Instruct; none narrows the differential further, and the reasoning-trained models widen it. Lower block: AlphaMed's companion checkpoint trained from the base model, which keeps most of the base model's diversity and has the best zero-shot pass@32 of any 7B model. Bold: best value over all six models.}
\label{tab:medical}
\end{table}""")

# ---------------- tab:era
ER = [("qwen25-7b-base", "Qwen2.5-7B (base)"), ("qwen25-7b-it", "Qwen2.5-7B-Instruct"), ("alphamed-7b", "AlphaMed-7B"),
      ("alphamed-7b-base-rl", "AlphaMed-7B (base-RL)"), ("huatuo-o1-7b", "HuatuoGPT-o1-7B"), ("m1-7b", "m1-7B"),
      ("qwen3-8b-base", "Qwen3-8B-Base"), ("qwen3-8b", "Qwen3-8B")]
body = []
for t, nm in ER:
    m = PH["models"][f"go_full/{t}"]
    body.append(f"{nm} & " + " & ".join(f"{100 * m[f'era_{e}'][k]:.1f}" for e in ["<=2022", "2023-24", "2025+"] for k in ["pass1", "pass32"]) + " \\\\")
n = {e: PH["models"]["go_full/qwen3-8b"][f"era_{e}"]["n"] for e in ["<=2022", "2023-24", "2025+"]}
write("tab_era", r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
""" + f" & \\multicolumn{{2}}{{c}}{{$\\le$2022 ($n{{=}}{n['<=2022']}$)}} & \\multicolumn{{2}}{{c}}{{2023--24 ($n{{=}}{n['2023-24']}$)}} & \\multicolumn{{2}}{{c}}{{2025+ ($n{{=}}{n['2025+']}$)}} \\\\" + r"""
Model & pass@1 & pass@32 & pass@1 & pass@32 & pass@1 & pass@32 \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Phenopacket Store accuracy (\%, exact diagnosis) by publication year of the source case report (1,401 cases, 32 samples, $T{=}0.7$). Every model is far more accurate on cases published up to 2022 than on 2023--24 cases; 2025+ cases fall in between.}
\label{tab:era}
\end{table}""")
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
