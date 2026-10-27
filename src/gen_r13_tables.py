"""Tables from r13_an.py: pooled effects by joint cluster bootstrap (tab:meta) and per-pair checks (tab:r13pairs)."""
import json
import sys
D = json.load(open(sys.argv[1]))
LAB = [("pass32", "pass@32 (points)", 100), ("distinct", "distinct answers", 1), ("rec5", "recall@5, one list (points)", 100),
       ("tv_list", "TV, one list", 1), ("cond_tv", "conditional TV", 1), ("sev_named", "severe named (points)", 100),
       ("auroc_agree32", "AUROC, agreement", 1)]
SETS = ["zero-shot", "worked examples", "examples from another model"]


def f(m, sc, x):
    return ("{:+.1f}" if sc == 100 else ("{:+.2f}" if m == "distinct" else "{:+.3f}")).format(sc * x)


def cell(m, sc, est, ci):
    sig = "" if ci[0] > 0 or ci[1] < 0 else r"$^{\dagger}$"
    return f"${f(m, sc, est)}$ [{f(m, sc, ci[0])}, {f(m, sc, ci[1])}]{sig}"


rows = []
for m, lab, sc in LAB:
    cells = []
    for s in SETS:
        v = D[s]["pooled"][m]
        cells += [cell(m, sc, v["pooled"], v["pooled_ci"]), cell(m, sc, v["qwen_minus_other"], v["contrast_ci"])]
    rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
T1 = r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Zero-shot (5 pairs)} & \multicolumn{2}{c}{Worked examples (6)} & \multicolumn{2}{c}{Another model's examples (6)} \\
Post-trained $-$ base & pooled & Qwen $-$ other & pooled & Qwen $-$ other & pooled & Qwen $-$ other \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Pooled base $\rightarrow$ post-trained effects across pairs (DDXPlus, $T{=}0.7$). \emph{Pooled}: mean of the pairs' differences; \emph{Qwen $-$ other}: mean over the Qwen pairs minus the mean over the OLMo (and Mistral) pairs. 95\% intervals come from a joint bootstrap that draws the 49 pathologies once per resample and applies the draw to every pair, so the correlation between pairs scored on the same cases is kept (2,000 resamples). $^{\dagger}$: interval includes zero.}
\label{tab:meta}
\end{table}"""
NAMES = {"qwen3-8b-base": "Qwen3-8B", "qwen25-7b-base": "Qwen2.5-7B", "qwen25-32b-base": "Qwen2.5-32B", "olmo3-base": "OLMo~3 7B",
         "olmo31-32b-base": "OLMo~3.1 32B", "mistral7b-base": "Mistral-7B"}
rows = []
for s, short in [("zero-shot", "zero-shot"), ("worked examples", "examples")]:
    for t, r in D[s]["pairs"].items():
        b, p = r["models"]["base"], r["models"]["post"]
        sb, sp = b["selective"], p["selective"]
        rows.append(f"{NAMES[t]}, {short} & {b['ent']:.2f} $\\rightarrow$ {p['ent']:.2f} & {b['miss_severe_true']:.2f} $\\rightarrow$ {p['miss_severe_true']:.2f} & "
                    f"{b['miss_all_warranted']:.2f} $\\rightarrow$ {p['miss_all_warranted']:.2f} & {sb['acc@50_stated_p']:.2f} / {sb['acc@50_agree32']:.2f} & "
                    f"{sp['acc@50_stated_p']:.2f} / {sp['acc@50_agree32']:.2f} & {b['auroc_agree32_nonunanimous_both']:.2f} $\\rightarrow$ {p['auroc_agree32_nonunanimous_both']:.2f} ({r['nonunanimous_both_n']}) \\\\")
    if s == "zero-shot":
        rows.append(r"\midrule")
T2 = r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
 & Entropy of & \multicolumn{2}{c}{Lists that miss (share)} & \multicolumn{2}{c}{Accuracy at 50\% coverage, stated $p$ / agreement} & Agreement AUROC, \\
Pair & one list (nats) & severe true dx & all warranted severe & base & post-trained & both non-unanimous ($n$) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Further checks (DDXPlus, $T{=}0.7$, base $\rightarrow$ post-trained). \emph{Entropy}: of one list's renormalized probabilities, per case mean. \emph{Lists that miss}: share of one model's lists that do not contain a severe true pathology among their five items (cases whose true pathology has severity 1--2), and that name none of the severe conditions the reference weights at least 0.1 (cases with at least one). \emph{Accuracy at 50\% coverage}: accuracy of the single-call answer on the half of cases ranked most confident by the stated probability or by 31-sample agreement. \emph{Agreement AUROC}: on the cases where neither model's valid samples are unanimous, so both models are compared on the same cases and agreement is not tied.}
\label{tab:r13pairs}
\end{table}"""
print(T1 + "\n\n" + T2)
