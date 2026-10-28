"""Chance-corrected pass@32 table (tab:chance) from r18_chance.py output.

    python gen_r18_tables.py r18.json > tab_chance.tex
"""
import json
import sys

R = json.load(open(sys.argv[1]))
NAMES = {"Qwen3-8B": "Qwen3-8B", "Qwen2.5-7B": "Qwen2.5-7B", "Qwen2.5-32B": "Qwen2.5-32B", "OLMo 3 7B": "OLMo~3 7B",
         "OLMo 3.1 32B": "OLMo~3.1 32B", "Mistral-7B": "Mistral-7B", "Qwen2.5-72B": "Qwen2.5-72B$^{*}$"}
LAB = {"zs": "zero-shot", "fs": "reference examples", "rf": "another model's examples"}


def cell(v):
    lo, hi = v["cluster_ci"]
    return f"${100 * v['diff']:+.1f}$" + ("" if (lo > 0 or hi < 0) else "$^{\\dagger}$")


rows = []
for k, nm in NAMES.items():
    r = R["pairs"][k]
    cells = []
    for st in ["zs", "fs", "rf", "best"]:
        cells += [cell(r[st]["raw"]), cell(r[st]["corrected"])] if st in r else ["---", "---"]
    bs = r["best_setting"]
    best = " / ".join(LAB[bs[t]].split()[0] for t in bs)
    rows.append(f"{nm} & " + " & ".join(cells) + f" & {best} \\\\")
print(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccl}
\toprule
 & \multicolumn{2}{c}{Zero-shot} & \multicolumn{2}{c}{Reference examples} & \multicolumn{2}{c}{Another model's examples} & \multicolumn{2}{c}{Each at its best prompt} & Best prompt \\
Pair & raw & corrected & raw & corrected & raw & corrected & raw & corrected & (base / post) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Chance-corrected pass@32 (DDXPlus, $T{=}0.7$, 980 cases): post-trained minus base, in points. \emph{Raw}: pass@32; \emph{corrected}: pass@32 minus the pass@32 reached when each case's samples are scored against the true pathology of a random case with a different pathology (mean of 20 permutations), the part that does not depend on the patient (base models 0.05--0.19, post-trained 0.01--0.04). \emph{Each at its best prompt}: each model at the prompt where its own pass@32 is highest. Top answers and validity as in every other table. $^{\dagger}$: condition-clustered 95\% interval includes zero. Mistral's base model does not answer the zero-shot prompt (\S\ref{sec:mistral}). $^{*}$: added after the headline tests were fixed; run zero-shot and with reference examples only (\S\ref{sec:big32}).}
\label{tab:chance}
\end{table}""")
