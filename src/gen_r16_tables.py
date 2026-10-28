"""Table for breadth within one list versus across calls (tab:breadth). Reads r16.json (r16_breadth.py).

    python gen_r16_tables.py r16.json > tab_breadth.tex
"""
import json
import sys

R = json.load(open(sys.argv[1]))
PAIRS = [("pair_q3", "qwen3-8b-base", "qwen3-8b", "Qwen3-8B"), ("pair25_t07", "qwen25-7b-base", "qwen25-7b-it", "Qwen2.5-7B"),
         ("big32_ddx", "qwen25-32b-base", "qwen25-32b-it", "Qwen2.5-32B"), ("ddx_olmo", "olmo3-base", "olmo3-instruct", "OLMo~3 7B"),
         ("big32_ddx", "olmo31-32b-base", "olmo31-32b-instruct", "OLMo~3.1 32B")]
DAG = r"$^{\dagger}$"
rows = []
for d, b, p, nm in PAIRS:
    pr = R["pairs"][f"{d}:{b}>{p}"]
    for lab, t in [("base", b), ("post-trained", p)]:
        m = R["models"][f"{d}/{t}"]
        def cell(k, f):
            s = f.format(m[k])
            if lab == "post-trained" and k in pr:
                lo, hi = pr[k]["cluster_ci"]
                s += "" if (lo > 0 or hi < 0) else DAG
            return s
        rows.append(f"{nm if lab == 'base' else ''} & {lab} & {cell('list_union_1', '{:.2f}')} & {cell('list_mass_1', '{:.2f}')} & {cell('warranted_1', '{:.2f}')} & "
                    f"{cell('list_union_32', '{:.1f}')} & {cell('list_mass_32', '{:.2f}')} & {cell('warranted_32', '{:.2f}')} & "
                    f"{cell('samp_distinct_32', '{:.2f}')} & {cell('samp_mass_32', '{:.2f}')} \\\\")
print(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccccccc}
\toprule
 & & \multicolumn{3}{c}{One stated list} & \multicolumn{3}{c}{32 stated lists together} & \multicolumn{2}{c}{32 sampled top answers} \\
Pair & Model & Conditions & Mass & Warranted & Conditions & Mass & Warranted & Distinct & Mass \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Breadth within one call and across calls (DDXPlus, zero-shot, $T{=}0.7$, 980 cases). \emph{Conditions}: conditions named (one list; or the union of the lists of up to 32 valid samples); \emph{mass}: reference probability of the named conditions; \emph{warranted}: share of the conditions with reference weight $\geq 0.1$ that are named. Sampled top answers: distinct answers and the reference mass they cover (Cov@32). Base models with fewer than 32 valid samples are scored on the valid ones, which understates their breadth across calls. $^{\dagger}$: post-trained minus base not significant (condition-clustered 95\% interval includes zero); every other difference is.}
\label{tab:breadth}
\end{table}""")
