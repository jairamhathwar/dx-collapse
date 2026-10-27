"""Table of pooled effects (tab:meta) from meta_tost.py output."""
import json
import sys
M = json.load(open(sys.argv[1]))["meta"]
LAB = [("pass32", "pass@32 (points)", 100), ("distinct", "distinct answers", 1), ("rec5", "recall@5, one list (points)", 100),
       ("tv_list", "TV, one list", 1), ("cond_tv", "conditional TV", 1), ("sev_named", "severe named (points)", 100),
       ("auroc_agree32", "AUROC, agreement", 1), ("auroc_stated", "AUROC, stated $p$", 1)]
SETS = ["zero-shot", "worked examples", "examples from another model"]
rows = []
for m, lab, sc in LAB:
    cells = []
    for s in SETS:
        r = M[s][m]; p = r["pooled"]
        f = "{:+.1f}" if sc == 100 else ("{:+.2f}" if m == "distinct" else "{:+.3f}")
        cells.append(f"${f.format(sc * p['est'])}$ [{f.format(sc * p['ci'][0])}, {f.format(sc * p['ci'][1])}]")
        cells.append(f"{r['p_between']:.3f}")
    rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
print(r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{Zero-shot (5 pairs)} & \multicolumn{2}{c}{Worked examples (6)} & \multicolumn{2}{c}{Another model's examples (6)} \\
Post-trained $-$ base & pooled [95\% CI] & $p$, Qwen vs other & pooled [95\% CI] & $p$, Qwen vs other & pooled [95\% CI] & $p$, Qwen vs other \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Pooled base $\rightarrow$ post-trained effects across pairs (DDXPlus, $T{=}0.7$): DerSimonian--Laird random-effects meta-analysis of the per-pair differences, with standard errors from their condition-clustered intervals. \emph{$p$, Qwen vs other}: test that the pooled effect differs between the Qwen pairs and the OLMo and Mistral pairs ($Q_{\text{between}}$, 1 df). With five or six pairs these tests have little power; they replace counts of significant pairs, not a full model of heterogeneity.}
\label{tab:meta}
\end{table}""")
