"""Table for the readout reversal and silent errors (tab:reversal). Reads r25.json (r25_reversal.py).

    python gen_r25_tables.py r25.json > tab_reversal.tex
"""
import json
import sys

R = json.load(open(sys.argv[1]))
PAIRS = [("Qwen3-8B", "qwen3-8b-base", "qwen3-8b"), ("Qwen2.5-7B", "qwen25-7b-base", "qwen25-7b-it"),
         ("Qwen2.5-32B", "qwen25-32b-base", "qwen25-32b-it"), ("Qwen2.5-72B", "qwen25-72b-base", "qwen25-72b-it"),
         ("OLMo~3 7B", "olmo3-base", "olmo3-instruct"), ("OLMo~3.1 32B", "olmo31-32b-base", "olmo31-32b-instruct"),
         ("Mistral-7B$^{*}$", "mistral7b-base", "mistral7b-it"), ("MedGemma-4B$^{*}$", "medgemma-4b-pt", "medgemma-4b-it")]
D = r"$^{\dagger}$"


def key(t):
    return next(k for k in R["models"] if k.endswith("/" + t))


def ns(x):
    lo, hi = x["cluster_ci"]
    return "" if (lo > 0 or hi < 0) else D


def pt(x):
    return f"${100 * x['diff']:+.1f}${ns(x)}"


rows = []
for nm, b, p in PAIRS:
    for lab, t in [("base", b), ("post-trained", p)]:
        m, w = R["models"][key(t)], R["within"][key(t)]
        sil = m["caught20_stated_among_silent"]
        rows.append(f"{nm if lab == 'base' else ''} & {lab} & {m['list_1_first']:.3f} & {m['pass_32']:.3f} & {pt(w['reversal'])} & {pt(w['reversal_corrected'])} & "
                    f"{100 * m['silent_share_of_errors']['diff']:.0f}\\% & {100 * sil['diff']:.0f}\\% \\\\")
print(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccccc}
\toprule
 & & First list & 32 samples & \multicolumn{2}{c}{List $-$ samples} & Silent & Silent errors \\
Pair & Model & recall@5 & pass@32 & raw & chance-corrected & errors & flagged by stated $p$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Readout reversal and silent errors (DDXPlus, $T{=}0.7$, 980 cases; zero-shot except $^{*}$, worked examples). \emph{First list}: the first valid sample's stated list contains the true pathology. \emph{List $-$ samples}: first list minus pass@32 of the same model, in points; \emph{chance-corrected}: each term minus what it reaches against the true pathology of another case with a different pathology. \emph{Silent errors}: share of wrong first answers with which every other valid sample agrees, so agreement cannot flag them. \emph{Flagged by stated $p$}: share of those referred when the 20\% of cases with the lowest stated probability are referred. $^{\dagger}$: condition-clustered 95\% interval includes zero. Post-trained minus base, the list $-$ samples difference rises by 17--41 points in every pair (all intervals exclude zero).}
\label{tab:reversal}
\end{table}""")
