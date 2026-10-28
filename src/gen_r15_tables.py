"""Tables for the seeded pick (tab:seeded), the free-length list (tab:freelen) and the closed-set case reports
(tab:closedcase). Reads seeded.json (seeded_an.py), r14_freelen.json (r14_an.py --groups freelen) and closed.json
(medcase_closed_an.py) from DIR; writes one .tex file per table into OUT.

    python gen_r15_tables.py DIR OUT
"""
import json
import sys
from pathlib import Path

D, OUT = sys.argv[1], Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
DAG = r"$^{\dagger}$"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def bold(s, on):
    return f"\\textbf{{{s}}}" if on else s


# ---------------- tab:seeded
S = json.load(open(f"{D}/seeded.json"))
names = [("qwen3-8b", "Qwen3-8B"), ("qwen25-7b-it", "Qwen2.5-7B-Instruct"), ("qwen25-32b-it", "Qwen2.5-32B-Instruct"),
         ("olmo3-instruct", "OLMo-3-7B-Instruct"), ("olmo31-32b-instruct", "OLMo-3.1-32B-Instruct")]


def d(t, k, sc=100, f="{:+.1f}"):
    v = S["diffs"][t][k]
    return f"${f.format(sc * v['diff'])}$" + ("" if v["p_holm"] < 0.05 else DAG)


rows = []
for t, nm in names:
    m = S["models"][t]
    rows.append(f"{nm} & {100 * m['follows']:.0f}\\% & {m['pick_distinct']:.2f} & {m['default_distinct']:.2f} & {m['base_distinct']:.2f} & "
                f"{m['pick_pass32']:.3f} & {d(t, 'seeded-default_pass32')} & {d(t, 'seeded-base_pass32')} & {d(t, 'seeded-default_pass1')} & "
                f"{m['rule_pass32']:.3f} & {m['pick_auroc_agree']:.3f} & {m['default_auroc_agree']:.3f} \\\\")
(OUT / "tab_seeded.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccccc}
\toprule
 & Follows & \multicolumn{3}{c}{Distinct top answers} & \multicolumn{3}{c}{pass@32} & pass@1 & pass@32 if & \multicolumn{2}{c}{Agreement AUROC} \\
Model & the rule & seeded & ordinary & base & seeded & $-$ ordinary & $-$ base & $-$ ordinary & rule followed & seeded & ordinary \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Seeded pick (DDXPlus, zero-shot, $T{=}0.7$, 980 cases). Each of 32 calls receives its own random number $u$ (stratified over $[0,1]$) and is asked to state its list and then pick the condition at which the running total of its rescaled top-five probabilities first reaches $u$. \emph{Follows the rule}: share of calls whose pick is that condition. \emph{Ordinary}: the same model's zero-shot samples; \emph{base}: its base model's. \emph{pass@32 if rule followed}: the rule applied by us to each call's own list. Differences in points; $^{\dagger}$: not significant (condition-clustered bootstrap, Holm across the 40 tests of this analysis). Agreement for the seeded calls is the share of the other 31 picks equal to the first list's top answer.}
\label{tab:seeded}
\end{table}
""")

# ---------------- tab:freelen
F = json.load(open(f"{D}/r14_freelen.json"))
pairs = [("qwen3-8b-base", "qwen3-8b", "Qwen3-8B"), ("qwen25-7b-base", "qwen25-7b-it", "Qwen2.5-7B"), ("qwen25-32b-base", "qwen25-32b-it", "Qwen2.5-32B")]
cols = [("list_len", "{:.1f}", None), ("rec_all", "{:.3f}", "hi"), ("rec5", "{:.3f}", "hi"), ("tv_list", "{:.3f}", "lo"), ("mass_all", "{:.3f}", "hi"),
        ("cond_tv", "{:.3f}", "lo"), ("sev_named", "{:.3f}", None), ("distinct", "{:.2f}", "hi"), ("pass32", "{:.3f}", "hi")]
keys = [f"@freelen_t07:{b}>{p}" for b, p, _ in pairs]
tests = [(k, c) for k in keys for c, _, _ in cols]
adj = dict(zip(tests, holm([F["pairs"][k][c]["p"] for k, c in tests])))
rows = []
for (b, p, nm), k in zip(pairs, keys):
    for who, t in [("base", b), ("post", p)]:
        m = F["models"][f"@freelen_t07/{t}"]
        cells = []
        for c, f, better in cols:
            dd = F["pairs"][k][c]["diff"]
            win = None
            if better and adj[(k, c)] < 0.05:
                win = "post" if (dd > 0) == (better == "hi") else "base"
            cells.append(bold(f.format(m[c]), win == who))
        rows.append((nm if who == "base" else "") + f" & {'base' if who == 'base' else 'post-trained'} & " + " & ".join(cells) + " \\\\")
    rows.append("\\midrule")
(OUT / "tab_freelen.tex").write_text(r"""\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llccccccccc}
\toprule
 & & \multicolumn{7}{c}{One stated list} & \multicolumn{2}{c}{Sampled top answers} \\
Pair & Model & Length & Contains true & Recall@5 & TV$\downarrow$ & Mass & Cond.\ TV$\downarrow$ & Severe & Distinct & pass@32 \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Free-length lists (DDXPlus, zero-shot, $T{=}0.7$, 980 cases, 32 samples): the prompt asks for every condition the model considers plausible, as many or as few as it judges appropriate. \emph{Length}: conditions per valid list; \emph{contains true}: the list names the true pathology; \emph{mass}: reference probability of the named conditions; \emph{severe}: share of warranted severe conditions named. Bold: significantly better model of the pair (condition-clustered bootstrap, Holm across the table's 21 tests); length and severe named are not ranked.}
\label{tab:freelen}
\end{table}
""")

# ---------------- tab:closedcase
C = json.load(open(f"{D}/closed.json"))
cp = [("qwen3-8b-base", "qwen3-8b", "Qwen3-8B"), ("qwen25-7b-base", "qwen25-7b-it", "Qwen2.5-7B"),
      ("qwen25-32b-base", "qwen25-32b-it", "Qwen2.5-32B"), ("olmo3-base", "olmo3-instruct", "OLMo~3 7B")]
A, B = [], []
for b, p, nm in cp:
    for lab, t in [("base", b), ("post-trained", p)]:
        m, w = C["models"][t], C["within"][t]
        A.append(f"{nm if lab == 'base' else ''} & {lab} & {m['valid']:.3f} & {m['pass1']:.3f} & {m['pass32']:.3f} & {m['distinct']:.2f} & {m['rec5']:.3f} & "
                 f"{m['alt_list']:.3f} & {m['alt_samples']:.3f} & {m['auroc_stated']:.3f} & {m['auroc_agree32']:.3f} \\\\")
    q = C["pairs"][f"{b}>{p}"]
    f = lambda k, sc=100, fm="{:+.1f}": f"${fm.format(sc * q[k]['diff'])}$" + ("" if q[k]["p_holm"] < 0.05 else DAG)
    wl = C["within"][p]["alt_list-samples"]
    B.append(f"{nm} & {f('pass1')} & {f('pass32')} & {f('distinct', 1, '{:+.2f}')} & {f('rec5')} & {f('alt_list')} & {f('alt_samples')} & "
             f"${100 * wl['diff']:+.1f}$ & ${100 * C['within'][b]['alt_list-samples']['diff']:+.1f}$ & {f('vote_gain')} \\\\")
(OUT / "tab_closedcase.tex").write_text(r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llccccccccc}
\toprule
 & & & & & & Recall@5 & \multicolumn{2}{c}{Authors' alternatives} & \multicolumn{2}{c}{AUROC, single-call answer} \\
Pair & Model & Valid & pass@1 & pass@32 & Distinct & one list & one list & 32 samples & stated $p$ & agree@32 \\
\midrule
""" + "\n".join(A) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\medskip
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{6}{c}{Post-trained $-$ base} & \multicolumn{2}{c}{Alternatives: list $-$ samples} & Vote@32 \\
Pair & pass@1 & pass@32 & Distinct & recall@5 & alt., list & alt., samples & post & base & gain \\
\midrule
""" + "\n".join(B) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Closed-set case reports (MedCaseReasoning, 897 cases, zero-shot, 32 samples, $T{=}0.7$). Each prompt lists ten candidates: the final diagnosis, the authors' alternatives (category-only entries removed) and, to fill the list, other cases' final diagnoses that share no content word with the candidates, in random order. Answers are matched to the candidates by name, with no judge; a sample is valid if its top answer is a candidate. The OLMo base model was capped at 1,024 tokens, and text after an invented chat turn is ignored. Differences in points; $^{\dagger}$: not significant (paired case bootstrap, 10,000 resamples, Holm across the 36 pair tests). Every list-minus-samples interval excludes zero.}
\label{tab:closedcase}
\end{table}
""")
print("ok")
