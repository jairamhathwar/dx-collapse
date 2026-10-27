"""Table for published case reports (tab:medcase). Reads medcase/medcase.json and medcase/medcase_strict.json
(medcase_analysis.py)."""
import json
import sys

D = sys.argv[1] if len(sys.argv) > 1 else '.'
M = json.load(open(f'{D}/medcase/medcase.json'))
S = json.load(open(f'{D}/medcase/medcase_strict.json'))
PAIRS = [('Qwen3-8B', 'qwen3-8b-base', 'qwen3-8b'), ('Qwen2.5-7B', 'qwen25-7b-base', 'qwen25-7b-it'),
         ('Qwen2.5-32B', 'qwen25-32b-base', 'qwen25-32b-it'), ('OLMo~3 7B', 'olmo3-base', 'olmo3-instruct')]


def dc(x, sc=1, f='{:+.3f}'):
    return f"${f.format(sc * x['diff'])}$" + ('' if x['p_holm'] < 0.05 else '$^{\\dagger}$')


def auc_d(m):
    d, lo, hi = m['d_auroc_stated-agree32']
    return f"${d:+.3f}$" + ('' if lo > 0 or hi < 0 else '$^{\\dagger}$')


A, B = [], []
for nm, b, p in PAIRS:
    for lab, t in [('base', b), ('post-trained', p)]:
        m = M['models'][t]
        A.append(f"{nm if lab == 'base' else ''} & {lab} & {m['valid']:.3f} & {m['pass1']:.3f} & {m['pass32']:.3f} & {m['distinct']:.2f} & "
                 f"{m['rec1']:.3f} & {m['rec5']:.3f} & {100 * (m['vote32'] - m['single']):+.1f} & {m['auroc_stated_p']:.3f} & {m['auroc_agree32']:.3f} & {auc_d(m)} \\\\")
    q, r = M['pairs'][f'{b}>{p}'], S['pairs'][f'{b}>{p}']
    B.append(f"{nm} & {dc(q['pass1'], 100, '{:+.1f}')} & {dc(q['pass32'], 100, '{:+.1f}')} & {dc(q['distinct'], 1, '{:+.2f}')} & "
             f"{dc(q['rec1'], 100, '{:+.1f}')} & {dc(q['rec5'], 100, '{:+.1f}')} & {dc(q['vote32-single'], 100, '{:+.1f}')} & "
             f"{dc(r['pass32'], 100, '{:+.1f}')} & {dc(r['rec5'], 100, '{:+.1f}')} \\\\")
ag = M.get('judge_agreement', {})
print(r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccccccccc}
\toprule
 & & & & & & \multicolumn{2}{c}{One stated list} & Vote@32 & \multicolumn{3}{c}{AUROC, single-call answer correct} \\
Pair & Model & Valid & pass@1 & pass@32 & Distinct & recall@1 & recall@5 & $-$ single & stated $p$ & agree@32 & stated $-$ agree \\
\midrule
""" + '\n'.join(A) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\medskip
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{6}{c}{Post-trained $-$ base (judge 1)} & \multicolumn{2}{c}{Strict (both judges)} \\
Pair & pass@1 & pass@32 & Distinct & recall@1 & recall@5 & Vote@32 $-$ single (post) & pass@32 & recall@5 \\
\midrule
""" + '\n'.join(B) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Published case reports (MedCaseReasoning test split, 897 PMC case reports, zero-shot, 32 samples, $T{=}0.7$). A name is correct if the judge (Mistral-Small-24B-Instruct) says it names the report's final diagnosis; \emph{strict}: if Qwen2.5-32B-Instruct agrees""" +
      (f" (the judges agree on {100 * ag['agree']:.1f}\\% of {ag['n']:,} judged names, Cohen's $\\kappa$ {ag['kappa']:.2f})" if ag else "") +
      r""". Invalid samples count as wrong for pass@$k$; the list and agreement columns use valid samples. Distinct: distinct top answers among valid samples, with all correct answers merged and others compared as normalized text. Accuracy and recall differences in points. $^{\dagger}$: not significant (paired case bootstrap, 10,000 resamples, Holm over the 28 pair tests; for the AUROC difference, the 95\% interval includes zero).}
\label{tab:medcase}
\end{table}""")
