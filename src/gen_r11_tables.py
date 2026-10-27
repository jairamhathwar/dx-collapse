"""Table for published case reports (tab:medcase). Reads medcase/medcase.json and medcase/medcase_strict.json
(medcase_analysis.py)."""
import json
import sys

D = sys.argv[1] if len(sys.argv) > 1 else '.'
M = json.load(open(f'{D}/medcase/medcase.json'))
S = json.load(open(f'{D}/medcase/medcase_strict.json'))
PAIRS = [('Qwen3-8B', 'qwen3-8b-base', 'qwen3-8b'), ('Qwen2.5-7B', 'qwen25-7b-base', 'qwen25-7b-it'),
         ('Qwen2.5-32B', 'qwen25-32b-base', 'qwen25-32b-it'), ('OLMo~3 7B', 'olmo3-base', 'olmo3-instruct')]
DAG = r'$^{\dagger}$'


def dc(x, sc=1, f='{:+.3f}'):
    return f"${f.format(sc * x['diff'])}$" + ('' if x['p_holm'] < 0.05 else DAG)


def ci(d, sc=1, f='{:+.3f}'):
    return f"${f.format(sc * d[0])}$" + ('' if d[1] > 0 or d[2] < 0 else DAG)


A, B = [], []
for nm, b, p in PAIRS:
    for lab, t in [('base', b), ('post-trained', p)]:
        m = M['models'][t]
        A.append(f"{nm if lab == 'base' else ''} & {lab} & {m['valid']:.3f} & {m['pass1']:.3f} & {m['pass32']:.3f} & {m['distinct']:.2f} & "
                 f"{m['rec5']:.3f} & {m['alt_list']:.3f} & {m['alt_samples']:.3f} & {100 * (m['vote32'] - m['single']):+.1f} & "
                 f"{m['auroc_stated_p']:.3f} & {m['auroc_agree32']:.3f} & {ci(m['d_auroc_stated-agree32'])} \\\\")
    q, r = M['pairs'][f'{b}>{p}'], S['pairs'][f'{b}>{p}']
    B.append(f"{nm} & {dc(q['pass1'], 100, '{:+.1f}')} & {dc(q['pass32'], 100, '{:+.1f}')} & {dc(q['distinct'], 1, '{:+.2f}')} & "
             f"{dc(q['rec5'], 100, '{:+.1f}')} & {dc(q['alt_list'], 100, '{:+.1f}')} & {dc(q['alt_samples'], 100, '{:+.1f}')} & "
             f"{ci(M['models'][p]['d_alt_list-samples'], 100, '{:+.1f}')} & {dc(q['vote32-single'], 100, '{:+.1f}')} & "
             f"{dc(r['pass32'], 100, '{:+.1f}')} & {dc(r['rec5'], 100, '{:+.1f}')} \\\\")
ag = M['judge_agreement']
print(r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llccccccccccc}
\toprule
 & & & & & & Recall@5 & \multicolumn{2}{c}{Authors' alternatives} & Vote@32 & \multicolumn{3}{c}{AUROC, single-call answer correct} \\
Pair & Model & Valid & pass@1 & pass@32 & Distinct & one list & one list & 32 samples & $-$ single & stated $p$ & agree@32 & stated $-$ agree \\
\midrule
""" + '\n'.join(A) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\medskip
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccccccc}
\toprule
 & \multicolumn{6}{c}{Post-trained $-$ base (primary grading)} & List $-$ samples & Vote $-$ single & \multicolumn{2}{c}{Strict grading} \\
Pair & pass@1 & pass@32 & Distinct & recall@5 & alt., list & alt., samples & (post, alt.) & (post) & pass@32 & recall@5 \\
\midrule
""" + '\n'.join(B) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Published case reports (MedCaseReasoning test split, 897 PMC case reports, zero-shot, 32 samples, $T{=}0.7$). A name is correct if the primary judge (Mistral-Small-24B-Instruct) says it names the report's final diagnosis; \emph{strict}: only if Phi-4 agrees as well """ +
      f"(the judges agree on {100 * ag['agree']:.1f}\\% of {ag['n']:,} judged names, Cohen's $\\kappa$ {ag['kappa']:.2f})" +
      r""". \emph{Authors' alternatives}: share of the diagnoses other than the final one that the report's authors considered (extracted from its reasoning notes; 893 cases with at least one, 3.5 on average) that one list names, or that any of the 32 sampled top answers names; \emph{list $-$ samples}: the difference within the post-trained model. Invalid samples count as wrong for pass@$k$; the list and agreement columns use valid samples. Distinct: distinct top answers among valid samples, grouped by normalized text whether correct or not (as for voting and agreement). Differences in points. $^{\dagger}$: not significant (paired case bootstrap, 10,000 resamples, Holm over the 36 pair tests; for the AUROC and list $-$ samples differences, the 95\% interval includes zero).}
\label{tab:medcase}
\end{table}""")
