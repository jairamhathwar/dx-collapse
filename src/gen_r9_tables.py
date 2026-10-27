"""Greedy table with matched-prompt comparisons (tab:greedy). Reads r7b/greedy.json and r9/greedy_matched.json."""
import json
import sys
D = sys.argv[1] if len(sys.argv) > 1 else '.'
G = json.load(open(f'{D}/r7b/greedy.json')); M = json.load(open(f'{D}/r9/greedy_matched.json'))
GM = {**G['models'], **M['models']}
fam = [('Qwen3-8B', 'qwen3-8b-base', 'qwen3-8b'), ('Qwen2.5-7B', 'qwen25-7b-base', 'qwen25-7b-it'), ('Qwen2.5-32B', 'qwen25-32b-base', 'qwen25-32b-it'),
       ('OLMo~3 7B', 'olmo3-base', 'olmo3-instruct'), ('OLMo~3.1 32B', 'olmo31-32b-base', 'olmo31-32b-instruct'), ('Mistral-7B', 'mistral7b-base', 'mistral7b-it')]


def dc(x, sc=1, f='{:+.3f}'):
    sig = x['cluster_ci'][0] > 0 or x['cluster_ci'][1] < 0
    return f"${f.format(sc * x['diff'])}$" + ('' if sig else '$^{\\dagger}$')


def pairs(b, p):  # same examples, reference-free base, original
    return [M['pairs'][f'greedy/{b}>greedy_fs/{p}'], M['pairs'][f'greedy_rf/{b}>greedy/{p}'], G['pairs'][f'greedy:{b}>{p}']]


A, B = [], []
for nm, b, p in fam:
    lv = [GM[f'greedy/{b}'], GM[f'greedy_rf/{b}'], GM[f'greedy/{p}'], GM[f'greedy_fs/{p}']]
    P = pairs(b, p)
    A.append(f"{nm} & " + ' & '.join(f"{m['top1']:.3f}" for m in lv) + ' & ' + ' & '.join(dc(d['top1'], 100, '{:+.1f}') for d in P) + r' \\')
    B.append(f"{nm} & " + ' & '.join(dc(d['rec5'], 100, '{:+.1f}') for d in P) + ' & ' + ' & '.join(dc(d['tv_list']) for d in P)
             + ' & ' + ' & '.join(dc(d['sev_named']) for d in P) + r' \\')
print(r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{4}{c}{Top-1 accuracy} & \multicolumn{3}{c}{Top-1, post $-$ base (points)} \\
 & \multicolumn{2}{c}{base, examples} & \multicolumn{2}{c}{post-trained} & same & ref.-free & \\
Pair & reference & ref.-free & zero-shot & examples & examples & base & original \\
\midrule
""" + '\n'.join(A) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\medskip
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{3}{c}{Recall@5, post $-$ base (points)} & \multicolumn{3}{c}{TV one list, post $-$ base} & \multicolumn{3}{c}{Severe named, post $-$ base} \\
Pair & same & ref.-free & original & same & ref.-free & original & same & ref.-free & original \\
\midrule
""" + '\n'.join(B) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Greedy single call (DDXPlus, 980 cases, temperature 0, one answer per case). Base models are prompted with the worked examples of \S\ref{sec:fewshot}, whose lists are the reference top five (``reference'') or Qwen2.5-32B-Instruct's answers (``ref.-free''); post-trained models are prompted zero-shot or with the reference examples. Three comparisons: \emph{same examples} (both arms see the reference examples), \emph{ref.-free base} (reference-free base against zero-shot post-trained, so neither arm sees a reference list), and \emph{original} (reference-example base against zero-shot post-trained). Differences are over cases where both models gave a valid answer, so they can differ slightly from the difference of the level columns. $^{\dagger}$: the condition-clustered 95\% interval includes zero.}
\label{tab:greedy}
\end{table}""")
