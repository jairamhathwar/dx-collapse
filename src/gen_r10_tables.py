"""Tables for must-not-miss conditions defined outside DDXPlus (tab:extmnm). Reads r10/*.json (slurm/r10_an.sh)."""
import json
import sys
D = sys.argv[1] if len(sys.argv) > 1 else '.'
L = lambda f: json.load(open(f'{D}/r10/{f}'))
ZS = [('Qwen3-8B, $T{=}0.7$', 'pair_q3:qwen3-8b-base>qwen3-8b'), ('Qwen3-8B, $T{=}1$', 'ddx_qwen3_t1:qwen3-8b-base>qwen3-8b'),
      ('Qwen2.5-7B, $T{=}0.7$', 'pair25_t07:qwen25-7b-base>qwen25-7b-it'), ('Qwen2.5-7B, $T{=}1$', 'pair25_t1:qwen25-7b-base>qwen25-7b-it'),
      ('Qwen2.5-32B, $T{=}0.7$', 'big32_ddx:qwen25-32b-base>qwen25-32b-it'), ('OLMo~3 7B, $T{=}0.7$', 'ddx_olmo:olmo3-base>olmo3-instruct'),
      ('OLMo~3 7B, $T{=}1$', 'ddx_olmo_t1:olmo3-base>olmo3-instruct'), ('OLMo~3.1 32B, $T{=}0.7$', 'big32_ddx:olmo31-32b-base>olmo31-32b-instruct')]
F = '/mmfs1/home/jairamh/thesis_overflow/runs/fewshot_t07:'
FS = [('Qwen3-8B, few-shot', F + 'qwen3-8b-base>qwen3-8b'), ('Qwen2.5-7B, few-shot', F + 'qwen25-7b-base>qwen25-7b-it'),
      ('Qwen2.5-32B, few-shot', F + 'qwen25-32b-base>qwen25-32b-it'), ('OLMo~3 7B, few-shot', F + 'olmo3-base>olmo3-instruct'),
      ('OLMo~3.1 32B, few-shot', F + 'olmo31-32b-base>olmo31-32b-instruct'), ('Mistral-7B, few-shot', F + 'mistral7b-base>mistral7b-it')]


def dc(x, f='{:+.3f}'):
    ci = x.get('cluster_ci') or x.get('ci')
    return f"${f.format(x['diff'])}$" + ('' if ci[0] > 0 or ci[1] < 0 else '$^{\\dagger}$')


def level(d, key, side):
    run, pr = key.rsplit(':', 1)
    tag = pr.split('>')[0 if side == 'b' else 1]
    m = [k for k in d['models'] if k.endswith(f"{run.split('/')[-1]}/{tag}")]
    return d['models'][m[0]]['sev_named']


rows = []
for grp, items in [('zs', ZS), ('fs', FS)]:
    U, H, C = L(f'review2_{grp}_union.json'), L(f'review2_{grp}_harm.json'), L(f'review2_{grp}_chestpain.json')
    for nm, k in items:
        st = U['severe_test'][k]
        rows.append(f"{nm} & {level(U, k, 'b'):.3f} & {level(U, k, 'p'):.3f} & {dc(U['pairs'][k]['sev_named'])} & "
                    f"{st['sig_lower_q05']} / {st['sig_higher_q05']} & {dc(H['pairs'][k]['sev_named'])} & {dc(C['pairs'][k]['sev_named'])} \\\\")
    if grp == 'zs':
        rows.append(r'\midrule')
M, Cn = L('mnm_union.json'), L('controls_union.json')['length_matched_union2']
MR = []
for nm, tag, std in [('Qwen3-8B', 'qwen3-8b', 'pair_q3/qwen3-8b'), ('Qwen2.5-7B-Instruct', 'qwen25-7b-it', 'pair25_t07/qwen25-7b-it'),
                     ('Qwen2.5-32B-Instruct', 'qwen25-32b-it', 'big32_ddx/qwen25-32b-it'), ('OLMo-3-7B-Instruct', 'olmo3-instruct', 'ddx_olmo/olmo3-instruct'),
                     ('OLMo-3.1-32B-Instruct', 'olmo31-32b-instruct', 'big32_ddx/olmo31-32b-instruct'), ('Mistral-7B-Instruct', 'mistral7b-it', 'mistral_t07/mistral7b-it')]:
    pk = [p for p in M['pairs'] if p.split('>')[0].endswith(f'mnm_t07/{tag}') and p.split('>')[1].endswith(std)][0]
    a, b = pk.split('>')
    MR.append(f"{nm} & {M['models'][b]['sev_named']:.3f} & {Cn[tag]['sev_named']:.3f} & {M['models'][a]['sev_named_full']:.3f} & "
              f"{dc(M['pairs'][pk]['sev_named_full'])} & {M['models'][b]['unwarranted']:.2f} & {Cn[tag]['unwarranted']:.2f} & {M['models'][a]['unwarranted_full']:.2f} \\\\")
print(r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{4}{c}{Union set (7 conditions)} & Harm-based & Chest pain \\
Pair & base & post & post $-$ base & sig.\ lost / gained & post $-$ base & post $-$ base \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\medskip
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{4}{c}{Warranted must-not-miss conditions named (union set)} & \multicolumn{3}{c}{Unwarranted per list} \\
Post-trained model & standard & two lists & must-not-miss & $-$ standard & standard & two lists & must-not-miss \\
\midrule
""" + '\n'.join(MR) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Must-not-miss conditions defined outside DDXPlus (DDXPlus, 980 cases). \emph{Harm-based}: the DDXPlus conditions among the 15 diseases behind about half of serious misdiagnosis-related harms in the USA~\citep{newmantoker2024burden} (NSTEMI/STEMI, pulmonary embolism, pneumonia, pulmonary neoplasm). \emph{Chest pain}: the life-threatening causes of acute chest pain in the 2021 AHA/ACC guideline~\citep{gulati2021chestpain} (NSTEMI/STEMI, unstable angina, pulmonary embolism, Boerhaave, spontaneous pneumothorax). \emph{Union}: either set. Top: share of the set's conditions with reference probability $\geq 0.1$ that one list names, and the number of the union's conditions named significantly less / more often (Benjamini--Hochberg over the 7 conditions). Bottom: the must-not-miss prompt (\S\ref{sec:decoding}) scored against the union set; \emph{two lists}: the union of two standard lists, a length-matched comparison; unwarranted: the set's conditions named that the reference weights below 0.1. $^{\dagger}$: the condition-clustered 95\% interval includes zero.}
\label{tab:extmnm}
\end{table}""")
