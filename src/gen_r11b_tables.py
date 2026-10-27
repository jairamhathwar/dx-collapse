"""Tables for chance-corrected pass@32 (tab:chance) and the stricter packing test (tab:packstrict).
Reads r11/r11.json and r11/shuffle_fs.json (r11_an.py)."""
import json
import sys

D = sys.argv[1] if len(sys.argv) > 1 else '.'
Z = json.load(open(f'{D}/r11/r11.json'))
S = json.load(open(f'{D}/r11/shuffle_fs.json'))['shuffled_pass32']
ZS = Z['shuffled_pass32']
PAIRS = [('Qwen3-8B', 'qwen3-8b-base', 'qwen3-8b', 'pair_q3'), ('Qwen2.5-7B', 'qwen25-7b-base', 'qwen25-7b-it', 'pair25_t07'),
         ('Qwen2.5-32B', 'qwen25-32b-base', 'qwen25-32b-it', 'big32_ddx'), ('OLMo~3 7B', 'olmo3-base', 'olmo3-instruct', 'ddx_olmo'),
         ('OLMo~3.1 32B', 'olmo31-32b-base', 'olmo31-32b-instruct', 'big32_ddx'), ('Mistral-7B', 'mistral7b-base', 'mistral7b-it', None)]
NAMES = {'pair_q3/qwen3-8b-base': 'Qwen3-8B-Base', 'pair25_t07/qwen25-7b-base': 'Qwen2.5-7B', 'big32_ddx/qwen25-32b-base': 'Qwen2.5-32B',
         'ddx_olmo/olmo3-base': 'OLMo-3-1025-7B', 'big32_ddx/olmo31-32b-base': 'OLMo-3-1125-32B', 'pair_q3/qwen3-8b': 'Qwen3-8B',
         'pair25_t07/qwen25-7b-it': 'Qwen2.5-7B-Instruct', 'big32_ddx/qwen25-32b-it': 'Qwen2.5-32B-Instruct',
         'ddx_olmo/olmo3-instruct': 'OLMo-3-7B-Instruct', 'big32_ddx/olmo31-32b-instruct': 'OLMo-3.1-32B-Instruct'}


def get(d, suffix):
    v = d[[k for k in d if k.endswith(suffix)][0]]
    return v['real'], v['shuffled']


rows = []
for nm, b, p, zd in PAIRS:
    cells = []
    for src, sfx in [(ZS, zd), (S, 'fewshot_t07'), (S, 'fewshot_rf_t07')]:
        if sfx is None:
            cells += ['---'] * 3
            continue
        rb, sb = get(src, f'{sfx}/{b}')
        rp, sp = get(src, f'{sfx}/{p}')
        cells += [f'{sb:.3f} / {sp:.3f}', f'${100 * (rp - rb):+.1f}$', f'${100 * ((rp - sp) - (rb - sb)):+.1f}$']
    rows.append(f"{nm} & " + ' & '.join(cells) + r' \\')
T1 = r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{3}{c}{Zero-shot} & \multicolumn{3}{c}{Worked examples (reference lists)} & \multicolumn{3}{c}{Worked examples (another model's lists)} \\
Pair & shuffled (base / post) & raw & corrected & shuffled (base / post) & raw & corrected & shuffled (base / post) & raw & corrected \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Chance-corrected pass@32 (DDXPlus, $T{=}0.7$, 980 cases). \emph{Shuffled}: pass@32 when each case's samples are scored against the true pathology of a random case with a different pathology (mean of 20 permutations), the part of pass@32 that does not depend on the patient. \emph{Raw}: post-trained minus base pass@32, in points; \emph{corrected}: the same after subtracting each model's shuffled value. Mistral's base model does not answer the zero-shot prompt (\S\ref{sec:mistral}).}
\label{tab:chance}
\end{table}"""

P = Z['packing_strict']
rows = []
for nm, b, p, zd in PAIRS[:5]:
    v = P[f'{zd}:{b}>{p}']
    own = f'{zd}/{b}'
    dc = lambda x: f"${x['diff']:+.3f}$" + ('' if x['cluster_ci'][0] > 0 or x['cluster_ci'][1] < 0 else r'$^{\dagger}$')
    rows.append(f"{nm} & {v['mean_tv_to'][own]:.3f} & {NAMES[v['closest_other_base']]} & {dc(v['own-closest_other_base'])} & "
                f"{NAMES[v['closest_other_post']]} & {dc(v['own-closest_other_post'])} & {dc(v['own-mean_other_post'])} \\\\")
T2 = r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcllclc}
\toprule
 & TV to own & \multicolumn{2}{c}{Closest other base model} & \multicolumn{2}{c}{Closest other post-trained model} & Own $-$ mean of other \\
Post-trained model of & base mean & model & own $-$ it & model & own $-$ it & post-trained models \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{A stricter packing test (DDXPlus, zero-shot, $T{=}0.7$, 980 cases). For each post-trained model, the TV between its lists and the mean list of each other final model (five base, five post-trained) for the same patient. Negative differences: the post-trained list is closer to its own base model's view than to the comparison. $^{\dagger}$: the condition-clustered 95\% interval includes zero.}
\label{tab:packstrict}
\end{table}"""
print(T1 + "\n\n" + T2)
