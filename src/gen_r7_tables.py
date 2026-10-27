import json
R='r7b/'
G=json.load(open(R+'greedy.json')); GM=G['models']; GP=G['pairs']
fam=[('Qwen3-8B','qwen3-8b-base','qwen3-8b'),('Qwen2.5-7B','qwen25-7b-base','qwen25-7b-it'),('Qwen2.5-32B','qwen25-32b-base','qwen25-32b-it'),
     ('OLMo~3 7B','olmo3-base','olmo3-instruct'),('OLMo~3.1 32B','olmo31-32b-base','olmo31-32b-instruct'),('Mistral-7B','mistral7b-base','mistral7b-it')]
def dc(x,sc=1,f='{:+.3f}'):
    sig=x['cluster_ci'][0]>0 or x['cluster_ci'][1]<0
    return f"${f.format(sc*x['diff'])}$"+('' if sig else '$^{\\dagger}$')
L=[]
for nm,b,p in fam:
    mb,mp=GM[f'greedy/{b}'],GM[f'greedy/{p}']; d=GP[f'greedy:{b}>{p}']
    L.append(f"{nm} & {mb['top1']:.3f} & {mp['top1']:.3f} & {dc(d['top1'],100,'{:+.1f}')} & {mb['rec5']:.3f} & {mp['rec5']:.3f} & {dc(d['rec5'],100,'{:+.1f}')} & {dc(d['tv_list'])} & {dc(d['sev_named'])} \\\\")
T1=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{3}{c}{Top-1 accuracy} & \multicolumn{3}{c}{Recall@5 of the list} & \multicolumn{2}{c}{Post $-$ base} \\
Pair & base & post & post $-$ base & base & post & post $-$ base & TV one list & Severe named \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Greedy single call (DDXPlus, 980 cases, temperature 0, one answer per case). Base models with the few-shot prompt of \S\ref{sec:fewshot}, post-trained models with the zero-shot prompt. Differences in points for accuracy and recall. $^{\dagger}$: the condition-clustered 95\% interval includes zero.}
\label{tab:greedy}
\end{table}"""
F=json.load(open(R+'frontier_review2.json')); FM={k.split('/runs/')[-1]:v for k,v in F['models'].items()}
rb=json.load(open(R+'robust.json'))
L=[]
names={'qwen3-8b-base':'Qwen3-8B-Base','qwen3-8b':'Qwen3-8B','qwen25-7b-base':'Qwen2.5-7B','qwen25-7b-it':'Qwen2.5-7B-Instruct','olmo3-base':'OLMo-3-1025-7B','olmo3-instruct':'OLMo-3-7B-Instruct'}
for i,(b,p) in enumerate([('qwen3-8b-base','qwen3-8b'),('qwen25-7b-base','qwen25-7b-it'),('olmo3-base','olmo3-instruct')]):
    if i: L.append('\\midrule')
    for t in (b,p):
        for j,T in enumerate(['04','07','10','13']):
            m=FM[f'frontier_t{T}/{t}']; c=json.load(open(R+f'frontier_cons_t{T}.json'))['models'][t]
            inv=rb.get(f'frontier_t{T}/{t}',{}).get('invalid_rate')
            invs=f"{100*inv:.1f}" if inv is not None else '---'
            L.append(f"{names[t] if j==0 else ''} & {int(T)/10:.1f} & {invs} & {m['pass1']:.3f} & {m['pass32']:.3f} & {m['distinct']:.2f} & {m['tv_list']:.3f} & {c['auroc_stated_p']:.3f} & {c['auroc_consist@32']:.3f} \\\\")
T2=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcrrrrrrr}
\toprule
 & & Invalid & & & & TV & \multicolumn{2}{c}{AUROC} \\
Model & $T$ & (\%) & pass@1 & pass@32 & Distinct & one list & stated & agree@32 \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Temperature frontier with the few-shot prompt (DDXPlus, 980 cases, 32 samples, top-$p$ 1 at every temperature). Invalid rates were computed for $T{=}0.4$ and 1.3 only.}
\label{tab:frontier}
\end{table}"""
M=json.load(open(R+'mnm.json')); MM=M['models']; MP=M['pairs']
mods=[('Qwen3-8B','mnm_t07/qwen3-8b','zs_ref/qwen3-8b','fewshot_t07/qwen3-8b-base'),('Qwen2.5-7B-Instruct','mnm_t07/qwen25-7b-it','zs_ref/qwen25-7b-it','fewshot_t07/qwen25-7b-base'),
      ('Qwen2.5-32B-Instruct','mnm_t07/qwen25-32b-it','big32_ddx/qwen25-32b-it','fewshot_t07/qwen25-32b-base'),('OLMo-3-7B-Instruct','mnm_t07/olmo3-instruct','meas_ddx/olmo3-instruct','fewshot_t07/olmo3-base'),
      ('OLMo-3.1-32B-Instruct','mnm_t07/olmo31-32b-instruct','big32_ddx/olmo31-32b-instruct','fewshot_t07/olmo31-32b-base'),('Mistral-7B-Instruct','mnm_t07/mistral7b-it','meas_ddx/mistral7b-it','fewshot_t07/mistral7b-base')]
L=[]
for nm,m_,d_,b_ in mods:
    m,d,b=MM[m_],MM[d_],MM[b_]
    ape=lambda x: x['severe_inclusion'].get('Acute pulmonary edema')
    L.append(f"{nm} & {100*m['valid_rate']:.0f} & {m['len']:.1f} + {m['len_full']-m['len']:.1f} & {d['rec5']:.3f} & {m['rec5']:.3f} & {b['sev_named']:.3f} & {d['sev_named']:.3f} & {m['sev_named_full']:.3f} & {d['unwarranted']:.2f} & {m['unwarranted_full']:.2f} & {ape(d):.2f} & {ape(m):.2f} \\\\")
T3=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccccccccc}
\toprule
 & Valid & Conditions & \multicolumn{2}{c}{Recall@5} & \multicolumn{3}{c}{Severe named (ref.\ $\geq 0.1$)} & \multicolumn{2}{c}{Unwarranted severe} & \multicolumn{2}{c}{Acute pulm.\ edema} \\
Model & (\%) & likely + mnm & default & likely part & base & default & mnm & default & mnm & default & mnm \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Must-not-miss prompt (DDXPlus, $T{=}0.7$, 980 cases, 32 samples): the five most likely conditions, then up to three dangerous conditions to rule out. \emph{Default}: the same model with the standard zero-shot prompt; \emph{base}: its base model with the few-shot prompt. Recall@5 of the most-likely part; severe conditions and unwarranted severe conditions (severe, reference $< 0.1$, per list) over the whole list; acute pulmonary edema: share of lists naming it where the reference gives it any weight.}
\label{tab:mnm}
\end{table}"""
open('r7_tables.tex','w').write(T1+'\n\n'+T2+'\n\n'+T3+'\n'); print(T1); print(T3)
