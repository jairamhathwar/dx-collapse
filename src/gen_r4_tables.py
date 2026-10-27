import json
R='r4/'
rf=json.load(open(R+'review2_rf.json')); mt=json.load(open(R+'review2_mt.json'))
rb=json.load(open(R+'robust.json')); cm=json.load(open(R+'cons_mt.json'))['models']
P={k.split('/runs/')[-1]:v for k,v in rf['pairs'].items()}
pairs=[('Qwen3-8B','qwen3-8b-base','qwen3-8b'),('Qwen2.5-7B','qwen25-7b-base','qwen25-7b-it'),('Qwen2.5-32B','qwen25-32b-base','qwen25-32b-it'),
       ('OLMo~3 7B','olmo3-base','olmo3-instruct'),('OLMo~3.1 32B','olmo31-32b-base','olmo31-32b-instruct'),('Mistral-7B','mistral7b-base','mistral7b-it')]
cols=[('pass32',100,'{:+.1f}'),('tv_samples',1,'{:+.3f}'),('tv_list',1,'{:+.3f}'),('tv_list_nogold',1,'{:+.3f}'),('ddf1',1,'{:+.3f}'),('within_list',1,'{:+.3f}')]
L=[]
for name,b,p in pairs:
    v=P[f'fewshot_rf_t07:{b}>{p}']; cells=[]
    for k,sc,f in cols:
        x=v[k]
        if k=='within_list':
            sig = x['cluster_ci'][0]>0 or x['cluster_ci'][1]<0
        else:
            sig = x['p_holm']<0.05
        cells.append(f"${f.format(sc*x['diff'])}$"+('' if sig else '$^{\\dagger}$'))
    L.append(name+' & '+' & '.join(cells)+' \\\\')
T1=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccc}
\toprule
 & pass@32 & TV & \multicolumn{2}{c}{TV, one list} & & Misallo- \\
Pair (base $\rightarrow$ post) & (points) & samples & all & w/o true & DDF1 & cated \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Few-shot control with reference-free examples (DDXPlus, $T{=}0.7$, 980 cases): the same three training patients as in \S\ref{sec:fewshot}, but each example list is Qwen2.5-32B-Instruct's own greedy answer rather than the reference's top five. Post-trained minus base. $^{\dagger}$: not significant (condition-clustered bootstrap, Holm over 84 tests; for the misallocation, the clustered interval includes zero).}
\label{tab:refree}
\end{table}"""
M={k.split('/runs/')[-1]:v for k,v in mt['models'].items()}
rows=[('Qwen3-8B-Base','0.7','qwen3-8b-base-t07'),('','1','qwen3-8b-base-t1'),('Qwen3-8B','0.7','qwen3-8b-t07'),('','1','qwen3-8b-t1'),('','1.3','qwen3-8b-t13'),('','1.6','qwen3-8b-t16'),
      None,('Qwen2.5-7B','0.7','qwen25-7b-base-t07'),('','1','qwen25-7b-base-t1'),('Qwen2.5-7B-Instruct','0.7','qwen25-7b-it-t07'),('','1','qwen25-7b-it-t1'),('','1.3','qwen25-7b-it-t13'),('','1.6','qwen25-7b-it-t16')]
import re
ZSM={k.split('/runs/')[-1]:v for k,v in json.load(open('r3/review2_zs.json'))['models'].items()}
FALL={'qwen3-8b-t07':'pair_q3/qwen3-8b','qwen3-8b-t1':'ddx_qwen3_t1/qwen3-8b','qwen25-7b-it-t07':'pair25_t07/qwen25-7b-it','qwen25-7b-it-t1':'pair25_t1/qwen25-7b-it'}
L=[]
for r in rows:
    if r is None: L.append('\\midrule'); continue
    name,t,tag=r
    m=M.get(f'matchT/{tag}') or ZSM.get(FALL.get(tag,''))
    if m is None:  # models not in a pair: compute from robust/cons only
        pass
    rbv=rb[f'matchT/{tag}']; c=cm[tag]
    pass1=m['pass1'] if m else rbv['pass1_model']; pass32=m['pass32'] if m else rbv['pass32_model']
    distinct=m['distinct'] if m else rbv['distinct_model']
    tvs=m['tv_samples'] if m else None; tvl=m['tv_list'] if m else rbv['tv_list_all']
    cells=[t, f"{100*rbv['invalid_rate']:.1f}", f"{pass1:.3f}", f"{pass32:.3f}", f"{distinct:.2f}",
           (f"{tvs:.3f}" if tvs is not None else '---'), f"{tvl:.3f}", f"${100*(c['vote@32']-c['single']):+.1f}$", f"{c['auroc_stated_p']:.3f}", f"{c['auroc_consist@32']:.3f}"]
    L.append(name+' & '+' & '.join(cells)+' \\\\')
T2=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & & Invalid & & & & TV & TV & Vote@32 & \multicolumn{2}{c}{AUROC} \\
Model & $T$ & (\%) & pass@1 & pass@32 & Distinct & samples & one list & gain & stated & agree@32 \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Matched-temperature control (DDXPlus, 980 cases, 32 samples, zero-shot prompt): the post-trained models sampled at higher temperatures (top-$p$ 1) against their base models at $T{=}0.7$ and $T{=}1$. Columns as in Tables~\ref{tab:sampled},~\ref{tab:stated} and~\ref{tab:consequences}.}
\label{tab:matcht}
\end{table}"""
open('r4_tables.tex','w').write(T1+'\n\n'+T2+'\n'); print(T1); print(T2)
