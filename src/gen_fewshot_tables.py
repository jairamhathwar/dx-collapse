import json
z=json.load(open('review2.json')); f=json.load(open('review2_fewshot.json'))
rob=json.load(open('robust_fewshot.json')); con=json.load(open('cons_fewshot.json'))['models']
P=[('pair_q3','qwen3-8b-base','qwen3-8b','Qwen3-8B-Base','Qwen3-8B','Qwen3-8B'),
   ('pair25_t07','qwen25-7b-base','qwen25-7b-it','Qwen2.5-7B (base)','Qwen2.5-7B-Instruct','Qwen2.5-7B'),
   ('big32_ddx','qwen25-32b-base','qwen25-32b-it','Qwen2.5-32B (base)','Qwen2.5-32B-Instruct','Qwen2.5-32B'),
   ('ddx_olmo','olmo3-base','olmo3-instruct','OLMo-3-1025-7B (base)','OLMo-3-7B-Instruct','OLMo~3 7B'),
   ('big32_ddx','olmo31-32b-base','olmo31-32b-instruct','OLMo-3-1125-32B (base)','OLMo-3.1-32B-Instruct','OLMo~3.1 32B')]
cols=[('pass1',1),('pass32',1),('cov32',1),('tv_list',-1),('tv_list_nogold',-1),('ddf1',1),('tv_samples',-1)]
L=[]
for i,(d,b,p,nb,np_,_) in enumerate(P):
    vals={}
    for t in (b,p):
        m=f['models'][f'fewshot_t07/{t}']; c=con[t]
        vals[t]=[m[k] for k,_ in cols]
    for t,name in ((b,nb),(p,np_)):
        m=f['models'][f'fewshot_t07/{t}']; c=con[t]; r=rob[f'fewshot_t07/{t}']
        cells=[f"{100*r['invalid_rate']:.1f}"]
        for j,(k,s) in enumerate(cols):
            o=b if t==p else p
            v=vals[t][j]; best=(v*s>vals[o][j]*s)
            x=f"{v:.3f}"; cells.append(f"\\textbf{{{x}}}" if best else x)
        cells.append(f"${100*(c['vote@32']-c['single']):+.1f}$")
        cells.append(f"${100*(c['top3_list1']-c['top3_vote@32']):+.1f}$")
        cells.append(f"{c['auroc_stated_p']:.3f}"); cells.append(f"{c['auroc_consist@5']:.3f}")
        L.append(name+' & '+' & '.join(cells)+' \\\\')
    if i<len(P)-1: L.append('\\midrule')
T1=r"""\begin{table}[t]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
 & Invalid & & & & \multicolumn{2}{c}{TV, one list} & & TV & Vote@32 & Top-3 list & \multicolumn{2}{c}{AUROC} \\
Model & (\%) & pass@1 & pass@32 & Cov@32 & all & w/o true & DDF1 & samples & gain & $-$ samples & stated & agree@5 \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Few-shot elicitation on DDXPlus ($T{=}0.7$, 980 cases, 32 samples; three worked examples in the prompt for every model, \S\ref{sec:fewshot}). Invalid: samples naming no listed condition. TV: distance to the reference differential, for one stated list (with the true pathology removed from both in ``w/o true'') and for the sampled top answers. Vote@32 gain and top-3 list minus top-3 of the samples are in accuracy points. AUROC: for the single-call answer being correct, of its stated probability and of agreement with the next 4 samples. Bold: better of the pair.}
\label{tab:fewshot}
\end{table}"""
rows=[]
keys=[('pass32',100,'{:+.1f}'),('tv_samples',1,'{:+.3f}'),('tv_list',1,'{:+.3f}'),('tv_list_nogold',1,'{:+.3f}'),('ddf1',1,'{:+.3f}'),('mass_list',1,'{:+.3f}')]
for d,b,p,_,_,lab in P:
    cells=[]
    for k,sc,fmt in keys:
        out=[]
        for src,dd in ((z,d),(f,'fewshot_t07')):
            r=src['pairs'][f'{dd}:{b}>{p}'][k]
            s=fmt.format(sc*r['diff']).replace('+0.000','0.000')
            out.append(f"${s}$"+("$^{\\dagger}$" if r['p_holm']>=0.05 else ''))
        cells.append(' / '.join(out))
    rows.append(lab+' & '+' & '.join(cells)+' \\\\')
T2=r"""\begin{table}[t]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lllllll}
\toprule
 & pass@32 & TV & \multicolumn{2}{c}{TV, one list} & & Reference mass \\
Pair (base $\rightarrow$ post) & (points) & samples & all & w/o true & DDF1 & of one list \\
\midrule
"""+'\n'.join(rows)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Base $\rightarrow$ post-trained differences under the zero-shot prompt / the few-shot prompt (DDXPlus, $T{=}0.7$). $^{\dagger}$: not significant under the condition-clustered bootstrap with Holm correction (over 112 tests for the zero-shot and 70 for the few-shot pairs; Appendix~\ref{app:clustered}).}
\label{tab:fewshotdiff}
\end{table}"""
open('fewshot_tables.tex','w').write(T1+'\n\n'+T2+'\n')
print(T1); print(T2)
