import json
T='third/'; F='fsan/'
r2=json.load(open(T+'review2.json'))['models']
rb3=json.load(open(T+'robust.json')); rb0=json.load(open(F+'robustness.json'))
cf=json.load(open(T+'cons_fewshot_t07.json'))['models']; ct=json.load(open(T+'cons_think_t07.json'))['models']
rows=[('Mistral-7B-v0.3 (base), few-shot','fewshot_t07/mistral7b-base',rb3['fewshot_t07/mistral7b-base'],cf['mistral7b-base']),
      ('Mistral-7B-Instruct-v0.3, few-shot','fewshot_t07/mistral7b-it',rb3['fewshot_t07/mistral7b-it'],cf['mistral7b-it']),
      None,
      ('Qwen3-8B-Base','think_t07/qwen3-8b-base',rb0['pair_q3/qwen3-8b-base'],ct['qwen3-8b-base']),
      ('Qwen3-8B, non-thinking','think_t07/qwen3-8b',rb0['pair_q3/qwen3-8b'],ct['qwen3-8b']),
      ('Qwen3-8B, thinking','think_t07/qwen3-8b-think',rb3['think_t07/qwen3-8b-think'],ct['qwen3-8b-think'])]
cols=[('pass1',1,'{:.3f}'),('pass32',1,'{:.3f}'),('distinct',1,'{:.2f}'),('cov32',1,'{:.3f}'),('tv_list',-1,'{:.3f}'),('tv_list_nogold',-1,'{:.3f}'),('ddf1',1,'{:.3f}'),('tv_samples',-1,'{:.3f}')]
blocks=[[r for r in rows[:2]],[r for r in rows[3:]]]
L=[]
for bi,blk in enumerate(blocks):
    for name,key,rb,c in blk:
        m=r2[key]; cells=[f"{100*rb['invalid_rate']:.1f}"]
        for k,s,fmt in cols:
            best=max(r2[x[1]][k]*s for x in blk)
            v=m[k]; x=fmt.format(v); cells.append(f"\\textbf{{{x}}}" if abs(v*s-best)<1e-12 else x)
        cells += [f"${100*(c['vote@32']-c['single']):+.1f}$", f"${100*(c['top3_list1']-c['top3_vote@32']):+.1f}$", f"{c['auroc_stated_p']:.3f}", f"{c['auroc_consist@5']:.3f}"]
        L.append(name+' & '+' & '.join(cells)+' \\\\')
    if bi==0: L.append('\\midrule')
T1=r"""\begin{table}[t]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrrrrrrr}
\toprule
 & Invalid & & & & & \multicolumn{2}{c}{TV, one list} & & TV & Vote@32 & Top-3 list & \multicolumn{2}{c}{AUROC} \\
Model & (\%) & pass@1 & pass@32 & Distinct & Cov@32 & all & w/o true & DDF1 & samples & gain & $-$ samples & stated & agree@5 \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{A third family and a reasoning mode on DDXPlus ($T{=}0.7$, 980 cases, 32 samples). Top: Mistral-7B v0.3 with the few-shot prompt of \S\ref{sec:fewshot}. Bottom: Qwen3-8B-Base and Qwen3-8B with thinking off (the main runs) and on, with the zero-shot prompt. Columns as in Table~\ref{tab:fewshot}; Distinct: distinct top answers per 32 samples. Bold: best within each block.}
\label{tab:third}
\end{table}"""
# aggregation table
R2={}
for fn in ['r3/review2_zs.json','r3/review2_fs.json','r3/review2_third.json']:
    R2.update({k.split('/runs/')[-1]:v for k,v in json.load(open(fn))['models'].items()})
zs=json.load(open(F+'aggregate_zs.json')); fs=json.load(open(F+'aggregate_fs.json')); th=json.load(open(T+'aggregate.json'))
FP='/mmfs1/home/jairamh/thesis_overflow/runs/fewshot_t07'
spec=[('zs',zs,'pair_q3','qwen3-8b-base','qwen3-8b','Qwen3-8B'),('zs',zs,'pair25_t07','qwen25-7b-base','qwen25-7b-it','Qwen2.5-7B'),
      ('zs',zs,'big32_ddx','qwen25-32b-base','qwen25-32b-it','Qwen2.5-32B'),('zs',zs,'ddx_olmo','olmo3-base','olmo3-instruct','OLMo~3 7B'),
      ('zs',zs,'big32_ddx','olmo31-32b-base','olmo31-32b-instruct','OLMo~3.1 32B'),
      ('fs',fs,FP,'qwen3-8b-base','qwen3-8b','Qwen3-8B'),('fs',fs,FP,'qwen25-7b-base','qwen25-7b-it','Qwen2.5-7B'),
      ('fs',fs,FP,'qwen25-32b-base','qwen25-32b-it','Qwen2.5-32B'),('fs',fs,FP,'olmo3-base','olmo3-instruct','OLMo~3 7B'),
      ('fs',fs,FP,'olmo31-32b-base','olmo31-32b-instruct','OLMo~3.1 32B'),('fs',th,'fewshot_t07','mistral7b-base','mistral7b-it','Mistral-7B v0.3'),
      ('th',th,'think_t07','qwen3-8b-base','qwen3-8b-think','Qwen3-8B, thinking')]
L=[]; prev=None
lab={'zs':'\\multicolumn{9}{l}{\\emph{Zero-shot prompt}} \\\\','fs':'\\multicolumn{9}{l}{\\emph{Few-shot prompt}} \\\\','th':'\\multicolumn{9}{l}{\\emph{Zero-shot prompt, thinking on}} \\\\'}
for g,R,d,b,p,name in spec:
    if g!=prev:
        if prev: L.append('\\midrule')
        L.append(lab[g]); prev=g
    mb=R['models'][f'{d}/{b}']; mp=R['models'][f'{d}/{p}']
    c=R['pairs'][f'{d}:{b}>{p}']['comparisons']
    tv=c['post_agg@5-base_agg@5:tv']; sv=c['post_agg@5-base_agg@5:sev']
    def ci(x,fmt='{:+.3f}'):
        sig = x['cluster_ci'][0]>0 or x['cluster_ci'][1]<0
        return f"${fmt.format(x['diff'])}$"+('' if sig else '$^{\\dagger}$')
    kb=f"{d}/{b}".split('/runs/')[-1]; kp=f"{d}/{p}".split('/runs/')[-1]
    cells=[f"{R2[kb]['tv_list']:.3f}"]+[f"{mb[f'agg@{m}:tv']:.3f}" for m in (5,32)]+[f"{R2[kp]['tv_list']:.3f}"]+[f"{mp[f'agg@{m}:tv']:.3f}" for m in (5,32)]+[ci(tv),ci(sv)]
    L.append(name+' & '+' & '.join(cells)+' \\\\')
T2=r"""\begin{table}[t]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{TV, base model} & \multicolumn{3}{c}{TV, post-trained model} & \multicolumn{2}{c}{Post $-$ base, 5 lists each} \\
Pair & 1 list & mean of 5 & mean of 32 & 1 list & mean of 5 & mean of 32 & TV & Severe named \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Averaging stated lists across calls (DDXPlus, $T{=}0.7$, 980 cases). TV to the reference differential of one stated list and of the mean of the first 5 or 32 valid lists, for the base and the final post-trained model. Last two columns: post-trained minus base with five averaged lists each, for TV and for the share of severe conditions with reference probability $\geq 0.1$ that are named. $^{\dagger}$: the condition-clustered 95\% interval includes zero.}
\label{tab:aggregate}
\end{table}"""
open('third_tables.tex','w').write(T1+'\n\n'+T2+'\n'); print(T1); print(T2)
