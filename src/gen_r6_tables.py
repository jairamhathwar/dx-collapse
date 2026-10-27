import json
norm=lambda d:{k.split('/runs/')[-1]:v for k,v in d.items()}
zs=json.load(open('r6/review2_zs.json')); st=json.load(open('r3/review2_stages.json'))
M=norm(st['models']); M.update(norm(zs['models']))
out={}
# ---- Table 1 (sampled) from the analysis scorer; list-sampled columns from robustness.json
rb=json.load(open('fsan/robustness.json'))
rows=[('Qwen3-8B, $T{=}0.7$','pair_q3',[('base','qwen3-8b-base'),('post-trained','qwen3-8b')]),
      ('Qwen3-8B, $T{=}1$','ddx_qwen3_t1',[('base','qwen3-8b-base'),('post-trained','qwen3-8b')]),
      ('Qwen2.5-7B, $T{=}0.7$','pair25_t07',[('base','qwen25-7b-base'),('instruct','qwen25-7b-it')]),
      ('Qwen2.5-7B, $T{=}1$','pair25_t1',[('base','qwen25-7b-base'),('instruct','qwen25-7b-it')]),
      ('Qwen2.5-32B, $T{=}0.7$','big32_ddx',[('base','qwen25-32b-base'),('instruct','qwen25-32b-it')]),
      None,
      ('OLMo~3 7B, $T{=}0.7$','ddx_olmo',[('base','olmo3-base'),('SFT+DPO+RLVR','olmo3-instruct'),('RL-Zero (from base)','olmo3-rlzero-general')]),
      ('OLMo~3.1 32B, $T{=}0.7$','big32_ddx',[('base','olmo31-32b-base'),('SFT+DPO+RLVR','olmo31-32b-instruct')])]
L=[]
for r in rows:
    if r is None: L.append('\\midrule'); continue
    name,d,ms=r
    vals={t:M[f'{d}/{t}'] for _,t in ms}; rbs={t:rb[f'{d}/{t}'] for _,t in ms}
    cols=[('pass1','{:.3f}'),('pass32','{:.3f}'),('distinct','{:.2f}'),('cov32','{:.3f}')]
    lcols=[('pass32_listsample','{:.3f}'),('distinct_listsample','{:.2f}')]
    for i,(lab,t) in enumerate(ms):
        cells=[]
        for k,f in cols:
            v=vals[t][k]; best=max(vals[x][k] for _,x in ms); s=f.format(v); cells.append(f'\\textbf{{{s}}}' if v==best else s)
        k,f=lcols[0]; v=rbs[t][k]; best=max(rbs[x][k] for _,x in ms); s=f.format(v); cells.append(f'\\textbf{{{s}}}' if v==best else s)
        cells.append('{:.2f}'.format(rbs[t]['distinct_listsample']))
        L.append((name if i==0 else '')+' & '+lab+' & '+' & '.join(cells)+' \\\\')
out['T1rows']='\n'.join(L)
# ---- Table 2 rows (stated) with conditional TV and recall@5
P=[('Qwen3-8B','pair_q3','qwen3-8b-base','qwen3-8b','post-trained'),('Qwen2.5-7B','pair25_t07','qwen25-7b-base','qwen25-7b-it','instruct'),
   ('Qwen2.5-32B','big32_ddx','qwen25-32b-base','qwen25-32b-it','instruct'),('OLMo~3 7B','ddx_olmo','olmo3-base','olmo3-instruct','Instruct (RLVR)'),
   ('OLMo~3.1 32B','big32_ddx','olmo31-32b-base','olmo31-32b-instruct','Instruct (RLVR)')]
cols=[('tv_list',-1),('tv_samples',-1),('tv_list_nogold',-1),('mass_list',1),('cond_tv',-1),('ddf1',1),('rec5',1),('sev_named',1)]
L=[]
for name,d,b,p,lab in P:
    mb,mp=M[f'{d}/{b}'],M[f'{d}/{p}']
    for i,(tag,m,o) in enumerate([('base',mb,mp),(lab,mp,mb)]):
        cells=[]
        for k,s in cols:
            v=m[k]; x=f'{v:.3f}'; cells.append(f'\\textbf{{{x}}}' if v*s>o[k]*s else x)
        L.append((name if i==0 else '')+' & '+tag+' & '+' & '.join(cells)+' \\\\')
out['T2rows']='\n'.join(L)
# ---- packing table
pk={}; [pk.update(json.load(open(f'r5/{f}'))) for f in ['packing_zs.json','packing_fs.json']]
lab={'pair_q3:qwen3-8b-base>qwen3-8b':'Qwen3-8B, $T{=}0.7$','ddx_qwen3_t1:qwen3-8b-base>qwen3-8b':'Qwen3-8B, $T{=}1$','pair25_t07:qwen25-7b-base>qwen25-7b-it':'Qwen2.5-7B, $T{=}0.7$',
     'pair25_t1:qwen25-7b-base>qwen25-7b-it':'Qwen2.5-7B, $T{=}1$','big32_ddx:qwen25-32b-base>qwen25-32b-it':'Qwen2.5-32B, $T{=}0.7$','ddx_olmo:olmo3-base>olmo3-instruct':'OLMo~3 7B, $T{=}0.7$',
     'ddx_olmo_t1:olmo3-base>olmo3-instruct':'OLMo~3 7B, $T{=}1$','big32_ddx:olmo31-32b-base>olmo31-32b-instruct':'OLMo~3.1 32B, $T{=}0.7$'}
L=[]; prev=None
for k,v in pk.items():
    kk=k.split('/runs/')[-1]
    grp='fs' if 'fewshot' in kk else ('th' if 'think' in kk else 'zs')
    if grp!=prev:
        if prev: L.append('\\midrule')
        L.append({'zs':'\\multicolumn{8}{l}{\\emph{Zero-shot prompt}} \\\\','fs':'\\multicolumn{8}{l}{\\emph{Few-shot prompt}} \\\\','th':'\\multicolumn{8}{l}{\\emph{Thinking on}} \\\\'}[grp]); prev=grp
    name=lab.get(kk) or {'qwen3-8b-base>qwen3-8b':'Qwen3-8B','qwen25-7b-base>qwen25-7b-it':'Qwen2.5-7B','qwen25-32b-base>qwen25-32b-it':'Qwen2.5-32B','olmo3-base>olmo3-instruct':'OLMo~3 7B','olmo31-32b-base>olmo31-32b-instruct':'OLMo~3.1 32B','mistral7b-base>mistral7b-it':'Mistral-7B','qwen3-8b-base>qwen3-8b-think':'Qwen3-8B, thinking'}[kk.split(':')[1]]
    m=v['means']; dd=v['diffs']
    def cell(key):
        x=dd[key]; sig=x['cluster_ci'][0]>0 or x['cluster_ci'][1]<0; return f"${x['diff']:+.3f}$"+('' if sig else '$^{\\dagger}$')
    L.append(f"{name} & {m['own']:.3f} & {m['same_gold']:.3f} & {m['ref']:.3f} & {m['base_loo']:.3f} & {m['post_loo']:.3f} & {cell('own-same_gold')} & {cell('own-base_loo')} \\\\")
out['packing']=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{3}{c}{TV of a post-trained list to} & \multicolumn{2}{c}{Within-model benchmark} & \multicolumn{2}{c}{Differences} \\
Pair & own base mean & same-diagnosis means & reference & base list & post list & own $-$ same dx & own $-$ base list \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Does a post-trained list reproduce its base model's view of the patient? (DDXPlus, 980 cases.) TV between a post-trained list and the mean of the base model's valid lists for the same case (own base mean), the base model's means for the other cases with the same true pathology, and the reference. Within-model benchmarks: a base list against the mean of the base model's other lists for the case, and a post-trained list against the mean of the post-trained model's other lists. $^{\dagger}$: the condition-clustered 95\% interval includes zero.}
\label{tab:packing}
\end{table}"""
# ---- ambiguity table
CS={'pair_q3':['qwen3-8b-base','qwen3-8b'],'pair25_t07':['qwen25-7b-base','qwen25-7b-it'],'big32_q':['qwen25-32b-base','qwen25-32b-it'],'ddx_olmo':['olmo3-base','olmo3-instruct'],'big32_o':['olmo31-32b-base','olmo31-32b-instruct']}
nm={'qwen3-8b-base':'Qwen3-8B-Base','qwen3-8b':'Qwen3-8B','qwen25-7b-base':'Qwen2.5-7B','qwen25-7b-it':'Qwen2.5-7B-Instruct','qwen25-32b-base':'Qwen2.5-32B','qwen25-32b-it':'Qwen2.5-32B-Instruct','olmo3-base':'OLMo-3-1025-7B','olmo3-instruct':'OLMo-3-7B-Instruct','olmo31-32b-base':'OLMo-3-1125-32B','olmo31-32b-instruct':'OLMo-3.1-32B-Instruct'}
L=[]
for f,ts in CS.items():
    d=json.load(open(f'cons/{f}.json'))['models']
    for t in ts:
        m=d[t]; L.append(f"{nm[t]} & ${m['rho_stated_p']:+.3f}$ & ${m['rho_consist@5']:+.3f}$ & ${m['rho_consist@32']:+.3f}$ \\\\")
out['ambiguity']=r"""\begin{table}[t]
\centering\small
\begin{tabular}{lccc}
\toprule
Model ($T{=}0.7$) & Stated $p$ & Agree@5 & Agree@32 \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\caption{Spearman correlation between each uncertainty signal (lower confidence or agreement) and the entropy of the case's reference differential (DDXPlus, 980 cases). Neither readout tracks how ambiguous a case is.}
\label{tab:ambiguity}
\end{table}"""
# ---- severe test table with mean shift
Z=[('Qwen3-8B, $T{=}0.7$','pair_q3:qwen3-8b-base>qwen3-8b'),('Qwen3-8B, $T{=}1$','ddx_qwen3_t1:qwen3-8b-base>qwen3-8b'),('Qwen2.5-7B, $T{=}0.7$','pair25_t07:qwen25-7b-base>qwen25-7b-it'),
   ('Qwen2.5-7B, $T{=}1$','pair25_t1:qwen25-7b-base>qwen25-7b-it'),('Qwen2.5-32B, $T{=}0.7$','big32_ddx:qwen25-32b-base>qwen25-32b-it'),('OLMo~3 7B, $T{=}0.7$','ddx_olmo:olmo3-base>olmo3-instruct'),
   ('OLMo~3 7B, $T{=}1$','ddx_olmo_t1:olmo3-base>olmo3-instruct'),('OLMo~3.1 32B, $T{=}0.7$','big32_ddx:olmo31-32b-base>olmo31-32b-instruct')]
fs=json.load(open('r6/review2_fs.json')); th=json.load(open('r6/review2_third.json'))
FSs={k.split('/runs/')[-1]:v for k,v in fs['severe_test'].items()}; THs={k.split('/runs/')[-1]:v for k,v in th['severe_test'].items()}
F=[('Qwen3-8B, few-shot','fewshot_t07:qwen3-8b-base>qwen3-8b'),('Qwen2.5-7B, few-shot','fewshot_t07:qwen25-7b-base>qwen25-7b-it'),('Qwen2.5-32B, few-shot','fewshot_t07:qwen25-32b-base>qwen25-32b-it'),
   ('OLMo~3 7B, few-shot','fewshot_t07:olmo3-base>olmo3-instruct'),('OLMo~3.1 32B, few-shot','fewshot_t07:olmo31-32b-base>olmo31-32b-instruct')]
def row(name,s):
    worst=s['worst'][0][1].replace('Possible NSTEMI / STEMI','NSTEMI/STEMI').replace('Acute pulmonary edema','acute pulmonary edema').replace('Stable angina','stable angina').replace('Pulmonary embolism','pulmonary embolism').replace('Larygospasm','laryngospasm')
    return f"{name} & {s['lower']} / {s['n_conditions']-s['lower']} & {s['sig_lower_q05']} / {s['sig_higher_q05']} & {s['null_lower_p_ge_obs']:.2f} & ${s['mean_shift']:+.3f}$ & {s['mean_shift_p']:.3f} & {worst} ({s['worst'][0][0]:+.2f}) \\\\"
L=[row(n,zs['severe_test'][k]) for n,k in Z]+['\\midrule']+[row(n,FSs[k]) for n,k in F]+[row('Mistral-7B, few-shot',THs['fewshot_t07:mistral7b-base>mistral7b-it'])]
out['severe']=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccccl}
\toprule
 & Named less / & Significant & $P$(as many & Mean shift & & Largest loss \\
Pair (base $\rightarrow$ post) & more often & less / more & less often) & (17 conditions) & $P$ & (shift) \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{The 17 severe conditions one by one (DDXPlus). For each condition, the share of a model's lists that name it, over the cases whose reference gives it any weight; post-trained minus base, with intervals that resample the true pathologies (10,000 resamples) and Benjamini--Hochberg correction across the 17 conditions. $P$ columns compare the count of conditions named less often and the mean shift across conditions with a null that swaps each case's base and post-trained lists at random (2,000 swaps); the count test is weak against an overall upward shift, so the mean shift and the largest single loss are shown too.}
\label{tab:severetest}
\end{table}"""
json.dump(out,open('r6_tables.json','w'))
print(out['T1rows']); print(out['T2rows']); print('\n'.join(L))
