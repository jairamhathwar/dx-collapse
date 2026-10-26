import json
R='r3/'
zs=json.load(open(R+'review2_zs.json')); fs=json.load(open(R+'review2_fs.json')); th=json.load(open(R+'review2_third.json'))
norm=lambda d:{k.split('/runs/')[-1]:v for k,v in d.items()}
M={}; [M.update(norm(x['models'])) for x in (zs,fs,th)]
P={}; [P.update(norm(x['pairs'])) for x in (zs,fs,th)]
out={}
# ---- Table 2 (stated vs sampled), T=0.7 pairs
rows=[('Qwen3-8B','pair_q3','qwen3-8b-base','qwen3-8b','post-trained'),('Qwen2.5-7B','pair25_t07','qwen25-7b-base','qwen25-7b-it','instruct'),
      ('Qwen2.5-32B','big32_ddx','qwen25-32b-base','qwen25-32b-it','instruct'),('OLMo~3 7B','ddx_olmo','olmo3-base','olmo3-instruct','Instruct (RLVR)'),
      ('OLMo~3.1 32B','big32_ddx','olmo31-32b-base','olmo31-32b-instruct','Instruct (RLVR)')]
cols=[('tv_list',-1),('tv_samples',-1),('tv_list_nogold',-1),('mass_list',1),('u',-1),('ddf1',1),('sev_named',1)]
def val(m,k): return m['tv_list']-(1-m['mass_list']) if k=='u' else m[k]
L=[]
for name,d,b,p,lab in rows:
    mb,mp=M[f'{d}/{b}'],M[f'{d}/{p}']
    for first,(tag,m,o) in enumerate([('base',mb,mp),(lab,mp,mb)]):
        cells=[]
        for k,s in cols:
            v=val(m,k); x=f"{v:.3f}"; cells.append(f"\\textbf{{{x}}}" if v*s>val(o,k)*s else x)
        L.append((name if first==0 else '')+' & '+tag+' & '+' & '.join(cells)+' \\\\')
T2=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llccccccc}
\toprule
 & & \multicolumn{3}{c}{TV to reference$\downarrow$} & Reference & Misallo- & & Severe \\
Comparison ($T{=}0.7$) & Model & One list & Samples & List w/o gold & mass named & cated$\downarrow$ & DDF1 & named \\
\midrule
"""+'\n'.join(L)+r"""
\midrule
Chance & patient-blind & 0.794 & 0.774 & --- & 0.238 & 0.032 & 0.272 & 0.149 \\
Oracle & reference & 0.265 & 0.18--0.25 & --- & 0.735 & 0.000 & --- & 0.975 \\
 & one answer: reference mode & --- & 0.694 & --- & --- & --- & --- & --- \\
 & one answer: true pathology & --- & 0.714 & --- & --- & --- & --- & --- \\
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{Stated vs.\ sampled readouts on DDXPlus ($T{=}0.7$, 980 cases, valid samples). \emph{One list}: one sample's stated differential (per-case mean over valid samples). \emph{Samples}: empirical distribution of top answers. \emph{List w/o gold}: TV after removing the true pathology from list and reference. A list's TV is exactly the reference mass it leaves out ($1-$ mass named) plus the mass it \emph{misallocates} among the conditions it names. DDF1: set F1 of the named conditions against the reference's support. \emph{Severe named}: share of severe conditions with reference probability $\geq 0.1$ that one list names. Chance: the top five of the average reference differential as a fixed list, and 32 draws from that average as samples. Oracle: the reference's own top five as the list and as many reference draws as valid samples; and samplers that always give one answer, each case's reference mode or its true pathology. Bold: better model of each pair.}
\label{tab:stated}
\end{table}"""
out['T2']=T2
# ---- tab:clustered
ZS=[('Qwen3-8B, $T{=}0.7$','pair_q3:qwen3-8b-base>qwen3-8b'),('Qwen3-8B, $T{=}1$','ddx_qwen3_t1:qwen3-8b-base>qwen3-8b'),
    ('Qwen2.5-7B, $T{=}0.7$','pair25_t07:qwen25-7b-base>qwen25-7b-it'),('Qwen2.5-7B, $T{=}1$','pair25_t1:qwen25-7b-base>qwen25-7b-it'),
    ('Qwen2.5-32B, $T{=}0.7$','big32_ddx:qwen25-32b-base>qwen25-32b-it'),('OLMo~3 7B, $T{=}0.7$','ddx_olmo:olmo3-base>olmo3-instruct'),
    ('OLMo~3 7B, $T{=}1$','ddx_olmo_t1:olmo3-base>olmo3-instruct'),('OLMo~3.1 32B, $T{=}0.7$','big32_ddx:olmo31-32b-base>olmo31-32b-instruct')]
cc=[('pass32',100,'{:+.1f}'),('distinct',1,'{:+.2f}'),('tv_samples',1,'{:+.3f}'),('tv_list',1,'{:+.3f}'),('tv_list_nogold',1,'{:+.3f}'),('ddf1',1,'{:+.3f}'),('auroc_agree32',1,'{:+.3f}')]
L=[]
for name,key in ZS:
    v=P[key]; cells=[]
    for k,sc,f in cc:
        x=v[k]; s=f"${f.format(sc*x['diff'])}$"+('' if x['p_holm']<0.05 else '$^{\\dagger}$')
        ci=f"{{\\scriptsize $[{f.format(sc*x['cluster_ci'][0])},{f.format(sc*x['cluster_ci'][1])}]$}}"
        cells.append(s+' \\newline '+ci)
    L.append(name+' & '+' & '.join(cells)+' \\\\')
out['clustered_rows']='\n'.join(L)
# ---- severe test table
L=[]
for name,key in ZS:
    st=zs['severe_test'][key]
    L.append(f"{name} & {st['lower']} / {st['higher']} & {st['sig_lower_q05']} / {st['sig_higher_q05']} & {st['null_lower_mean']:.1f} & {st['null_lower_p_ge_obs']:.2f} \\\\")
TS=r"""\begin{table}[t]
\centering\small
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccc}
\toprule
 & Named less / more often & Significant (FDR 5\%) & Expected less & $P$(at least as \\
Pair (base $\rightarrow$ post) & (point estimates) & less / more & often by chance & many less) \\
\midrule
"""+'\n'.join(L)+r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\caption{The 17 severe conditions one by one (DDXPlus). For each condition, the share of a model's lists that name it, over the cases whose reference gives it any weight; differences post-trained minus base with intervals that resample the true pathologies (10,000 resamples) and Benjamini--Hochberg correction across the 17 conditions. The last two columns compare the number of conditions named less often with a null that swaps each case's base and post-trained lists at random (2,000 swaps).}
\label{tab:severetest}
\end{table}"""
out['TS']=TS
json.dump(out,open('r3_tables.json','w'))
print(T2); print(out['clustered_rows'][:600]); print(TS)
