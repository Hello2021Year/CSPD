"""Figures 3 and 4 and Appendix B (composition dynamics, progressive training, appending, leave-one-out).

Values are the published rounded scores; reported aggregates are kept as published rather than
recomputed from rounded per-domain values. Expert TIES uses the same domain subsets at each stage.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
out=Path(__file__).resolve().parents[1]/'figures'
out.mkdir(parents=True, exist_ok=True)
student=[55.94,58.61,44.38,42.71,50.41]
ind=np.array([student,[68.49,62.57,47.03,65.87,60.99],[70.73,60.55,48.71,66.71,61.67],[72.34,66.54,51.23,64.58,63.67],[80.47,64.67,57.77,68.42,67.83]])
seq=np.array([student,ind[1],[64.53,59.21,49.31,52.71,56.44],[68.85,65.87,47.51,50.88,58.28],[81.41,60.18,57.67,61.42,65.17]])
merge=np.array([student,[66.35,52.69,48.71,67.33,58.77],[67.34,52.54,51.20,64.83,58.98],[70.42,66.32,51.29,65.92,63.49],[81.25,65.42,57.07,66.13,67.47]])
# Panel (a): appended order I, IS, ISC, ISCM (final = Table 4 composition); source as plot_modular_composition.py.
iscm=np.array([student,[68.49,62.57,47.03,65.87,60.99],[71.25,61.83,50.13,65.96,62.29],[74.69,64.97,52.97,61.71,63.58],[79.90,65.72,57.92,67.33,67.72]])
labels=['Student','I','SI','CSI','MCSI']; colors=['#2878b5','#32954a','#9260ad','#d34b49','#171717']
fig,(ax,h)=plt.subplots(1,2,figsize=(8.1,3.35),gridspec_kw={'width_ratios':[1.2,1]},layout='constrained')
for j,(name,c) in enumerate(zip(['Math','Code','Science','IF','Overall'],colors)):
 ax.plot(range(5),ind[:,j],marker='o' if j<4 else 's',label=name,color=c,lw=1.4 if j<4 else 2,ms=3.6)
ax.set(title='(a) Continual addition',xticks=range(5),xticklabels=['Student','I','SI','CSI','MCSI'],ylabel='Score (%)',ylim=(39,85),xlabel='Input order (new module prepended)')
for x,y in enumerate(ind[:,4]): ax.annotate(f'{y:.1f}',(x,y),xytext=(0,7 if x in (0,4) else -12),textcoords='offset points',ha='center',fontsize=7.5,fontweight='bold')
ax.axhline(66.61,color='#171717',ls='--',lw=.8,alpha=.6); ax.text(4.25,66.61,'Expert\navg.',fontsize=7,va='center',style='italic')
ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.18)
ax.legend(ncol=3,fontsize=7,loc='upper left',frameon=False,columnspacing=.8,handlelength=1.2)
full=ind[-1,:4]
removed=np.array([[72.34,66.54,51.23,64.58],[76.93,60.25,58.14,68.75],[80.68,65.79,56.66,65.92],[80.73,64.37,56.88,55.54]])
d=np.column_stack((removed-full,[63.67-67.83,66.02-67.83,67.26-67.83,64.38-67.83]))
im=h.imshow(d,cmap='RdBu',vmin=-13,vmax=13,aspect='auto')
for i in range(4):
 for j in range(5):h.text(j,i,f'{d[i,j]:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(d[i,j])>7 else '#171717')
h.set(title='(b) Removing one prompt from MCSI',xticks=range(5),xticklabels=['AIME24','LCB','GPQA-D','IFBench','Avg.'],yticks=range(4),yticklabels=['w/o Math','w/o Code','w/o Science','w/o IF'],xlabel='Evaluated benchmark')
h.tick_params(axis='x',labelsize=8);h.axvline(3.5,color='white',lw=2)
fig.colorbar(im,ax=h,orientation='vertical',fraction=.05,pad=.025,label='Score change after removal (pp)')
fig.savefig(out/'composition_dynamics.pdf');fig.savefig(out/'composition_dynamics.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,5,figsize=(11,2.35),layout='constrained')
for j,ax in enumerate(axes):
 for arr,name,c,marker,ls in [(ind,'Independent SP','#2878b5','o','-'),(seq,'Progressive SP','#d46b30','s','-'),(merge,'Expert TIES','#777777','^','--')]:
  ax.plot(range(5),arr[:,j],label=name,color=c,marker=marker,lw=1.5,ms=3,ls=ls)
 ax.set_title(['Math (AIME24)','Code (LCB v5)','Science (GPQA-D)','IF (IFBench)','Overall'][j],fontsize=9)
 ax.set_xticks(range(5),labels,fontsize=7,rotation=30)
 ax.tick_params(axis='y',labelsize=7)
 ax.grid(axis='y',alpha=.2)
 if j==0: ax.set_ylabel('Score (%)',fontsize=8)
fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False,fontsize=8.5)
fig.savefig(out/'progressive_domain_comparison.pdf');fig.savefig(out/'progressive_domain_comparison.png',dpi=180)

# Appendix: prepended (main text, Figure 3a) vs appended order, side by side.
fig,axs=plt.subplots(1,2,figsize=(8.1,3.2),layout='constrained',sharey=True)
for ax,arr,xl,title in [(axs[0],ind,['Student','I','SI','CSI','MCSI'],'(a) Prepend new prompt (main text, Fig. 3a)'),
                        (axs[1],iscm,['Student','I','IS','ISC','ISCM'],'(b) Append new prompt')]:
 for j,(name,c) in enumerate(zip(['Math','Code','Science','IF','Overall'],colors)):
  ax.plot(range(5),arr[:,j],marker='o' if j<4 else 's',label=name,color=c,lw=1.4 if j<4 else 2,ms=3.6)
 ax.set(title=title,xticks=range(5),xticklabels=xl,ylim=(39,85),xlabel='Input order')
 for x,y in enumerate(arr[:,4]): ax.annotate(f'{y:.1f}',(x,y),xytext=(0,7 if x in (0,4) else -12),textcoords='offset points',ha='center',fontsize=7.5,fontweight='bold')
 ax.axhline(66.61,color='#171717',ls='--',lw=.8,alpha=.6); ax.text(4.25,66.61,'Expert\navg.',fontsize=7,va='center',style='italic')
 ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.18)
axs[0].set_ylabel('Score (%)')
axs[0].legend(ncol=3,fontsize=7,loc='upper left',frameon=False,columnspacing=.8,handlelength=1.2)
fig.savefig(out/'incremental_append.pdf');fig.savefig(out/'incremental_append.png',dpi=180);plt.close(fig)

# Appendix: leave-one-out for MCSI (main text, Figure 3b) and ISCM, same colour scale.
# ISCM: full = ISCM; w/o Math = ISC (append trajectory step 3); w/o Code = ISM (mu3loo-ISM);
# w/o Science = ICM (mu3loo-ICM); w/o IF = SCM (mu3pre-SCM). Supplied by the author on 2026-11-19.
full_iscm=np.array([79.90,65.72,57.92,67.33]); avg_iscm=67.72
rem_iscm=np.array([[74.69,64.97,52.97,61.71],[79.90,63.77,58.02,68.17],[81.25,66.39,56.47,64.88],[81.09,66.02,56.12,54.46]])
avg_rem_iscm=np.array([63.58,67.47,67.25,64.42])
d_iscm=np.column_stack((rem_iscm-full_iscm,avg_rem_iscm-avg_iscm))
fig,axs=plt.subplots(1,2,figsize=(8.1,3.0),layout='constrained')
for ax,dd,title in [(axs[0],d,'(a) Removing one prompt from MCSI (main text, Fig. 3b)'),(axs[1],d_iscm,'(b) Removing one prompt from ISCM')]:
 im=ax.imshow(dd,cmap='RdBu',vmin=-13,vmax=13,aspect='auto')
 for i in range(4):
  for j in range(5):ax.text(j,i,f'{dd[i,j]:+.1f}',ha='center',va='center',fontsize=8.5,color='white' if abs(dd[i,j])>7 else '#171717')
 ax.set(title=title,xticks=range(5),xticklabels=['AIME24','LCB','GPQA-D','IFBench','Avg.'],yticks=range(4),yticklabels=['w/o Math','w/o Code','w/o Science','w/o IF'],xlabel='Evaluated benchmark')
 ax.title.set_fontsize(9);ax.tick_params(axis='x',labelsize=8);ax.axvline(3.5,color='white',lw=2)
fig.colorbar(im,ax=axs,orientation='vertical',fraction=.04,pad=.02,label='Score change after removal (pp)')
fig.savefig(out/'loo_compare.pdf');fig.savefig(out/'loo_compare.png',dpi=180);plt.close(fig)
