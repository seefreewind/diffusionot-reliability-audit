"""Frozen aggregate rendering only. Units: five model seeds, three initial MC runs each.
Raw observations are displayed without uncertainty intervals or statistical testing.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
FIGURE_NUMBER = 3
D=Path(__file__).resolve().parent
OUT=D.parents[1]/'figures'; OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'figure.facecolor':'white','savefig.facecolor':'white'})
B='#216B91'; O='#BB682F'; G='#68737C'; LIGHT='#C2C8CC'; SEEDS=[11,23,47,71,101]
def panel(ax,s): ax.text(-.10 if FIGURE_NUMBER==1 else -.15,1.08,s,transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')
def threshold(ax):
 ax.axhline(5,color=G,lw=.9,ls=(0,(4,3)),zorder=1)
 ax.text(.03,5.55,'Predefined 5% admission threshold',transform=ax.get_yaxis_transform(),color=G,fontsize=7.8,va='bottom')
def box(ax,x,y,w,h,s,c=G):
 ax.add_patch(Rectangle((x-w/2,y-h/2),w,h,facecolor='white',edgecolor=c,lw=.8))
 ax.text(x,y,s,ha='center',va='center',fontsize=8.5,color=c,linespacing=1.4)
def arrow(ax,x,y0,y1,c=G): ax.annotate('',xy=(x,y1),xytext=(x,y0),arrowprops={'arrowstyle':'->','color':c,'lw':.8})
if FIGURE_NUMBER==1:
 fig,axs=plt.subplots(1,2,figsize=(6.5,4.25),gridspec_kw={'width_ratios':[.91,1.3]})
 fig.subplots_adjust(left=.055,right=.985,bottom=.07,top=.87,wspace=.19)
 for a in axs: a.axis('off'); a.set_xlim(0,1); a.set_ylim(0,1)
 for a,l in zip(axs,'AB'): panel(a,l)
 axs[0].set_title('Evidence hierarchy',fontweight='bold',pad=19)
 axs[1].set_title('Provenance of the two audit objects',fontweight='bold',pad=19)
 a=axs[0]
 for y,s in [( .9,'Identity provenance'),(.7,'Population/support validity'),(.5,'Numerical & refit stability')]: box(a,.5,y,.96,.12,s,B)
 arrow(a,.5,.835,.77); arrow(a,.5,.635,.57)
 a.text(.5,.345,'STOP',ha='center',va='center',fontsize=10,fontweight='bold',color=O)
 a.text(.5,.263,'Admission criteria unmet',ha='center',color=O,fontsize=8.3)
 box(a,.5,.125,.96,.12,'External lineage validation',G)
 a.text(.5,-.035,'Direct lineage outcomes\nnot evaluated',ha='center',color=G,fontsize=8.2)
 a=axs[1]; box(a,.5,.88,.73,.18,'Canonical LARRY cohort\n130,887 cells',B)
 # Dashed artifact branch shows a provenance question, not a mapped subset.
 a.plot([.34,.23],[.79,.70],color=O,lw=.8,ls='--'); a.plot([.66,.77],[.79,.70],color=B,lw=.8)
 box(a,.23,.56,.43,.22,'Published artifact\n43,968 rows',O)
 box(a,.77,.56,.43,.22,'ID-preserving\nreimplementation\n130,887 cells',B)
 arrow(a,.23,.44,.38,O); arrow(a,.77,.44,.38,B)
 a.text(.23,.30,'Deterministic mapping\nunavailable',ha='center',color=O,fontsize=8.1)
 a.text(.77,.30,'Canonical identities\nretained',ha='center',color=B,fontsize=8.1)
 a.text(.23,.115,'Planned clone-level\nvalidation unavailable',ha='center',color=O,fontsize=8.1)
 a.text(.77,.115,'Pre-lineage\nreliability audit',ha='center',color=B,fontsize=8.1)
 a.text(.5,-.035,'Row counts do not establish a subset relationship.',ha='center',fontsize=7.8,color=G)
elif FIGURE_NUMBER==2:
 df=pd.read_csv(D/'source_data.tsv',sep='\t')
 fig,axs=plt.subplots(1,2,figsize=(6.5,3.2),sharey=True)
 fig.subplots_adjust(left=.11,right=.98,bottom=.18,top=.75,wspace=.20)
 handles=[]
 for a,tr,c,title,passes,l in zip(axs,['2_to_4','4_to_6'],[B,O],['Early transition: day 2→4','Late transition: day 4→6'],['5/5 seeds pass','0/5 seeds pass'],'AB'):
  panel(a,l); a.set_title(title,pad=12,fontweight='bold')
  for dt,off,mark in [(.01,-.15,'o'),(.005,0,'s'),(.0025,.15,'^')]:
   part=df[(df.transition==tr)&np.isclose(df.dt,dt)].set_index('seed').loc[SEEDS]
   a.scatter(np.arange(5)+off,part.out_of_support_percent,s=28,color=c,marker=mark,zorder=3)
  threshold(a); a.set_ylim(0,23); a.set_xlim(-.5,4.5); a.set_xticks(range(5),SEEDS); a.set_xlabel('Paired seed'); a.set_yticks([0,5,10,15,20])
  a.text(.04,.97,passes,transform=a.transAxes,va='top',fontsize=8.5,fontweight='bold',color=c)
 axs[0].set_ylabel('Out-of-support endpoints (%)')
 handles=[Line2D([],[],color=G,marker=m,ls='',label=f'dt = {dt:g}',markersize=5) for dt,m in [(.01,'o'),(.005,'s'),(.0025,'^')]]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.995),ncol=3)
elif FIGURE_NUMBER==3:
 paired=pd.read_csv(D/'paired_support.tsv',sep='\t'); dt=pd.read_csv(D/'dt_support.tsv',sep='\t')
 fig,axs=plt.subplots(1,2,figsize=(6.5,3.15),sharey=True)
 fig.subplots_adjust(left=.11,right=.97,bottom=.18,top=.73,wspace=.27)
 for a,l in zip(axs,'AB'): panel(a,l); threshold(a); a.set_ylim(0,27); a.set_yticks([0,5,10,15,20])
 axs[0].set_title('Removal of diffusion (dt = 0.0025)',pad=13,fontweight='bold')
 for i,s in enumerate(SEEDS):
  part=paired[paired.seed==s].set_index('mode'); v=[part.loc['full SDE','out_of_support_percent'],part.loc['drift-only (d=0)','out_of_support_percent']]
  axs[0].plot([i-.15,i+.15],v,lw=1,color=LIGHT)
  axs[0].scatter(i-.15,v[0],color=B,s=28,zorder=3); axs[0].scatter(i+.15,v[1],color=O,s=28,zorder=3)
 axs[0].set_xticks(range(5),SEEDS); axs[0].set_xlim(-.5,4.5); axs[0].set_xlabel('Paired seed'); axs[0].set_ylabel('4→6 out-of-support endpoints (%)')
 fig.legend(handles=[Line2D([],[],ls='',marker='o',color=c,label=s,markersize=5) for c,s in [(B,'Full SDE'),(O,'Drift-only')]],loc='upper left',bbox_to_anchor=(.10,.99),ncol=2)
 axs[1].set_title('Integration step-size refinement',pad=13,fontweight='bold')
 for i,s in enumerate(SEEDS):
  p=dt[(dt.seed==s)&(dt.transition=='4_to_6')].set_index('dt').loc[[.01,.005,.0025]]
  axs[1].plot(range(3),p.out_of_support_percent,color=B,lw=.9,marker=['o','s','^','D','v'][i],ms=3.8)
  end=p.out_of_support_percent.iloc[-1]; label_y=end+({47:-.50,101:.50}.get(s,0))
  if label_y!=end:axs[1].plot([2.02,2.1],[end,label_y],color=LIGHT,lw=.6)
  axs[1].text(2.13,label_y,str(s),color=B,fontsize=8,va='center')
 axs[1].text(.04,.96,'Maximum change from baseline:\n0.06 percentage points',transform=axs[1].transAxes,fontsize=8,va='top',linespacing=1.4)
 axs[1].set_xticks(range(3),['0.01','0.005','0.0025']); axs[1].set_xlim(-.2,2.6); axs[1].set_xlabel('Integration step size (dt)')
elif FIGURE_NUMBER==4:
 df=pd.read_csv(D/'source_data.tsv',sep='\t'); variance=pd.read_csv(D/'variance.tsv',sep='\t')
 assert (variance.variance>0).all(), 'Log scale requires strictly positive stored variance components.'
 fig,axs=plt.subplots(2,2,figsize=(6.5,5.25))
 fig.subplots_adjust(left=.115,right=.97,bottom=.19,top=.88,wspace=.35,hspace=.82)
 for a,l in zip(axs.flat,'ABCD'): panel(a,l)
 a=axs[0,0]; a.set_title('Probability variance components',pad=13,fontweight='bold')
 a.bar([0,1],variance.variance,color=[B,O],width=.45); a.set_yscale('log'); a.set_ylim(1e-8,.5); a.set_yticks([1e-7,1e-5,1e-3,1e-1],['10⁻⁷','10⁻⁵','10⁻³','10⁻¹']); a.set_xticks([0,1],['Within-fit MC','Between-fit\npipeline']); a.set_ylabel('Probability variance')
 for x,y,t in zip([0,1],variance.variance,['2.48×10⁻⁷','0.0166']): a.text(x,y*1.65,t,ha='center',va='bottom',fontsize=8.2)
 a.text(.50,.93,'≈6.7×10⁴-fold',transform=a.transAxes,ha='center',fontsize=8.5)
 for a,part,title,med,ylabel in [(axs[0,1],'B','Velocity cosine vs seed 11','0.478','Cosine similarity'),(axs[1,0],'C','Growth-rank correlation vs seed 11','0.349','Spearman correlation')]:
  p=df[df.panel==part]; a.scatter(range(4),p.value,color=B,s=28,zorder=3); a.set_ylim(0,1.06); a.set_xticks(range(4),p.seed.astype(int)); a.set_xlabel('Seed compared with seed 11'); a.set_ylabel(ylabel); a.set_title(title,pad=13,fontweight='bold')
  a.text(.04,.98,f'Median vs reference = {med}',transform=a.transAxes,va='top',fontsize=8)
 a=axs[1,1]; a.set_title('Initial sampling-run convergence',pad=13,fontweight='bold')
 for r in df[df.panel=='D'].itertuples():
  ok=r.value==1; a.scatter(int(r.replicate)-1,SEEDS.index(int(r.seed)),s=72,marker='o',facecolors=B if ok else 'white',edgecolors=B if ok else O,linewidths=1.0)
 a.set_xlim(-.5,2.5); a.set_ylim(4.6,-.6); a.set_xticks([0,1,2],['1','2','3']); a.set_yticks(range(5),SEEDS); a.set_xlabel('Initial 10,000-sample run'); a.set_ylabel('Paired seed')
 for spine in a.spines.values(): spine.set_visible(False)
 a.tick_params(length=0)
 fig.text(.69,.075,'11/15 initial 10,000-sample runs converged',ha='center',fontsize=8.2)
 fig.legend(handles=[Line2D([],[],ls='',marker='o',color=B,label='Converged',markersize=6),Line2D([],[],ls='',marker='o',markerfacecolor='white',color=O,label='Did not meet frozen convergence criterion',markersize=6)],loc='lower center',bbox_to_anchor=(.54,.008),ncol=2,fontsize=7.8)
else: raise ValueError('Select a figure number')
paths=[]
for suffix in ['.pdf','.png']:
 p=OUT/f'Figure{FIGURE_NUMBER}_V5{suffix}'; fig.savefig(p,dpi=300); paths.append(p)
# Editable SVG is retained as an internal vector QA artifact.
svg=OUT/f'Figure{FIGURE_NUMBER}_V5.svg'
fig.savefig(svg)
plt.close(fig)
(D/'figure_hash.txt').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in paths))
print(f'Figure {FIGURE_NUMBER}: PDF + 300 dpi PNG')
