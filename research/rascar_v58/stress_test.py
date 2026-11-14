"""Conditional robustness of fixed OOF rankings, not a new validation split."""
from pathlib import Path
import numpy as np,pandas as pd,itertools,json
A=Path(__file__).resolve().parent;Q=pd.read_csv(A.parent/'rascar_v56/kaggle_payload/rascar_v56/inputs/q.csv');T=pd.read_csv(A.parent/'rascar_v56/kaggle_payload/rascar_v56/inputs/t.csv');n=len(Q);y=Q.error.to_numpy();S=np.load(A/'results/scores.npz');names=list(S.files)
cls=sorted(Q.class_name.unique());day=Q.day.to_numpy();c=Q.class_name.to_numpy();W=[];scenarios=[]
for mix in ['train','test']:
 for shifts in itertools.product([.5,1,2],repeat=4):
  for dayfactor in [.5,1,2]:
   w=np.ones(n)
   for k,f in zip(cls,shifts):
    m=c==k;pr=y[m].mean();target=pr*f/(1-pr+pr*f)
    mass=(T.class_name.eq(k).mean()/Q.class_name.eq(k).mean()) if mix=='test' else 1.
    w[m&(y==1)]=mass*target/pr;w[m&(y==0)]=mass*(1-target)/(1-pr)
   w[day=='2026-10-25']*=dayfactor
   W.append(w);scenarios.append(dict(mix=mix,day25_weight=dayfactor,**dict(zip(cls,shifts))))
W=np.array(W)
def weighted_ap(s,weights):
 order=np.argsort(-s,kind='mergesort');s=s[order];yw=y[order][None,:]*weights[:,order];total=weights[:,order]
 ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
 tp=yw.cumsum(1)[:,ends];den=total.cumsum(1)[:,ends]
 diff=np.diff(np.c_[np.zeros(len(weights)),tp],axis=1)
 return ((tp/np.maximum(den,1e-15))*diff).sum(1)/np.maximum(tp[:,-1],1e-15)
bas=weighted_ap(S['V52'][:n],W);v57=weighted_ap(S['V57'][:n],W);records=[]
for name in names:
 ap=weighted_ap(S[name][:n],W);d=ap-bas;e=ap-v57
 records.append(dict(name=name,n_scenarios=len(W),delta52_min=d.min(),delta52_mean=d.mean(),delta52_positive=(d>0).mean(),delta57_min=e.min(),delta57_mean=e.mean(),delta57_positive=(e>1e-12).mean()))
pd.DataFrame(records).sort_values('delta52_mean',ascending=False).to_csv(A/'results/prior_shift_stress.csv',index=False)
# Stratified cluster bootstrap: same frame gets one shared weight for every box.
rng=np.random.default_rng(5801);frames=Q.frame_id.unique();fi=pd.Index(frames).get_indexer(Q.frame_id);fd=[np.flatnonzero(np.isin(frames,Q.loc[Q.day.eq(d),'frame_id'].unique())) for d in sorted(Q.day.unique())]
weights=[]
for _ in range(2000):
 sampled=np.concatenate([rng.choice(v,len(v),replace=True) for v in fd]);weights.append(np.bincount(sampled,minlength=len(frames))[fi])
weights=np.asarray(weights);ba=weighted_ap(S['V52'][:n],weights);a57=weighted_ap(S['V57'][:n],weights);out=[]
for name in names:
 a=weighted_ap(S[name][:n],weights);d=a-ba;e=a-a57
 out.append(dict(name=name,delta52_mean=d.mean(),delta52_low=np.quantile(d,.025),delta52_high=np.quantile(d,.975),delta52_positive=(d>0).mean(),delta57_mean=e.mean(),delta57_low=np.quantile(e,.025),delta57_high=np.quantile(e,.975),delta57_positive=(e>0).mean()))
pd.DataFrame(out).sort_values('delta52_mean',ascending=False).to_csv(A/'results/frame_bootstrap.csv',index=False)
# Annotation-boundary sensitivity and robust leave-one-frame-out comparison.
quality=pd.read_csv(A.parent/'rascar_best_audit_2026-11-14/verification/v52_error_audit.csv').set_index('id').loc[Q.id,'max_gt_iou'].to_numpy()
slices=[('all',np.ones(n,bool)),('exclude_iou_055_065',(quality<.55)|(quality>.65)),('exclude_iou_050_070',(quality<.50)|(quality>.70))]
for d in sorted(Q.day.unique()):
 hrs=Q.frame_id.str[-6:-4].astype(int);m=Q.day.eq(d).to_numpy()
 for desc,use in [('morning',hrs<10),('later',hrs>=10)]:slices.append((d+'_'+desc,m&use.to_numpy()))
for f in frames:slices.append(('without_'+f,Q.frame_id.ne(f).to_numpy()))
rows=[]
for name in names:
 for desc,m in slices:
  if np.unique(y[m]).size<2:continue
  w=m.astype(float)[None,:];aa=weighted_ap(S[name][:n],w)[0];bb=weighted_ap(S['V52'][:n],w)[0];cc=weighted_ap(S['V57'][:n],w)[0]
  rows.append(dict(name=name,slice=desc,n=int(m.sum()),ap=aa,delta52=aa-bb,delta57=aa-cc))
pd.DataFrame(rows).to_csv(A/'results/slices.csv',index=False)
(A/'audit/stress_protocol.json').write_text(json.dumps(dict(prior_scenarios=len(W),bootstrap_samples=2000,seed=5801,limitation='Conditional fixed OOF predictions, not selection-adjusted or new independent holdout. Label-shift stress reweights existing examples and assumes class-conditional distributions fixed.'),indent=2))
print(pd.DataFrame(records).sort_values('delta52_mean',ascending=False).head(15).to_string(index=False));print(pd.DataFrame(out).query("name in ['V52','V57']").to_string(index=False))
