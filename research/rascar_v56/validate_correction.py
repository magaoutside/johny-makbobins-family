"""Fixed-prediction robustness, paired frame bootstrap, and complete change audit."""
from pathlib import Path
import json, hashlib, zipfile
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
A=Path(__file__).resolve().parent;O=A/'correction_audit'
sel=json.loads((O/'selection.json').read_text());name=sel['winner']
q=pd.read_csv(O/'selected_oof.csv');y=q.error.to_numpy();base=q.v52.to_numpy();s=q.score.to_numpy()
z=pd.read_csv(O/'all_test_changes.csv');flag=z[z.flag.isin(['up','down'])].copy()

def pair_matrix(score):
    return (score[y==1,None]<=score[None,y==0])
old,new=pair_matrix(base),pair_matrix(s)
err=q[y==1].id.to_numpy();good=q[y==0].id.to_numpy()
pair_rows=[]
for i,j in zip(*np.where(old|new)):
    pair_rows.append({'error_id':err[i],'correct_id':good[j],'old_wrong':bool(old[i,j]),'new_wrong':bool(new[i,j]),'outcome':'fixed' if old[i,j] and not new[i,j] else 'new' if new[i,j] and not old[i,j] else 'remaining'})
pd.DataFrame(pair_rows).to_csv(O/'train_pair_changes.csv',index=False)
old_n=np.zeros(len(q),int);new_n=old_n.copy()
old_n[y==1]=old.sum(1);old_n[y==0]=old.sum(0)
new_n[y==1]=new.sum(1);new_n[y==0]=new.sum(0)
q['wrong_pairs_before']=old_n;q['wrong_pairs_after']=new_n
q['pair_change']=new_n-old_n
q.to_csv(O/'train_all_changes.csv',index=False)
q[old_n>0].to_csv(O/'train_original_49_changes.csv',index=False)

# Conditional resampling of fixed OOF predictions: this does NOT correct the
# uncertainty from choosing a method on these same validation data.
rng=np.random.default_rng(5701); frames=q.frame_id.unique();fi=pd.Index(frames).get_indexer(q.frame_id)
day_frames=[np.flatnonzero(np.isin(frames,q.loc[q.day.eq(day),'frame_id'].unique())) for day in sorted(q.day.unique())]
delta=[]
for _ in range(2000):
    sampled=np.concatenate([rng.choice(ix,len(ix),replace=True) for ix in day_frames]);w=np.bincount(sampled,minlength=len(frames))[fi]
    if np.unique(y[w>0]).size<2: continue
    delta.append(AP(y,s,sample_weight=w)-AP(y,base,sample_weight=w))
loo=[]
for f in frames:
    m=q.frame_id.ne(f).to_numpy();loo.append({'excluded_frame':f,'ap':float(AP(y[m],s[m])),'delta':float(AP(y[m],s[m])-AP(y[m],base[m]))})
pd.DataFrame(loo).to_csv(O/'leave_one_frame_out.csv',index=False)
counts={}
for ref in ['v52','a40']:
    d=flag[f'directional_change_vs_{ref}']
    counts[ref]={str(threshold):{'favorable':int((d>=threshold).sum()),'unfavorable':int((d<=-threshold).sum()),'small_or_unchanged':int((abs(d)<threshold).sum()),'favorable_frames':int(flag.loc[d>=threshold,'frame_id'].nunique())} for threshold in [.01,.025,.05]}
summary={'winner':name,'total_test':len(z),'suspected_test_boxes':len(flag),'suspected_test_frames':int(flag.frame_id.nunique()),'changed_test_over_0.01':int((abs(z.delta_vs_v52)>=.01).sum()),'visual_hypotheses_only':True,'visual_counts':counts,'train_pairs':{'before':int(old.sum()),'after':int(new.sum()),'fixed':int((old&~new).sum()),'new':int((~old&new).sum()),'remaining':int((old&new).sum())},'original_49_train':{'improved':int(((old_n>0)&(new_n<old_n)).sum()),'worsened':int(((old_n>0)&(new_n>old_n)).sum()),'unchanged':int(((old_n>0)&(new_n==old_n)).sum()),'fully_resolved':int(((old_n>0)&(new_n==0)).sum())},'bootstrap':{'n':len(delta),'unit':'frame, stratified by day','seed':5701,'mean_delta':float(np.mean(delta)),'p_delta_positive':float(np.mean(np.array(delta)>0)),'delta_95pct':np.quantile(delta,[.025,.975]).tolist(),'limitation':'Conditional fixed-score bootstrap; does not adjust for model/hyperparameter selection.'},'leave_one_frame_out':{'n':len(loo),'positive_delta':int(sum(v['delta']>0 for v in loo)),'min_delta':min(v['delta'] for v in loo),'max_delta':max(v['delta'] for v in loo)},'submission_sha256':hashlib.sha256((O/'submission_v57_auto_correction.csv').read_bytes()).hexdigest()}
(O/'robustness.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
