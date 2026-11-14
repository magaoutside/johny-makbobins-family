"""Causal-direction ablation; tune only on opposite day, reviewed pool diagnostic."""
from common25 import *
from features23 import frame_objects
import itertools
pool=pd.concat([pd.read_csv(ROOT/f'rascar_v20/inputs/approved_pseudo_boxes_{d}.csv') for d in DAYS])
pool=pool[pool.class_name.eq('cup')&pool.visual_decision.eq('accept')&pool.label_use.eq('geometry_and_presence')].copy()
mp=pd.read_csv(ROOT/'rascar_v20/inputs/machine_predictions.csv')

def features(rows,pool_mode=False):
 out=[]
 for r in rows.itertuples():
  b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];hh=mp[mp.frame_id.eq(r.frame_id)&mp.class_name.eq('nude_hands')&mp.confidence.ge(.5)] if pool_mode else frame_objects(r.frame_id)['nude_hands']
  hb=hh[B].to_numpy();gap=np.maximum(0,np.maximum(hb[:,:2]-b[2:],b[:2]-hb[:,2:]))
  dist=np.linalg.norm(gap,axis=1)/max(wh.max(),1)
  conf=hh.confidence.to_numpy();j=int(dist.argmin()) if len(hh) else -1
  out.append(dict(id=r.id,nearest_hand_gap=dist[j] if j>=0 else 99,hand_conf=conf[j] if j>=0 else 0,cup_conf=r.confidence))
 return pd.DataFrame(out).set_index('id')
fo=features(q);ft=features(te);fp=features(pool,True)
for name,f in [('oof',fo),('test',ft),('pool',fp)]:f.to_csv(A/f'features/cup_hand_{name}.csv')
pool.to_csv(A/'inputs/reviewed_pool_cups.csv',index=False)

def apply(rows,base,f,cut,gap,w,direction,safe):
 gate=rows.class_name.eq('cup').to_numpy()&(f.cup_conf.to_numpy()>=cut)&(f.nearest_hand_gap.to_numpy()<=gap)&(f.hand_conf.to_numpy()>=.5)
 if safe:gate &= guard('oof' if len(rows)==len(q) else 'test',rows,'cup')
 return np.clip(base+w*gate*((1-base) if direction=='raise_error' else -base),0,1)
params=list(itertools.product([.5,.7,.85,.95],[.25,.75,1.5],[.15,.35],['raise_error','lower_error'],[False,True]))
outputs=[]
for cut,gap,w,direction,safe in params:
 name=f'cup_{direction}_c{cut}_d{gap}_w{w}_g{int(safe)}';s=apply(q,base_oof,fo,cut,gap,w,direction,safe);t=apply(te,current,ft,cut,gap,w,direction,safe)
 p_gate=(fp.cup_conf>=cut)&(fp.nearest_hand_gap<=gap)&fp.hand_conf.ge(.5)
 r=register(name,s,t,dict(type='cup_hand_nearness',cut=cut,gap=gap,weight=w,direction=direction,semantic_duplicate_guard=safe,pool_positive_count=len(pool),pool_positive_activated=int(p_gate.sum()),pool_has_no_negatives=True,selection='grid exploratory; separate opposite-day selection below'))
 outputs.append((name,s,t,r))
# Include unchanged baseline in threshold selection, preserve ties in favor of no change.
choices=[('unchanged',base_oof,current,None)]+outputs
so=base_oof.copy();tall=[];selection=[]
for fit,held in zip(DAYS,DAYS[::-1]):
 m=q.day.eq(fit).to_numpy();hi=q.day.eq(held).to_numpy()
 best=max(choices,key=lambda a:metric(q[m],a[1][m])['pooled_ap'])
 so[hi]=best[1][hi];tall.append(best[2]);selection.append(dict(fit=fit,held=held,chosen=best[0],fit_ap=metric(q[m],best[1][m])['pooled_ap']))
register('cup_nested_day_selected',so,np.mean(tall,axis=0),dict(type='opposite_day_parameter_selection',selection=selection,warning='Only two days; fit-day OOF baseline came from the other base fold; no pristine outer model retraining'))
print('SELECTION',selection,flush=True)
