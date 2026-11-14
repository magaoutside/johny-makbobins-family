"""V58 adaptive development comparison. Only train OOF selects candidates; no visual-test annotations accessed."""
from pathlib import Path
import json,os
import pandas as pd,numpy as np
from sklearn.metrics import average_precision_score as AP
A=Path(__file__).resolve().parent;OLD=A.parent/'rascar_v56'; OUT=A/'results';OUT.mkdir(exist_ok=True)
SEED=os.environ.get('RASCAR_V58_SEED')
if SEED: OUT=OUT/('seed_'+SEED);OUT.mkdir(exist_ok=True)
COR=OLD/'correction_audit'
if SEED: COR=COR/('seed_'+str(5600+int(SEED[-1])))
Q=pd.read_csv(OLD/'kaggle_payload/rascar_v56/inputs/q.csv');T=pd.read_csv(OLD/'kaggle_payload/rascar_v56/inputs/t.csv')
y=Q.error.to_numpy();b=np.r_[Q.v52,T.v52]; rows=pd.concat([Q,T],ignore_index=True);cls=rows.class_name.to_numpy();n=len(Q)
z57=np.r_[pd.read_csv(COR/'selected_oof.csv').set_index('id').loc[Q.id,'score'],pd.read_csv(COR/'submission_v57_auto_correction.csv').set_index('id').loc[T.id,'score']]
S={};R=[]
def metric(z):
 s=z[:n];res={'ap':AP(y,s)}
 for c in ['day','class_name']:
  for k in Q[c].unique():
   m=Q[c].eq(k).to_numpy();res[k]=AP(y[m],s[m])
 bad=s[y==1,None]<=s[None,y==0];res['pairs']=int(bad.sum())
 return res
base=metric(b);v57=metric(z57)
def add(name,z,family):
 assert np.isfinite(z).all();z=np.clip(z,0,1);S[name]=z
 m=metric(z);m.update(name=name,family=family,delta52=m['ap']-base['ap'],delta57=m['ap']-v57['ap'],min_day_delta57=min(m[d]-v57[d] for d in Q.day.unique()),min_class_delta52=min(m[c]-base[c] for c in Q.class_name.unique()),max_test_delta52=float(abs(z[n:]-b[n:]).max()));R.append(m)
def pred(paths,col='p_error'):
 if SEED and len(paths)>1:
  selected=[p for p in paths if '_s'+str(5600+int(SEED[-1]))+'_' in p.name or '_s'+str(5800+int(SEED[-1]))+'_' in p.name]
  if selected:paths=selected
 d=pd.concat([pd.read_csv(p) for p in paths]).groupby(['id','split','fit_day'],as_index=False)[col].mean()
 return np.r_[d[d.split.eq('oof')].set_index('id').loc[Q.id,col],d[d.split.eq('test')].groupby('id')[col].mean().loc[T.id]]
add('V52',b,'baseline');add('V57',z57,'baseline')
# Revert or attenuate cup corrections: baseline cup ranking is already nearly perfect.
for w in [0,.5]:
 z=z57.copy();m=cls=='cup';z[m]=b[m]+w*(z57-b)[m];add(f'cup_keep{w}',z,'cup_guard')
# Reduce total correction as a sanity check for a broad, stable optimum.
for w in [.25,.5,.75]:add(f'shrink_all_{w}',b+w*(z57-b),'shrink')
# Independently vary hand and person contribution around the existing recipe.
for hw in [.5,.75,1.]:
 for pw in [.5,.75,1.]:
  z=z57.copy();z[cls=='cup']=b[cls=='cup'];
  for c,w in [('nude_hands',hw),('person',pw)]:z[cls==c]=b[cls==c]+w*(z57-b)[cls==c]
  add(f'guard_h{hw}_p{pw}',z,'stability')
# Monotonic projection of the correction onto baseline class ranks, using
# predictions only, independently on each target day. Never uses target labels.
from sklearn.isotonic import IsotonicRegression
for scope in ['cup','cup_hands','all']:
 for mode in ['isotonic','rank_transport']:
  z=z57.copy()
  classes=['cup'] if scope=='cup' else ['cup','nude_hands'] if scope=='cup_hands' else ['cup','nude_hands','person']
  for day in rows.day.unique():
   for c in classes:
    idx=np.flatnonzero((rows.day.to_numpy()==day)&(cls==c))
    if mode=='isotonic':
     z[idx]=IsotonicRegression(increasing=True).fit_transform(b[idx],z57[idx])+.0000001*b[idx]
    else:
     idx=idx[np.argsort(b[idx],kind='stable')];z[idx]=np.sort(z57[idx])
  add(f'project_{scope}_{mode}',z,'monotonic_projection')

# A second target: estimated probability GT IoU < .60. No new fits locally.
for fam in ['anchor','base13','base23']:
 p=pred([OLD/f'features/tabreg_{fam}_predictions.csv'],'cdf')
 for scope in ['person','nude_hands','all']:
  for w in [.05,.1]:
   z=z57.copy();m=(cls==scope) if scope!='all' else ~np.isin(cls,['cup','tray'])
   z[m]+=np.clip(w*(p-b),-.05,.05)[m];add(f'regcdf_{fam}_{scope}_{w}',z,'regcdf')
# Coarse sensitivity check: V57's person correction was at the edge of its
# original weight range. Keep hand/cup/tray unchanged; do not fit per-ID rules.
pf=pred(sorted((OLD/'gpu_recovered/rascar_v56/features').glob('gpu_sam2_full_anchor_s*_predictions.csv')))
pc=pred(sorted((OLD/'gpu_recovered/rascar_v56/features').glob('gpu_sam2_crop_anchor_s*_predictions.csv')))
for w in [.1,.15,.25,.3,.4]:
 for cap in [.05,.1]:
  z=z57.copy();m=cls=='person';z[m]=(b+np.clip(w*(pf-b),0,cap)+np.clip(w*(pc-b),-cap,0))[m]
  add(f'person_direction_w{w}_cap{cap}',z,'person_strength')
# Replace one side with a model seeing both segmentation feature views.
for family in ['both_anchor','both_meta','class_anchor','class_both']:
 paths=sorted((A/'features').glob(f'{family}_s*_predictions.csv'))
 if len(paths)!=3:continue
 p=pred(paths)
 for side in ['up','down','both']:
  z=z57.copy();m=cls=='person'
  f=p if side in ['up','both'] else pf;c=p if side in ['down','both'] else pc
  z[m]=(b+np.clip(.2*(f-b),0,.1)+np.clip(.2*(c-b),-.1,0))[m]
  add(f'directional_{family}_{side}',z,'directional_new_expert')

# Label-shift adaptation for new independently trained hand experts, with the
# SAME k=10 and 20% correction used by V57 (no new hyperparameter sweep).
def adapt(d):
 d=d.merge(rows[['id','day','class_name']],on='id',validate='many_to_one').copy();d['p_adapt']=np.nan
 for (fit,split,day,c),g in d.groupby(['fit_day','split','day','class_name']):
  tr=Q[Q.day.eq(fit)&Q.class_name.eq(c)];pr=(tr.error.sum()+1)/(len(tr)+2);target=pr
  p=np.clip(g.p_error.to_numpy(),.0001,.9999);odds=p/(1-p)
  for _ in range(100):
   a=odds*target/(1-target)*(1-pr)/pr;post=a/(1+a);nxt=(post.sum()+10*pr)/(len(g)+10)
   if abs(nxt-target)<1e-8:break
   target=nxt
  a=odds*target/(1-target)*(1-pr)/pr;d.loc[g.index,'p_adapt']=a/(1+a)
 return np.r_[d[d.split.eq('oof')].set_index('id').loc[Q.id,'p_adapt'],d[d.split.eq('test')].groupby('id').p_adapt.mean().loc[T.id]]
for family in ['both_anchor','both_meta','class_anchor','class_both']:
 paths=sorted((A/'features').glob(f'{family}_s*_predictions.csv'))
 if len(paths)!=3:continue
 if SEED:paths=[p for p in paths if '_s'+str(5800+int(SEED[-1]))+'_' in p.name]
 d=pd.concat([pd.read_csv(p) for p in paths]).groupby(['id','split','fit_day'],as_index=False).p_error.mean();p=adapt(d)
 z=z57.copy();m=cls=='nude_hands';z[m]=(b+np.clip(.2*(p-b),-.1,.1))[m]
 add(f'emhands_{family}',z,'new_em_hands')
 add(f'emhands_{family}_half',.5*(z+z57),'new_em_hands')

# Frame-delete jackknife: fit labels are perturbed, while fit-day GT prototype
# banks remain fixed. Thus this probes supervised-head sensitivity, not a new holdout.
from scipy.stats import trim_mean
jkpaths={fam:A/f'features/jk_{fam}_predictions.csv' for fam in ['anchor','full_anchor','crop_anchor']}
if all(p.exists() for p in jkpaths.values()):
 for adjust in [False,True]:
  for agg in ['mean','trimmed','median']:
   experts={};raws={}
   for fam,path in jkpaths.items():
    d=pd.read_csv(path)
    if adjust:
     p=np.clip(d.p_error.to_numpy(),.0001,.9999);od=p/(1-p)*d.full_prior/(1-d.full_prior)*(1-d.bag_prior)/d.bag_prior;d['p_error']=od/(1+od)
    fun=np.mean if agg=='mean' else np.median if agg=='median' else lambda x:trim_mean(x,.2)
    d=d.groupby(['id','split','fit_day'],as_index=False).p_error.agg(fun)
    raws[fam]=d
    experts[fam]=np.r_[d[d.split.eq('oof')].set_index('id').loc[Q.id,'p_error'],d[d.split.eq('test')].groupby('id').p_error.mean().loc[T.id]]
   z=z57.copy();m=np.isin(cls,['person','cup']);f,c=experts['full_anchor'],experts['crop_anchor']
   z[m]=(b+np.clip(.2*(f-b),0,.1)+np.clip(.2*(c-b),-.1,0))[m]
   add(f'jk_{agg}_prior{adjust}_sam',z,'jackknife')
   hand=adapt(raws['anchor']);h=cls=='nude_hands';z[h]=(b+np.clip(.2*(hand-b),-.1,.1))[h]
   add(f'jk_{agg}_prior{adjust}_all',z,'jackknife')
   add(f'jk_{agg}_prior{adjust}_half',.5*(z+z57),'jackknife')

# New GPU experts, two weights and a fixed class guard; test labels never used.
for family in ['both_anchor','both_meta','class_anchor','class_both']:
 paths=sorted((A/'features').glob(f'{family}_s*_predictions.csv'))
 if len(paths)!=3:continue
 p=pred(paths)
 for scope in ['person','nude_hands','all']:
  for w in [.1,.2]:
   z=z57.copy();m=(cls==scope) if scope!='all' else ~np.isin(cls,['cup','tray'])
   z[m]+=np.clip(w*(p-b),-.05,.05)[m];add(f'extra_{family}_{scope}_{w}',z,'new_expert')
   z=z57.copy();z[cls=='cup']=b[cls=='cup'];z[m]=b[m]+np.clip(w*(p-b),-.1,.1)[m];add(f'replace_{family}_{scope}_{w}',z,'new_expert_replace')
# New DINO heads replace the old dense contribution, rather than adding a
# second full score. Same cross-fit-day consensus policy as the V31 baseline.
def dense_scores(e):
 oo=e[e.split.eq('oof')].set_index('id').loc[Q.id,'overlap'].to_numpy()
 tt=np.stack([g.set_index('id').loc[T.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')])
 vqo=Q.v22.to_numpy();vqt=T.v22.to_numpy();target=np.clip((.85-oo)/.5,0,1);s=vqo.copy();valid=np.isfinite(oo);s[valid]+=.25*(target-vqo)[valid]
 target=np.clip((.85-tt)/.5,0,1);delta=target-vqt[None,:];same=((delta>=0).all(0)|(delta<=0).all(0))&np.isfinite(target).all(0)
 safe=np.where(np.isfinite(target),target,vqt[None,:]);conservative=np.where((delta>=0).all(0),safe.min(0),safe.max(0));z=vqt.copy();z[same]+=.25*(conservative-vqt)[same]
 return np.r_[s,z]
for backbone in ['vit_b','vit_l']:
 for kind in ['mlp','spatial']:
  paths=sorted((A/'features').glob(f'{backbone}_{kind}_{"s"+SEED if SEED else "mean3"}_fit*_evidence.csv'))
  if len(paths)!=2:continue
  e=pd.concat([pd.read_csv(p) for p in paths]);target=dense_scores(e);residual=(target-np.r_[Q.v31,T.v31])*np.where(cls=='person',1/1.2,1.)
  for scope in ['all','person','nude_hands','cup','tray']:
   for w in [.25,.5,1.]:
    z=z57.copy();m=np.ones(len(cls),bool) if scope=='all' else cls==scope;z[m]+=w*residual[m]
    add(f'dense_{backbone}_{kind}_{scope}_{w}',z,'nonlinear_dense')

for stage in ['frozen150','tuned350','tuned750']:
 paths=sorted((A/'features').glob(f'finetune_s3101_fit*_{stage}_evidence.csv'))
 if len(paths)!=2:continue
 target=dense_scores(pd.concat([pd.read_csv(p) for p in paths]));residual=(target-np.r_[Q.v31,T.v31])*np.where(cls=='person',1/1.2,1.)
 for scope in ['all','person','nude_hands','cup','tray']:
  for w in [.25,.5,1.]:
   z=z57.copy();m=np.ones(len(cls),bool) if scope=='all' else cls==scope;z[m]+=w*residual[m]
   add(f'finetune_{stage}_{scope}_{w}',z,'finetune_dense')

# GPU fine-tuning and actual RGB flip TTA, with the same old-dense replacement.
for family,pattern in [(f'ft_{bb}_{stage}',f'finetune_{bb}_{"s"+SEED if SEED else "mean3"}_fit*_{stage}_evidence.csv') for bb in ['vit_s','vit_b'] for stage in ['frozen150','tuned350','tuned750']]+[(f'tta_w{tw}',f'tta_spatial_w{tw}_{"s"+SEED if SEED else "mean3"}_fit*_evidence.csv') for tw in [.5,1.]]:
 paths=sorted((A/'features').glob(pattern))
 if len(paths)!=2:continue
 e=pd.concat([pd.read_csv(p) for p in paths]);target=dense_scores(e);residual=(target-np.r_[Q.v31,T.v31])*np.where(cls=='person',1/1.2,1.)
 for scope in ['all','person','nude_hands','cup','tray']:
  for w in [.5,1.]:
   z=z57.copy();m=np.ones(len(cls),bool) if scope=='all' else cls==scope;z[m]+=w*residual[m]
   add(f'{family}_{scope}_{w}',z,'fine_tta')
   if scope=='cup':
    m=cls=='tray';p=np.clip(b[m],.0001,.9999);z[m]=1/(1+np.exp(-(np.log(p/(1-p))-.6)))
    add(f'bounded_{family}_cup_{w}_tray0.6',b+np.clip(z-b,-.15,.15),'fine_tta_final')

# Combine complementary corrections on different classes. We use coarse,
# fixed weights and require day and seed confirmation before promotion.
for source,names in [('base_spatial',['dense_vit_b_spatial_cup_1.0']),('spatial_mean',['dense_vit_b_spatial_cup_1.0','dense_vit_l_spatial_cup_1.0']),('all_dense_mean',[f'dense_{bb}_{kk}_cup_1.0' for bb in ['vit_b','vit_l'] for kk in ['mlp','spatial']])]:
 if not all(k in S for k in names):continue
 cup_res=np.mean([S[k]-z57 for k in names],axis=0)
 for cw in [.5,1.]:
  for pw in [.2,.3,.4]:
   z=z57.copy()+cw*cup_res;m=cls=='person'
   z[m]=(b+np.clip(pw*(pf-b),0,.1)+np.clip(pw*(pc-b),-.1,0))[m]
   add(f'combo_{source}_c{cw}_p{pw}',z,'complementary_classes')

# Tray's baseline within-class ordering is perfect on OOF. Extend the original
# monotone calibration range, whose best V57 value was on the range boundary.
for source in ['V57','dense_vit_b_spatial_cup_1.0','combo_spatial_mean_c0.5_p0.2']:
 if source not in S:continue
 for shift in [.2,.6,.8,1.2]:
  z=S[source].copy();m=cls=='tray';p=np.clip(b[m],.0001,.9999);z[m]=1/(1+np.exp(-(np.log(p/(1-p))-shift)))
  add(f'traycal_{source}_m{shift}',z,'tray_calibration')

# Missing or contradictory new geometry is not evidence against the previous
# reliable dense component. Keep it in that case, for OOF and test alike.
paths=sorted((A/'features').glob(f'vit_b_spatial_{"s"+SEED if SEED else "mean3"}_fit*_evidence.csv'))
if len(paths)==2:
 e=pd.concat([pd.read_csv(p) for p in paths]);oo=e[e.split.eq('oof')].set_index('id').loc[Q.id,'overlap'].to_numpy();tt=np.stack([g.set_index('id').loc[T.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')]);target=np.clip((.85-tt)/.5,0,1);delta=target-T.v22.to_numpy()[None,:]
 support=np.r_[np.isfinite(oo),np.isfinite(tt).all(0)&((delta>=0).all(0)|(delta<=0).all(0))]
 for shift in [.4,.6]:
  source='dense_vit_b_spatial_cup_1.0' if shift==.4 else 'traycal_dense_vit_b_spatial_cup_1.0_m0.6'
  z=S[source].copy();m=(cls=='cup')&~support;z[m]=z57[m];z=b+np.clip(z-b,-.15,.15)
  add(f'valid_geometry_guard_tray{shift}',z,'validity_guard')

paths=sorted((A/'features').glob(f'tta_spatial_w0.5_{"s"+SEED if SEED else "mean3"}_fit*_evidence.csv'))
if len(paths)==2 and 'bounded_tta_w0.5_cup_1.0_tray0.6' in S:
 e=pd.concat([pd.read_csv(p) for p in paths]);oo=e[e.split.eq('oof')].set_index('id').loc[Q.id,'overlap'].to_numpy();tt=np.stack([g.set_index('id').loc[T.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')]);dd=np.clip((.85-tt)/.5,0,1)-T.v22.to_numpy()[None,:]
 support=np.r_[np.isfinite(oo),np.isfinite(tt).all(0)&((dd>=0).all(0)|(dd<=0).all(0))]
 z=S['bounded_tta_w0.5_cup_1.0_tray0.6'].copy();m=(cls=='cup')&~support;z[m]=z57[m]
 add('guarded_tta_cup_tray0.6_cap0.15',z,'final_guarded_tta')

# Bound the total change from the Public-confirmed V52, so several expert
# residuals cannot accumulate into a large unverified outlier correction.
for source in ['dense_vit_b_spatial_cup_1.0','traycal_dense_vit_b_spatial_cup_1.0_m0.2','traycal_dense_vit_b_spatial_cup_1.0_m0.6','traycal_dense_vit_b_spatial_cup_1.0_m0.8']:
 if source not in S:continue
 for cap in [.1,.15]:add(f'bounded_{source}_cap{cap}',b+np.clip(S[source]-b,-cap,cap),'bounded_final')

res=pd.DataFrame(R).sort_values('ap',ascending=False);res.to_csv(OUT/'candidates.csv',index=False);np.savez_compressed(OUT/'scores.npz',**S)
print(res[['name','ap','2026-10-25','2026-10-26','cup','nude_hands','person','tray','pairs','min_day_delta57','min_class_delta52']].head(20).to_string(index=False))
