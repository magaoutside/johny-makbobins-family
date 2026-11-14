"""Restrict V31 corrections by independently trained head agreement.
The two heads share DINO features; they are NOT statistically independent.
Coordinate perturbations are sensitivity tests, not confidence intervals.
"""
from pathlib import Path
import sys,itertools,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';V31=ROOT/'rascar_v31'
sys.path.insert(0,str(ROOT/'rascar_v25'));sys.path.insert(0,str(ROOT/'rascar_v30'))
from common25 import q,te,h,B,CL,DAYS,base_oof,current,metric,iou,align,guard
from bounded_geometry import upper_iou
import numpy as np,pandas as pd

def evidence(rows,split,fit):
 tail='oof' if split=='oof' else f'test_fit{fit[-2:]}'
 mask=align(V31/f'features/linear_mask_cut0.5_{tail}_evidence.csv',rows)
 det=align(V31/f'features/conv_detector_cut0.3_{tail}_evidence.csv',rows)
 low=align(V31/f'features/linear_mask_cut0.35_{tail}_evidence.csv',rows)
 high=align(V31/f'features/linear_mask_cut0.65_{tail}_evidence.csv',rows)
 out=[]
 for j,(_,r) in enumerate(rows.iterrows()):
  b=r[B].to_numpy(float);mr=mask.iloc[j][['ref_'+c for c in B]].to_numpy(float);dr=det.iloc[j][['ref_'+c for c in B]].to_numpy(float)
  f=dict(id=r.id,mask_iou=mask.iloc[j].overlap,det_iou=det.iloc[j].overlap,mutual_iou=float(iou(mr,dr)[0,0]),mask_conf=mask.iloc[j].confidence,det_conf=det.iloc[j].confidence,upper=1.,lower=0.,threshold_stability=0.)
  refs=np.stack([mr,dr,low.iloc[j][['ref_'+c for c in B]],high.iloc[j][['ref_'+c for c in B]]]).astype(float)
  if np.isfinite(refs).all():
   u=max(2.,.04*min(b[2:]-b[:2])) if r.class_name=='person' else 2.
   pert=[]
   for ref in refs:
    bs=np.array([ref+np.array(delta) for delta in itertools.product([-u,u],repeat=4)]);bs=bs[(bs[:,2:]>bs[:,:2]).all(1)]
    pert.extend(iou(b,bs).ravel())
   f['upper']=max(upper_iou(b,ref,u) for ref in refs);f['lower']=float(min(pert));f['threshold_stability']=float(iou(refs[2],refs[3])[0,0])
  out.append(f)
 return pd.DataFrame(out)

def save(name,s,t):
 assert np.isfinite(s).all() and np.isfinite(t).all() and min(s.min(),t.min())>=0 and max(s.max(),t.max())<=1
 q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
 au=align(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv',te).reset_index();au['new_score']=t;au['delta']=t-current
 au.to_csv(A/f'results/reviews/{name}.csv',index=False)
 bad=au.visual_status_v29.isin(['clear_error','likely_error'])|au.geometry_status.eq('robust_error_under_assumptions');good=au.visual_status_v29.eq('plausible_correct')
 return dict(experiment=name,**metric(q,s),changed_oof=int((abs(s-base_oof)>1e-9).sum()),changed_test=int((abs(t-current)>1e-9).sum()),known_error_score_down=int((bad&(au.delta<-.02)).sum()),plausible_correct_score_up=int((good&(au.delta>.02)).sum()),known_error_score_up=int((bad&(au.delta>.02)).sum()))

def main():
 ev={}
 for split,rows in [('oof',q),('test',te)]:
  fits=['oof'] if split=='oof' else DAYS
  ev[split]=[evidence(rows,split,fit) for fit in fits]
  for fit,e in zip(fits,ev[split]):e.to_csv(A/f'features/consensus_{split}_{fit}.csv',index=False)
 results=[]
 for policy in ['detector_veto','mutual_065','stable_bounds']:
  for family in CL+['all']:
   for direction in ['two_sided','penalty_only']:
    for weight in [.25,.5]:
     outputs=[]
     for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
      g=np.zeros(len(rows),bool)
      for cls in CL:
       ix=rows.class_name.eq(cls).to_numpy();g[ix]=guard(split,rows.loc[ix],cls)
      penalties=[];rescues=[];targets=[]
      for e in ev[split]:
       m=e.mask_iou.to_numpy();d=e.det_iou.to_numpy();agree=e.mutual_iou.ge(.5).to_numpy()
       pen=(m<.6)&(d<.6)&(m>.1)&(d>.1)&agree
       rescue=(m>.65)&(d>.65)&agree&g
       if policy=='mutual_065':pen&=e.mutual_iou.ge(.65).to_numpy();rescue&=e.mutual_iou.ge(.65).to_numpy()
       if policy=='stable_bounds':pen&=e.upper.lt(.6).to_numpy()&e.threshold_stability.ge(.7).to_numpy();rescue&=e.lower.gt(.65).to_numpy()
       penalties.append(pen);rescues.append(rescue);targets.append(np.clip((.85-m)/.5,0,1))
      pen=np.all(penalties,axis=0);rescue=np.all(rescues,axis=0);ta=np.stack(targets);safe=np.where(np.isfinite(ta),ta,base[None,:]);pen_target=safe.min(0);rescue_target=safe.max(0)
      if family!='all':pen&=rows.class_name.eq(family).to_numpy();rescue&=rows.class_name.eq(family).to_numpy()
      s=base.copy();s[pen]+=weight*np.maximum(pen_target[pen]-s[pen],0)
      if direction=='two_sided':s[rescue]-=weight*np.maximum(s[rescue]-rescue_target[rescue],0)
      outputs.append(s)
     results.append(save(f'consensus_{policy}_{family}_{direction}_w{weight}',*outputs))
 d=pd.DataFrame(results).sort_values('pooled_ap',ascending=False);d.to_csv(A/'results/consensus_experiments.csv',index=False);print(d.head(20).to_string(index=False))
if __name__=='__main__':main()
