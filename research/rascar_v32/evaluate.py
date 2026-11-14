"""Evaluate preserved cross-day predictions; test audit is descriptive, not truth."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,CL,DAYS,base_oof,current,metric,iou,align
import numpy as np,pandas as pd

def collect(df,rows,kind,cut):
 out=[]
 for _,r in rows.iterrows():
  ref=df[df.record_id.eq(r.id)&df.class_name.eq(r.class_name)]
  ref=ref[ref.threshold.eq(cut)] if kind.startswith(('watershed','conditional')) else ref[ref.confidence.ge(cut)]
  val=iou(r[B],ref[B]).ravel()
  if len(val):
   j=val.argmax();best=ref.iloc[j]
   out.append(dict(id=r.id,overlap=float(val[j]),confidence=float(best.confidence),references=len(ref),**{f'ref_{b}':best[b] for b in B}))
  else:out.append(dict(id=r.id,overlap=np.nan,confidence=0,references=0,**{f'ref_{b}':np.nan for b in B}))
 return pd.DataFrame(out)

def main():
 rec=pd.read_csv(A/'inputs/dense_records.csv');audit=align(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv',te).reset_index()
 entries=[];controls=[]
 kind=sys.argv[1];configs=[(kind,[.35,.5,.65] if kind.startswith(('watershed','conditional')) else [.15,.3,.5])]
 for kind,cuts in configs:
  boxes={day:pd.read_csv(A/f'features/{kind}_fit{day[-2:]}_boxes.csv') for day in DAYS}
  for cut in cuts:
   label=f'{kind}_cut{cut}';oo=pd.DataFrame({'id':q.id});tt=[]
   for fit in DAYS:
    sub=q[q.day.ne(fit)];f=collect(boxes[fit],sub,kind,cut);oo=oo.merge(f,how='left',on='id',suffixes=('','_next')) if len(oo.columns)==1 else oo.set_index('id').combine_first(f.set_index('id')).reset_index()
    t=collect(boxes[fit],te,kind,cut);tt.append(t.set_index('id').reindex(te.id).reset_index())
    man=rec[rec.split.eq('manual')&rec.day.ne(fit)&rec.class_name.isin(CL)]
    m=collect(boxes[fit],man,kind,cut).merge(man[['id','origin_id','class_name','frame_id']],on='id')
    m.to_csv(A/f'audit/{label}_manual_held{("26" if fit.endswith("25") else "25")}.csv',index=False)
    for cls in CL:
     z=m[m.class_name.eq(cls)].groupby('origin_id').overlap.mean()
     controls.append(dict(experiment=label,fit=fit,class_name=cls,n=len(z),coverage=float(z.notna().mean()),mean_iou=float(z.fillna(0).mean()),recall_iou06=float(z.ge(.6).mean())))
   oo=oo.set_index('id').reindex(q.id).reset_index();oo.to_csv(A/f'features/{label}_oof_evidence.csv',index=False)
   for fit,t in zip(DAYS,tt):t.to_csv(A/f'features/{label}_test_fit{fit[-2:]}_evidence.csv',index=False)
   ov=oo.overlap.to_numpy();ot=np.stack([x.overlap.to_numpy() for x in tt]);otarget=np.clip((.85-ov)/.5,0,1);ttarget=np.clip((.85-ot)/.5,0,1)
   # Both fitting days must agree on the direction. Missing detection abstains.
   tdelta=ttarget-current[None,:];same=(np.all(tdelta>=0,axis=0)|np.all(tdelta<=0,axis=0))&np.isfinite(ttarget).all(0)
   safe=np.where(np.isfinite(ttarget),ttarget,current[None,:])
   conservative=np.where(np.all(tdelta>=0,axis=0),np.min(safe,axis=0),np.max(safe,axis=0))
   for family in CL+['all']:
    om=np.isfinite(ov)&(q.class_name.eq(family).to_numpy() if family!='all' else True)
    tm=same&(te.class_name.eq(family).to_numpy() if family!='all' else True)
    for direction in ['two_sided','penalty_only']:
     for weight in [.1,.25]:
      name=f'{label}_{family}_{direction}_w{weight}';s=base_oof.copy();t=current.copy()
      delta=otarget-base_oof;dt=conservative-current
      if direction=='penalty_only':delta=np.maximum(delta,0);dt=np.maximum(dt,0)
      s[om]+=weight*delta[om];t[tm]+=weight*dt[tm]
      q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
      knownbad=audit.visual_status_v29.isin(['clear_error','likely_error'])|audit.geometry_status.eq('robust_error_under_assumptions');good=audit.visual_status_v29.eq('plausible_correct')
      changes=t-current
      entries.append(dict(experiment=name,head=kind,cut=cut,family=family,direction=direction,weight=weight,**metric(q,s),changed_oof=int((abs(s-base_oof)>1e-9).sum()),changed_test=int((abs(changes)>1e-9).sum()),known_error_score_down=int((knownbad&(changes<-.02)).sum()),plausible_correct_score_up=int((good&(changes>.02)).sum()),known_error_score_up=int((knownbad&(changes>.02)).sum()),plausible_correct_score_down=int((good&(changes<-.02)).sum())))
 suffix=sys.argv[1]
 summary=pd.DataFrame(entries).sort_values('pooled_ap',ascending=False);summary.to_csv(A/f'results/{suffix}_experiments.csv',index=False);pd.DataFrame(controls).to_csv(A/f'results/{suffix}_manual_localization.csv',index=False)
 print(summary.head(20).to_string(index=False));print('\nCONTROLS\n',pd.DataFrame(controls).to_string(index=False))
if __name__=='__main__':main()
