from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';sys.path.insert(0,str(A));sys.path.insert(0,str(ROOT/'rascar_v25'))
from common25 import q,te,h,B,base_oof,current,guard,iou
from consensus import save
import numpy as np,pandas as pd
TAG='handseg_wide' if '--wide' in sys.argv else 'handseg'
CONTEXTS=[8.,12.] if '--wide' in sys.argv else [2.4,4.]

def extract(rows,cut):
 raw=pd.read_csv(A/f'features/{TAG}_boxes.csv');sm=pd.read_csv(A/f'features/{TAG}_summary.csv');out=[]
 for _,r in rows.iterrows():
  d=dict(id=r.id);refs=[]
  for context,key in zip(CONTEXTS,['near','far']):
   z=raw[raw.id.eq(r.id)&raw.context.eq(context)&raw.threshold.eq(cut)];p=sm[sm.id.eq(r.id)&sm.context.eq(context)].iloc[0]
   ov=iou(r[B],z[B]).ravel();j=ov.argmax() if len(ov) else -1;b=z.iloc[j] if j>=0 else None
   d.update({key+'_iou':float(ov[j]) if j>=0 else np.nan,key+'_confidence':float(b.confidence) if b is not None else 0,key+'_touch':bool(b.touch_crop) if b is not None else True,key+'_p95':p.prob_p95,key+'_mean':p.prob_mean})
   refs.append(b[B].to_numpy(float) if b is not None else np.full(4,np.nan))
   d.update({key+'_'+x:float(b[x]) if b is not None else np.nan for x in B})
  d['mutual_iou']=float(iou(*refs)[0,0]);out.append(d)
 return pd.DataFrame(out)

def main():
 manual=h[h.class_name.eq('nude_hands')].copy();manual['id']=['manual_gt_'+str(i) for i in manual.index];pool=pd.read_csv(ROOT/'rascar_v22/inputs/reviewed_pool_hands.csv');allres=[];controls=[]
 for cut in [.35,.5,.65]:
  ev={}
  for split,rows in [('oof',q[q.class_name.eq('nude_hands')]),('test',te[te.class_name.eq('nude_hands')]),('manual',manual),('pool',pool)]:
   e=extract(rows,cut);e.to_csv(A/f'features/{TAG}_cut{cut}_{split}_evidence.csv',index=False);ev[split]=e
   if split in ['manual','pool']:
    for key in ['near','far']:
     val=e[key+'_iou'];controls.append(dict(source=split,cut=cut,context=key,n=len(e),coverage=float(val.notna().mean()),iou06_fraction=float(val.ge(.6).mean()),mean_iou=float(val.fillna(0).mean()),absence_false_flags=int((e.near_p95.lt(.1)&e.far_p95.lt(.1)).sum())))
  for policy in ['raw','stable_geometry','absence_veto']:
   for direction in ['two_sided','penalty_only']:
    for weight in [.1,.25]:
     outs=[]
     for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
      ix=rows.class_name.eq('nude_hands').to_numpy();e=ev[split];b=base[ix];v=e[['near_iou','far_iou']].to_numpy().T;targets=np.clip((.85-v)/.5,0,1);valid=np.isfinite(v).all(0);p=(targets>=b).all(0)&valid;r=(targets<=b).all(0)&valid
      tg=np.where(np.isfinite(targets),targets,b[None,:]);up=tg.min(0);dn=tg.max(0)
      if policy=='stable_geometry':
       agree=e.mutual_iou.ge(.65).to_numpy()&~e.near_touch.to_numpy()&~e.far_touch.to_numpy()&e.near_confidence.ge(.75).to_numpy()&e.far_confidence.ge(.75).to_numpy()
       p&=agree&(v<.55).all(0);r&=agree&(v>.65).all(0)&guard(split,rows.loc[ix],'nude_hands')
      if policy=='absence_veto':p=e.near_p95.lt(.1).to_numpy()&e.far_p95.lt(.1).to_numpy();r[:]=False;up[:]=.9
      s=base.copy();z=b.copy();z[p]+=weight*np.maximum(up[p]-b[p],0)
      if direction=='two_sided':z[r]-=weight*np.maximum(b[r]-dn[r],0)
      s[ix]=z;outs.append(s)
     allres.append(save(f'{TAG}_cut{cut}_{policy}_{direction}_w{weight}',*outs))
 pd.DataFrame(allres).sort_values('pooled_ap',ascending=False).to_csv(A/f'results/{TAG}_experiments.csv',index=False);pd.DataFrame(controls).to_csv(A/f'results/{TAG}_controls.csv',index=False)
 print(pd.DataFrame(allres).sort_values('pooled_ap',ascending=False).head(15).to_string(index=False));print(pd.DataFrame(controls).to_string(index=False))
if __name__=='__main__':main()
