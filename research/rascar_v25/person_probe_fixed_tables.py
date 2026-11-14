"""Refined single-class LB probe using stable table geometry, reviewed visually."""
from common25 import *
from tray_geometry import bank
from features23 import heads

def features(rows,fit):
 anchors,_=bank(fit);out=[]
 for r in rows.itertuples():
  b=np.array([getattr(r,c) for c in B]);f=dict(id=r.id,gate=0,strength=0.,table=-1,extension_iou=1.,**dict(zip(['ref_'+v for v in B],b)))
  if r.class_name!='person':out.append(f);continue
  hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();cc=(hh[:,:2]+hh[:,2:])/2;inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
  if inside.sum()!=1:out.append(f);continue
  head=cc[inside][0];tw=anchors[:,2:]-anchors[:,:2];tc=(anchors[:,:2]+anchors[:,2:])/2;rel=(head-tc)/tw
  j=np.linalg.norm(rel,axis=1).argmin();tb=anchors[j];wh=tw[j];rr=rel[j]
  # Stable far-side tabletop anchor. Abstain when box already covers most of table.
  gate=abs(rr[0])<.65 and -1.4<rr[1]<-.2 and b[3]<tb[1]+.75*wh[1]
  bb=b.copy();bb[3]=max(bb[3],tb[3]+.30*wh[1]);ov=float(iou(b,bb)[0,0]);strength=float(np.clip((.8-ov)/.3,0,1))*gate
  f.update(gate=int(gate),strength=strength,table=int(j),extension_iou=ov,**dict(zip(['ref_'+v for v in B],bb)));out.append(f)
 return pd.DataFrame(out).set_index('id')
fo=pd.concat([features(q[q.day.eq(held)],fit) for fit,held in zip(DAYS,DAYS[::-1])]).loc[q.id]
tf=[features(te,fit) for fit in DAYS];ft=(tf[0]+tf[1])/2
# Require both fit-day table maps to agree about eligibility.
ft.loc[tf[0].gate.ne(1)|tf[1].gate.ne(1),'strength']=0
for split,f in [('oof',fo),('test',ft)]:f.to_csv(A/f'features/person_fixed_tables_{split}.csv')
s=base_oof+.9*(1-base_oof)*fo.strength.to_numpy();t=current+.9*(1-current)*ft.strength.to_numpy()
register('person_hidden_legs_fixed_tables',s,t,dict(type='single_class_LB_probe',threshold_fit=False,anchor='6 robust table medians, opposite day; two-map agreement on test',gate='one head, far seat, bbox bottom ends before 75% of tabletop depth',extension='table bottom + .30 table height',weight=.9,reason_for_refinement='Initial live table detections sometimes included chair/legs and penalized already complete people; fixed-map proposal checked on images'))
(A/'submission_v25_person_hidden_legs_test.csv').write_bytes((A/'results/submissions/submission_person_hidden_legs_fixed_tables.csv').read_bytes())
assert np.array_equal(t[te.class_name.ne('person')],current[te.class_name.ne('person')])
import render23
render23.A=A
d=pd.read_csv(A/'results/reviews/person_hidden_legs_fixed_tables.csv');print(render23.render(d[d.delta.abs()>.02],'person_final',ref=ft))
d=q.assign(baseline_v22=base_oof,new_score=s);print(render23.render(d[d.error.eq(0)&d.new_score.gt(d.baseline_v22+.02)],'person_final_gt',gt=True,ref=fo))
