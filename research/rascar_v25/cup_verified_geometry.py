from common25 import *
from features23 import frame_objects
mp=pd.read_csv(ROOT/'rascar_v20/inputs/machine_predictions.csv');pool=pd.read_csv(A/'inputs/reviewed_pool_cups.csv')
hand=pd.read_csv(A/'gpu/results/wilor_detector/references.csv');hand=hand[hand.query_id.isna()&hand.confidence.ge(.5)]
outputs=[]
for source in ['teacher','wilor_full']:
 fs=[]
 for split,rows in [('oof',q),('test',te)]:
  f=align(A/f'features/cup_hand_{split}.csv',rows).copy()
  if source=='wilor_full':
   for r in rows.itertuples():
    b=np.array([getattr(r,c) for c in B]);d=hand[hand.frame_id.eq(r.frame_id)];bb=d[B].to_numpy();gap=np.maximum(0,np.maximum(bb[:,:2]-b[2:],b[:2]-bb[:,2:]));dist=np.linalg.norm(gap,axis=1)/max((b[2:]-b[:2]).max(),1);j=int(dist.argmin()) if len(d) else -1
    f.loc[r.id,'nearest_hand_gap']=dist[j] if j>=0 else 99;f.loc[r.id,'hand_conf']=d.iloc[j].confidence if j>=0 else 0
  f['confirmed_iou']=align(ROOT/f'rascar_v21/object_structure_{split}.csv',rows,'best_reference_iou')
  f.to_csv(A/f'features/cup_{source}_geometry_{split}.csv');fs.append(f)
 for cut in [.7,.85]:
  for distance in [.75,1.5]:
   for geometry in [.6,.75]:
    for weight in [.2,.4]:
     res=[]
     for split,rows,base,f in [('oof',q,base_oof,fs[0]),('test',te,current,fs[1])]:
      g=rows.class_name.eq('cup').to_numpy()&guard(split,rows,'cup')&f.cup_conf.ge(cut).to_numpy()&f.nearest_hand_gap.le(distance).to_numpy()&f.hand_conf.ge(.5).to_numpy()&f.confirmed_iou.ge(geometry).to_numpy()
      res.append(base*(1-weight*g))
     name=f'cup_verified_{source}_c{cut}_d{distance}_i{geometry}_w{weight}'
     r=register(name,*res,dict(type='cup_near_hand_with_geometry_confirmation',source=source,cut=cut,distance=distance,min_reference_iou=geometry,weight=weight))
     outputs.append((name,*res,r))
choices=[('unchanged',base_oof,current,None)]+outputs;s=base_oof.copy();tests=[];selection=[]
for fit,held in zip(DAYS,DAYS[::-1]):
 m=q.day.eq(fit).to_numpy();hm=q.day.eq(held).to_numpy();best=max(choices,key=lambda a:metric(q[m],a[1][m])['pooled_ap']);s[hm]=best[1][hm];tests.append(best[2]);selection.append(dict(fit=fit,held=held,experiment=best[0]))
register('cup_verified_day_selected',s,np.mean(tests,axis=0),dict(selection=selection,type='day-separated threshold selection',pool='17 visual positives, no negative GT; audited separately'))
