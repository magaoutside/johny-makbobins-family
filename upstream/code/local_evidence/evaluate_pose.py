from common import *
pose=pd.read_csv(A/'pose_references.csv')
def evidence(rows):
 out=[]
 for r in rows.itertuples():
  rec=dict(id=r.id,pose_iou=0.,head_outside=0.,multiple_heads=0.,body_outside=0.,pose_confidence=0.)
  if r.class_name!='person':out.append(rec);continue
  b=np.array([getattr(r,c) for c in B]);pool=pose[(pose.frame_id==r.frame_id)&(pose.confidence>=.3)].reset_index(drop=True)
  if pool.empty:out.append(rec);continue
  ov=iou(b,pool[B])[0];j=int(ov.argmax());ref=pool.iloc[j];rec['pose_iou']=float(ov[j]);rec['pose_confidence']=float(ref.confidence)
  heads=[]
  for rr in pool.itertuples():
   points=np.array([[getattr(rr,f'kp{k}_x'),getattr(rr,f'kp{k}_y'),getattr(rr,f'kp{k}_conf')] for k in range(5)]);valid=points[:,2]>=.35
   if valid.sum()<2:continue
   xy=np.average(points[valid,:2],axis=0,weights=points[valid,2]);heads.append((rr.Index,xy))
  for ix,xy in heads:
   outside=not (b[0]-5<=xy[0]<=b[2]+5 and b[1]-5<=xy[1]<=b[3]+5)
   if ix==j and ov[j]>=.2 and outside:rec['head_outside']=1.
  inside=[ix for ix,xy in heads if b[0]<=xy[0]<=b[2] and b[1]<=xy[1]<=b[3] and ov[ix]>=.1];rec['multiple_heads']=float(len(inside)>1)
  if ov[j]>=.3:
   pts=np.array([[ref[f'kp{k}_x'],ref[f'kp{k}_y'],ref[f'kp{k}_conf']] for k in range(5,17)]);valid=pts[:,2]>=.5;outside=(pts[:,0]<b[0]-8)|(pts[:,0]>b[2]+8)|(pts[:,1]<b[1]-8)|(pts[:,1]>b[3]+8)
   rec['body_outside']=float(np.clip((valid&outside).sum()/3,0,1))
  out.append(rec)
 return pd.DataFrame(out)
f=evidence(q);ft=evidence(te);f.to_csv(A/'pose_oof_features.csv',index=False);ft.to_csv(A/'pose_test_features.csv',index=False)
for mode in ['head','body','combined']:
 signal=f.head_outside.to_numpy() if mode=='head' else f.body_outside.to_numpy() if mode=='body' else np.maximum.reduce([f.head_outside,f.body_outside,.5*f.multiple_heads])
 signalt=ft.head_outside.to_numpy() if mode=='head' else ft.body_outside.to_numpy() if mode=='body' else np.maximum.reduce([ft.head_outside,ft.body_outside,.5*ft.multiple_heads])
 for w in [.2,.4]:register(f'pose_{mode}_{w:g}',q.score+w*(1-q.score)*signal,current+w*(1-current)*signalt,dict(weight=w,mode=mode,model='frozen COCO YOLO26m-pose; no fitting on Rascar'))
