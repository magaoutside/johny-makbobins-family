"""Table-conditioned size completion of independent tray references, cross-day."""
from common25 import *
from features23 import frame_objects,image
from sklearn.cluster import KMeans
from functools import lru_cache
import cv2
cv2.setNumThreads(2)
@lru_cache(None)
def bank(fit):
 tb=h[h.day.eq(fit)&h.class_name.eq('table')][B].to_numpy();cent=(tb[:,:2]+tb[:,2:])/2
 km=KMeans(6,random_state=25,n_init=10).fit(cent)
 anchors=np.array([np.median(tb[km.labels_==j],axis=0) for j in range(6)]);anchors=anchors[np.argsort(anchors[:,0])]
 out=[]
 for r in h[h.day.eq(fit)&h.class_name.eq('tray')].itertuples():
  b=np.array([getattr(r,c) for c in B]);j,rel=assign(b,anchors);tw=anchors[j,2:]-anchors[j,:2]
  out.append(dict(frame_id=r.frame_id,table=j,width=b[2]-b[0],height=b[3]-b[1],nw=(b[2]-b[0])/tw[0],nh=(b[3]-b[1])/tw[1],rel_x=rel[0],rel_y=rel[1]))
 return anchors,pd.DataFrame(out)
def assign(b,anchors):
 cc=(b[:2]+b[2:])/2;wh=anchors[:,2:]-anchors[:,:2];rel=(cc-(anchors[:,2:]+anchors[:,:2])/2)/wh
 j=np.linalg.norm(rel,axis=1).argmin();return j,rel[j]
def edge_support(im,b):
 # Fixed 3-pixel strips on original image; independent of query bounds.
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ed=cv2.Canny(gray,55,120).astype(float)/255
 x1,y1,x2,y2=np.clip(np.round(b),0,[1278,718,1279,719]).astype(int)
 if x2<=x1+2 or y2<=y1+2:return 0.
 vals=[ed[max(0,y1-2):y1+3,x1:x2].max(0).mean(),ed[y2-2:y2+3,x1:x2].max(0).mean(),ed[y1:y2,max(0,x1-2):x1+3].max(1).mean(),ed[y1:y2,x2-2:x2+3].max(1).mean()]
 # Require >=3 supported boundaries, a single table edge is insufficient.
 return float(np.sort(vals)[1])
def features(rows,fit,mode):
 anchors,bk=bank(fit);out=[]
 for r in rows.itertuples():
  f=dict(id=r.id,eligible=0,table=-1,supports=0,old_iou=0.,new_iou=0.,edge=0.,old_edge=0.,shape_ratio=1.,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
  if r.class_name!='tray':out.append(f);continue
  b=np.array([getattr(r,c) for c in B]);rr=frame_objects(r.frame_id)['tray'];rr=rr[rr.confidence.ge(.5)]
  if not len(rr):out.append(f);continue
  # Associate query to independent reference, but never fit extension to query.
  jj=iou(b,rr[B]).argmax();rb=rr.iloc[jj][B].to_numpy(float)
  j,rel=assign(rb,anchors);tw=anchors[j,2:]-anchors[j,:2]
  nearby=bk[bk.table.eq(j)&bk.frame_id.ne(r.frame_id)]
  if mode=='per_table' and len(nearby)<2:out.append(f);continue
  local=nearby if len(nearby)>=2 else bk[bk.frame_id.ne(r.frame_id)]
  if not len(local):out.append(f);continue
  # Both orientations allowed; choose the least alteration compatible with observed extent.
  wh=rb[2:]-rb[:2]
  target=np.array([local.nw.median()*tw[0],local.nh.median()*tw[1]])
  if mode=='fixed_pixels':target=local[['width','height']].median().to_numpy()
  deficit=np.maximum(0,target-wh);maxdelta=np.minimum(deficit,wh*.6)
  opts=[]
  for fx,fy in [(0,0),(.5,.5),(0,1),(1,0),(1,1),(.5,0),(.5,1),(0,.5),(1,.5)]:
   new=np.r_[rb[:2]-maxdelta*[fx,fy],rb[2:]+maxdelta*(1-np.array([fx,fy]))]
   new=np.clip(new,0,[1280,720,1280,720]);opts.append((edge_support(image(r.frame_id),new),new))
  strength,nb=max(opts,key=lambda z:z[0]);oldedge=edge_support(image(r.frame_id),rb)
  eligible=abs(rel[0])<.65 and abs(rel[1])<.65 and min(wh/np.maximum(target,1))<.85 and max(wh/np.maximum(target,1))<1.4
  f.update(eligible=int(eligible),table=int(j),supports=len(nearby),old_iou=float(iou(b,rb)[0,0]),new_iou=float(iou(b,nb)[0,0]),edge=strength,old_edge=oldedge,shape_ratio=float(min(wh/np.maximum(target,1))),**dict(zip(['ref_'+v for v in B],nb)))
  out.append(f)
 return pd.DataFrame(out).set_index('id')
if __name__=='__main__':
 for fit in DAYS:
  aa,bb=bank(fit);pd.DataFrame(aa,columns=B).to_csv(A/f'inputs/table_templates_{fit}.csv',index=False);bb.to_csv(A/f'inputs/tray_sizes_{fit}.csv',index=False)
 for mode in ['per_table','hierarchical','fixed_pixels']:
  of=[];tf=[]
  for fit,held in zip(DAYS,DAYS[::-1]):
   o=features(q[q.day.eq(held)],fit,mode);t=features(te,fit,mode);of.append(o);tf.append(t)
   o.to_csv(A/f'features/tray_{mode}_held_{fit}.csv');t.to_csv(A/f'features/tray_{mode}_test_{fit}.csv')
  fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
  for edge in [0.,.15,.3]:
   for weight in [.25,.5]:
    outs=[]
    for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
     gate=f.eligible.ge(.99).to_numpy()&f.edge.ge(edge).to_numpy()&guard(split,rows,'tray')
     if edge:gate &= f.edge.ge(f.old_edge*.8).to_numpy()
     # Compare geometric errors before/after completion; avoid replacing calibrated score wholesale.
     old=np.clip((.7-f.old_iou.to_numpy())/.4,0,1);new=np.clip((.7-f.new_iou.to_numpy())/.4,0,1)
     score=np.clip(base+weight*(new-old)*gate,0,1);outs.append(score)
    register(f'tray_{mode}_edge{edge}_w{weight}',*outs,dict(type='table_conditioned_completion',mode=mode,minimum_3rd_boundary_edge=edge,weight=weight,fit='opposite-day manual GT only',test_ids_as_rules=False))
