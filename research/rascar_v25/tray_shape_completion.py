"""Oriented rectangle proposals constrained by fit-day tray sizes and table location."""
from tray_geometry import bank,assign
from common25 import *
from features23 import image
from functools import lru_cache
import cv2
cv2.setNumThreads(2)
@lru_cache(maxsize=180)
def candidates(fid):
 im=image(fid);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);edge=cv2.Canny(gray,40,110)
 out=[]
 for kernel in [1,3,5]:
  ee=cv2.morphologyEx(edge,cv2.MORPH_CLOSE,np.ones((kernel,kernel),np.uint8))
  cs,_=cv2.findContours(ee,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
  for c in cs:
   rect=cv2.minAreaRect(c);w,hh=rect[1];area=w*hh
   if min(w,hh)<15 or max(w,hh)>145 or area<600:continue
   fill=cv2.contourArea(c)/max(area,1)
   poly=cv2.approxPolyDP(c,.025*cv2.arcLength(c,True),True)
   if fill<.79 or not 4<=len(poly)<=6:continue
   pts=cv2.boxPoints(rect);b=np.r_[pts.min(0),pts.max(0)]
   out.append(dict(b=b,fill=fill,angle=rect[2],side1=w,side2=hh,kernel=kernel))
 return out

def features(rows,fit):
 anchors,bk=bank(fit);result=[]
 for r in rows.itertuples():
  f=dict(id=r.id,shape_iou=0.,shape_fit=0.,rectangularity=0.,table=-1,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
  if r.class_name!='tray':result.append(f);continue
  b=np.array([getattr(r,c) for c in B]);j,rel=assign(b,anchors);best=-1
  nearby=bk[bk.table.eq(j)&bk.frame_id.ne(r.frame_id)];local=nearby if len(nearby)>=2 else bk[bk.frame_id.ne(r.frame_id)]
  if len(local)==0:result.append(f);continue
  tw=anchors[j,2:]-anchors[j,:2]
  for z in candidates(r.frame_id):
   bb=z['b'];k,rr=assign(bb,anchors)
   if k!=j or max(abs(rr))>.7:continue
   wh=bb[2:]-bb[:2];expected=local[['nw','nh']].to_numpy()*tw
   mismatch=np.min(np.linalg.norm(np.log(wh/expected),axis=1));fit=np.exp(-mismatch)
   ov=float(iou(b,bb)[0,0]);merit=ov*fit*z['fill']
   if merit>best:
    best=merit;f.update(shape_iou=ov,shape_fit=fit,rectangularity=z['fill'],table=int(j),**dict(zip(['ref_'+x for x in B],bb)))
  result.append(f)
 return pd.DataFrame(result).set_index('id')
if __name__=='__main__':
 of=[];tf=[];syn=[]
 for fit,held in zip(DAYS,DAYS[::-1]):
  o=features(q[q.day.eq(held)],fit);t=features(te,fit);of.append(o);tf.append(t)
  o.to_csv(A/f'features/tray_shape_held_{fit}.csv');t.to_csv(A/f'features/tray_shape_test_{fit}.csv')
  # A controlled boundary-recovery experiment; keep it separate from AP.
  records=[]
  for idx,r in h[h.day.eq(held)&h.class_name.eq('tray')].iterrows():
   for side in range(4):
    rec=r.to_dict();bb=r[B].to_numpy(float);wh=bb[2:]-bb[:2];bb[side]+=(1 if side<2 else -1)*.35*wh[side%2]
    rec.update(dict(zip(B,bb)));rec.update(id=f'synthetic_{idx}_{side}',original_id=idx,cut_side=side);records.append(rec)
  z=pd.DataFrame(records);f=features(z,fit)
  for r in z.itertuples():
   ff=f.loc[r.id];gt=h.loc[r.original_id,B].to_numpy(float);p=ff[['ref_'+x for x in B]].to_numpy(float)
   syn.append(dict(id=r.id,day=held,frame_id=r.frame_id,old_iou=float(iou([getattr(r,c) for c in B],gt)[0,0]),new_iou=float(iou(p,gt)[0,0]) if ff.shape_fit>=.55 and ff.rectangularity>=.85 else float(iou([getattr(r,c) for c in B],gt)[0,0]),active=int(ff.shape_fit>=.55 and ff.rectangularity>=.85),shape_fit=ff.shape_fit))
 fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
 for cut in [.55,.7]:
  for rectify in [.8,.9]:
   for weight in [.25,.5]:
    outs=[]
    for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
     g=rows.class_name.eq('tray').to_numpy()&guard(split,rows,'tray')&f.shape_fit.ge(cut).to_numpy()&f.rectangularity.ge(rectify).to_numpy()&f.shape_iou.ge(.3).to_numpy()
     p=np.clip((.75-f.shape_iou.to_numpy())/.5,0,1);outs.append(np.clip(base+weight*g*(p-base),0,1))
    register(f'tray_rectangular_fit{cut}_rect{rectify}_w{weight}',*outs,dict(type='oriented_rectangle_completion',template_fit=cut,rectangularity=rectify,weight=weight))
 pd.DataFrame(syn).to_csv(A/'results/tray_synthetic_recovery.csv',index=False)
 print('SYNTHETIC',pd.DataFrame(syn).groupby('day')[['old_iou','new_iou','active']].mean().to_dict())
