"""Pixel boundary training records, without any test visual labels."""
from common22 import *
rng=np.random.default_rng(2210)
out=[]
def add(fid,day,cls,b,gt,kind,weight=1.,error=None,ident=None):
    b=np.clip(b,0,[1280,720,1280,720])
    if min(b[2:]-b[:2])<5:return
    best=float(iou(b,gt).max()) if len(gt) else 0.
    out.append(dict(id=ident or f'spatial22_{len(out)}',frame_id=fid,day=day,class_name=cls,error=int(best<.6) if error is None else error,target_iou=best,kind=kind,weight=weight,split='train',**dict(zip(B,b))))
for day in DAYS:
    for (fid,cls),g in h[h.day.eq(day)&h.class_name.isin(['nude_hands','cup','tray'])].groupby(['frame_id','class_name']):
        gt=g[B].to_numpy()
        for b in gt:
            cc=(b[:2]+b[2:])/2;wh=b[2:]-b[:2]
            add(fid,day,cls,b,gt,'gt',2.)
            for scale in [.55,.7,.82,.9,1.12,1.25,1.4,1.65]:
                add(fid,day,cls,np.r_[cc-wh*scale/2,cc+wh*scale/2],gt,'scale')
            for _ in range(4):
                delta=rng.uniform(-.75,.75,2)*wh
                add(fid,day,cls,b+np.tile(delta,2),gt,'shift')
            dist=np.linalg.norm(((gt[:,:2]+gt[:,2:])/2-cc)/np.maximum(wh,12),axis=1)
            for j in np.argsort(dist)[1:3]:
                if dist[j]<4:
                    merged=np.r_[np.minimum(b[:2],gt[j,:2]),np.maximum(b[2:],gt[j,2:])]
                    add(fid,day,cls,merged,gt,'merge',2.)
        # Other annotated classes supply semantic hard negatives with known content.
        other=h[h.frame_id.eq(fid)&~h.class_name.eq(cls)&h.class_name.isin(CL)]
        if len(other):
            for _,r in other.sample(n=min(len(other),8),random_state=2210).iterrows():
                add(fid,day,cls,r[B].to_numpy(float),gt,'wrong_class',.7)
    for r in q[q.day.eq(day)&~q.class_name.eq('person')].itertuples():
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy()
        add(r.frame_id,day,r.class_name,np.array([getattr(r,c) for c in B]),gt,'real_query',5.,r.error,r.id)
    pool=pd.read_csv(A/'inputs/reviewed_pool_hands.csv')
    for fid,g in pool[pool.day.eq(day)].groupby('frame_id'):
        gt=g[B].to_numpy()
        for b in gt:
            cc=(b[:2]+b[2:])/2;wh=b[2:]-b[:2]
            for scale in [.6,.85,1,1.2,1.5]:add(fid,day,'nude_hands',np.r_[cc-wh*scale/2,cc+wh*scale/2],gt,'reviewed_pool',.35)
    crops=pd.read_csv(A/'inputs/reviewed_pool_crops.csv')
    for r in crops[crops.day.eq(day)&crops.boxes.eq(0)].itertuples():
        cr=np.array([r.roi_x1,r.roi_y1,r.roi_x2,r.roi_y2]);cc=(cr[:2]+cr[2:])/2
        for size in [24,40,60]:add(r.frame_id,day,'nude_hands',np.r_[cc-size/2,cc+size/2],[], 'reviewed_negative',.35)
d=pd.DataFrame(out)
# Cap repeated synthetic variations per class/day; retain real examples and reviewed pool.
keep=[]
for _,g in d.groupby(['day','class_name']):
    fixed=g[g.kind.isin(['real_query','real','gt','reviewed_pool','reviewed_negative'])]
    rest=g.drop(fixed.index)
    if len(rest)>1800:rest=rest.sample(1800,random_state=2210)
    keep.extend([fixed,rest])
d=pd.concat(keep,ignore_index=True)
for split,rows in [('held',q),('test',te)]:
    rr=rows.copy();rr['split']=split;rr['weight']=1.;rr['kind']='evaluation';d=pd.concat([d,rr],ignore_index=True)
d['record_index']=np.arange(len(d));d.to_csv(A/'inputs/spatial_records.csv',index=False)
print(d.groupby(['split','day','class_name']).size().to_string())
print('records',len(d),'unique crops',len(d.drop_duplicates(['frame_id']+B)))
