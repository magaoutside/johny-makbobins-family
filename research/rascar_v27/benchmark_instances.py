from contained_instances import *

records=[]
def add(fid,day,b,kind,gt):
    records.append(dict(id=f'instance27_{len(records)}',frame_id=fid,day=day,class_name='person',kind=kind,
        error=int(iou(b,gt).max()<.6),**dict(zip(B,b))))
for (fid,day),g in h[h.class_name.eq('person')].groupby(['frame_id','day']):
    gt=g[B].to_numpy()
    for box in gt:
        add(fid,day,box,'intact',gt)
        cc=(box[:2]+box[2:])/2;wh=box[2:]-box[:2]
        for scale in [.85,1.15,1.35]:add(fid,day,np.r_[cc-wh*scale/2,cc+wh*scale/2],'scaled',gt)
    for a,b in combinations(gt,2):
        wh1=a[2:]-a[:2];wh2=b[2:]-b[:2];cc1=(a[:2]+a[2:])/2;cc2=(b[:2]+b[2:])/2
        if np.linalg.norm((cc1-cc2)/np.maximum(wh1,wh2))<1.4:
            add(fid,day,np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])],'merged',gt)
d=pd.DataFrame(records);f=features(d,'oof');d['flag']=f.merged.to_numpy();d.to_csv(A/'results/contained_instances_benchmark.csv',index=False)
summary=[]
for (day,kind),g in d.groupby(['day','kind']):
    summary.append(dict(day=day,kind=kind,rows=len(g),errors=int(g.error.sum()),flagged=int(g.flag.sum()),
                        true_flags=int((g.flag&g.error.eq(1)).sum()),false_flags=int((g.flag&g.error.eq(0)).sum())))
pd.DataFrame(summary).to_csv(A/'results/contained_instances_benchmark_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
