from common16 import *
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
heads=pd.read_csv(V16/'wevis_heads_references.csv')
templates=[]
for fid,g in h[h.class_name.eq('person')].groupby('frame_id'):
    rr=heads[heads.frame_id.eq(fid)]
    if rr.empty:continue
    bb=g[B].to_numpy();hh=rr[B].to_numpy();centers=(hh[:,:2]+hh[:,2:])/2
    inside=((centers[None]>=bb[:,None,:2])&(centers[None]<=bb[:,None,2:])).all(2)
    dx=np.abs(centers[None,:,0]-(bb[:,None,0]+bb[:,None,2])/2)/np.maximum(bb[:,None,2]-bb[:,None,0],1)
    dy=(centers[None,:,1]-bb[:,None,1])/np.maximum(bb[:,None,3]-bb[:,None,1],1)
    reward=np.c_[np.where(inside,10-dx-dy,np.zeros_like(dx)),np.zeros((len(bb),len(bb)))];ii,jj=linear_sum_assignment(-reward)
    for i,j in zip(ii,jj):
        if j>=len(hh) or not inside[i,j]:continue
        templates.append(dict(frame_id=fid,day=g.iloc[i].day,head_x=centers[j,0],head_y=centers[j,1],head_w=hh[j,2]-hh[j,0],head_h=hh[j,3]-hh[j,1],**dict(zip(B,bb[i]))))
bank=pd.DataFrame(templates);bank.to_csv(V16/'head_geometry_templates.csv',index=False)
def predict(rows,fit=None):
    pool=bank if fit is None else bank[bank.day.eq(fit)];pb=pool[B].to_numpy();pc=pool[['head_x','head_y']].to_numpy();sizes=pool[['head_w','head_h']].to_numpy();result=[]
    for r in rows.itertuples():
        rec=dict(id=r.id,head_anchor_risk=0.,template_iou_q80=1.,template_iou_mean=1.,templates=0,head_count=0)
        if r.class_name!='person':result.append(rec);continue
        b=np.array([getattr(r,c) for c in B]);rr=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy()
        if not len(rr):result.append(rec);continue
        cc=(rr[:,:2]+rr[:,2:])/2;inside=((cc>=b[:2])&(cc<=b[2:])).all(1);rec['head_count']=int(inside.sum())
        if inside.sum()!=1:result.append(rec);continue
        hb=rr[inside][0];center=(hb[:2]+hb[2:])/2;size=hb[2:]-hb[:2]
        dist=np.linalg.norm((pc-center)/[1280,720],axis=1)
        eligible=(dist<.085)&(np.abs(np.log(np.maximum(sizes[:,0],1)/max(size[0],1)))<.55)
        ix=np.where(eligible)[0];ix=ix[np.argsort(dist[ix])[:5]]
        if len(ix)<2:result.append(rec);continue
        translated=pb[ix]+np.tile(center-pc[ix],(1,2));ov=iou(b,translated)[0]
        weights=np.exp(-dist[ix]**2/(2*.035**2));weights/=weights.sum();order=np.argsort(ov);cdf=np.cumsum(weights[order]);high=float(ov[order[np.searchsorted(cdf,.8)]])
        risk=float(expit((.6-high)/.06));aspect=(b[2]-b[0])/(b[3]-b[1]);wide=float(np.clip((1.8-aspect)/.4,0,1))
        rec.update(head_anchor_risk=risk*wide,template_iou_q80=high,template_iou_mean=float(ov@weights),templates=len(ix));result.append(rec)
    return pd.DataFrame(result)
parts=[]
for fit,held in zip(DAYS,DAYS[::-1]):parts.append(predict(q[q.day.eq(held)],fit))
f=pd.concat(parts).set_index('id').loc[q.id].reset_index();ft=predict(te)
f.to_csv(V16/'head_geometry_oof.csv',index=False);ft.to_csv(V16/'head_geometry_test.csv',index=False)
for w in [.25,.5,.75]:
    s=q.score+w*(1-q.score)*f.head_anchor_risk.to_numpy();t=current+w*(1-current)*ft.head_anchor_risk.to_numpy()
    register(f'head_geometry_{w}',s,t,dict(weight=w,template_rows=len(bank),fit_days=DAYS,quantile=.8,minimum_templates=2))
