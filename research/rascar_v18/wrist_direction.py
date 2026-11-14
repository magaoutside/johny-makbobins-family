"""Distal-vs-proximal geometry of a query relative to elbow and wrist."""
from common18 import *
poses=pd.read_csv(V16/'multiview_pose.csv')
def extract(rows):
    out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2;obs=[]
        if r.class_name=='nude_hands':
            for pp in poses[poses.frame_id.eq(r.frame_id)&poses.confidence.ge(.45)].itertuples():
                best=None
                for e,w in [(7,9),(8,10)]:
                    confidence=min(getattr(pp,f'kp{e}_conf'),getattr(pp,f'kp{w}_conf'))
                    if confidence<.55:continue
                    elbow=np.array([getattr(pp,f'kp{e}_{c}') for c in ['x','y']]);wrist=np.array([getattr(pp,f'kp{w}_{c}') for c in ['x','y']]);axis=wrist-elbow;length=np.linalg.norm(axis)
                    if length<10:continue
                    t=(center-wrist)@axis/length**2;perp=abs(np.cross(axis/length,center-wrist))/max(np.sqrt(wh.prod()),1);distance=np.linalg.norm((wrist-center)/wh)
                    if distance>1.2 or perp>.65:continue
                    rec=dict(view=pp.view,t=float(t),perp=float(perp),distance=float(distance),confidence=confidence,wrist_x=wrist[0],wrist_y=wrist[1],elbow_x=elbow[0],elbow_y=elbow[1])
                    if best is None or distance<best['distance']:best=rec
                if best is not None:obs.append(best)
        f=pd.DataFrame(obs)
        if len(f):f=f.sort_values('confidence',ascending=False).drop_duplicates('view')
        rec=dict(id=r.id,views=len(f),distal_fraction=float((f.t>.05).mean()) if len(f) else 0.,proximal_fraction=float((f.t<-.1).mean()) if len(f) else 0.,projection=float(f.t.median()) if len(f) else 0.,perp=float(f.perp.median()) if len(f) else 9.,wrist_distance=float(f.distance.median()) if len(f) else 9.)
        if len(f):
            for c in ['wrist_x','wrist_y','elbow_x','elbow_y']:rec[c]=float(f[c].median())
        out.append(rec)
    return pd.DataFrame(out)
fo,ft=extract(q),extract(te);fo.to_csv(A/'wrist_direction_oof.csv',index=False);ft.to_csv(A/'wrist_direction_test.csv',index=False)
for weight in [.25,.5]:
    outputs=[]
    for rows,base,f,test in [(q,q.score.to_numpy(),fo,False),(te,current,ft,True)]:
        wf=align(V16/f'wevis_features_{"test" if test else "oof"}.csv',rows)
        m=(f.views.to_numpy()>=2)&(f.proximal_fraction.to_numpy()>=.75)&(f.projection.to_numpy()<-.1)&(wf.hand_iou.to_numpy()<.3)
        outputs.append(base+weight*(1-base)*m)
    register(f'forearm_direction_w{weight}',*outputs,details=dict(weight=weight,minimum_views=2,minimum_proximal_fraction=.75,projection_max=-.1,wevis_iou_max=.3))
