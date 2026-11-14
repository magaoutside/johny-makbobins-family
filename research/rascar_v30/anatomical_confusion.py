"""A hand false-positive pattern: exposed elbow mistaken for a palm.

Require agreement over multiple pose views: query center lies nearer the elbow
than the wrist and upstream along the forearm. This is a soft penalty, not an
assumption that hands never touch elbows. Frozen pose outputs; no test labels.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,DAYS,current,base_oof,align
from features23 import frame_pose
from bounded_geometry import save

def describe(r):
    b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;scale=np.sqrt(wh.prod());votes={};evidence=[]
    for pr,kp in frame_pose(r.frame_id):
        wrists=kp[[9,10]];dist=np.linalg.norm(wrists[:,:2]-cc,axis=1)/scale
        if np.any(wrists[:,2]>.6):minw=float(dist[wrists[:,2]>.6].min())
        else:continue
        for e,w in [(7,9),(8,10)]:
            if min(kp[e,2],kp[w,2])<.65:continue
            elbow,wrist=kp[e,:2],kp[w,:2];axis=wrist-elbow;length=np.linalg.norm(axis)
            if not 10<length<180:continue
            de=np.linalg.norm(cc-elbow)/scale;dw=np.linalg.norm(cc-wrist)/scale
            proximal=(cc-wrist)@axis/(length**2)
            inside=((elbow>=b[:2])&(elbow<=b[2:])).all()
            vote=bool(inside and de<.5 and dw>max(.45,2*de) and proximal<-.35 and minw>.4)
            if vote:votes[pr.view]=True
            if de<1.:evidence.append((de,dw,proximal,minw,pr.view))
    best=min(evidence) if evidence else (9.,0.,0.,0.,-1)
    return dict(id=r.id,elbow_views=len(votes),elbow_distance=best[0],wrist_distance=best[1],
                projection=best[2],nearest_wrist=best[3])

def run():
    fs={}
    for split,rows in [('oof',q),('test',te)]:
        path=A/f'features/elbow_{split}.csv'
        if not path.exists():pd.DataFrame([describe(r) for r in rows.itertuples()]).to_csv(path,index=False)
        fs[split]=pd.read_csv(path)
    # Verify the geometry guard directly on true manual hands as a separate
    # positive-control set; report it without treating it as extra AP labels.
    gt=h[h.class_name.eq('nude_hands')].copy();gt['id']=['gt_'+str(i) for i in gt.index]
    fg=pd.DataFrame([dict(day=r.day,**describe(r)) for r in gt.itertuples()])
    fg.to_csv(A/'audit/elbow_manual_hand_controls.csv',index=False)
    results=[]
    for views in [2,3]:
        for semcut in [.7,.85]:
            for weight in [.25,.5]:
                outs=[]
                for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
                    sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_nude_hands')
                    gate=rows.class_name.eq('nude_hands').to_numpy()&fs[split].elbow_views.ge(views).to_numpy()&(sem<semcut)
                    outs.append(base+weight*np.maximum(.9-base,0)*gate)
                results.append(save(f'elbow_v{views}_sem{semcut}_w{weight}',*outs,dict(type='anatomical_elbow_confusion',
                    views=views,semantic_max=semcut,weight=weight,parameters='fixed geometric ratios, no fitted thresholds',test_labels_used=False)))
    pd.DataFrame(results).drop(columns='details').to_csv(A/'results/elbow_metrics.csv',index=False)
    print(pd.DataFrame(results).drop(columns='details').to_string(index=False))
    print('MANUAL FALSE FLAGS',fg.groupby('day').elbow_views.apply(lambda s:dict(n=len(s),v2=int((s>=2).sum()),v3=int((s>=3).sum()))).to_dict())

if __name__=='__main__':run()
