"""Hand postcorrection from landmark containment and finger-chain topology.

Uses existing frozen RTMW inference. No new pose inference is claimed. Unlike
box-IoU rescues, the features distinguish palm coverage from missing fingertips
and measure excessive empty space around the skeleton. SimCC values are raw
outputs, never interpreted as probabilities. Related RTMW variants are features,
not independent model votes.
"""
from pathlib import Path
from functools import lru_cache
import sys,json,joblib
import numpy as np,pandas as pd
from catboost import CatBoostClassifier
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v30'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,DAYS,iou,align,current,base_oof
from bounded_geometry import save
MODELS=['rtmw','rtmwx_bgr','rtmwx_rgb']; BANK={}

def load():
    for model in MODELS:
        d=pd.read_csv(ROOT/f'rascar_v25/gpu/results/{model}/keypoints.csv');groups={}
        for r in d.itertuples():
            pts=np.asarray(json.loads(r.xy),float);c=np.asarray(json.loads(r.confidence),float)
            b=np.r_[pts.min(0),pts.max(0)];wh=b[2:]-b[:2]
            if min(wh)<3 or max(wh)>150 or not np.isfinite(pts).all():continue
            groups.setdefault(r.frame_id,[]).append((pts,c,b))
        BANK[model]=groups

def descriptor(fid,b):
    wh=np.maximum(b[2:]-b[:2],1);f={}
    for model in MODELS:
        choices=BANK[model].get(fid,[])
        z=dict(found=0.,overlap=0.,area_ratio=0.,inside=0.,palm_inside=0.,tips_inside=0.,wrist_inside=0.,
               fully_contained_fingers=0.,cut_fingers=0.,raw_score=0.,raw_q10=0.,compactness=0.,bone_ratio=0.)
        for side in range(4):z[f'margin_{side}']=0.
        if choices:
            ovs=iou(b,[x[2] for x in choices])[0];idx=ovs.argmax();pts,c,pb=choices[idx]
            if ovs[idx]>.05:
                uv=(pts-b[:2])/wh;inside=((uv>=0)&(uv<=1)).all(1)
                palm=[0,1,5,9,13,17];tips=[4,8,12,16,20]
                fingers=[list(range(start,start+4)) for start in [1,5,9,13,17]]
                full=np.array([inside[ff].all() for ff in fingers]);cut=np.array([inside[ff].any() and not inside[ff].all() for ff in fingers])
                lengths=[]
                for finger in fingers:
                    chain=pts[[0]+finger];lengths.append(np.linalg.norm(np.diff(chain,axis=0),axis=1).sum())
                z.update(found=1.,overlap=float(ovs[idx]),area_ratio=float(np.prod(pb[2:]-pb[:2])/np.prod(wh)),
                    inside=float(inside.mean()),palm_inside=float(inside[palm].mean()),tips_inside=float(inside[tips].mean()),
                    wrist_inside=float(inside[0]),fully_contained_fingers=float(full.mean()),cut_fingers=float(cut.mean()),
                    raw_score=float(np.median(c)),raw_q10=float(np.quantile(c,.1)),
                    compactness=float(np.mean(np.linalg.norm(uv-uv.mean(0),axis=1))),
                    bone_ratio=float(max(lengths)/max(min(lengths),1)))
                margins=np.r_[uv.min(0),1-uv.max(0)]
                z.update({f'margin_{j}':float(v) for j,v in enumerate(margins)})
        f.update({model+'_'+k:v for k,v in z.items()})
    return f

def run():
    load();train=pd.read_csv(ROOT/'rascar_v24/inputs/boundary_training.csv')
    train=train[train.class_name.eq('nude_hands')].reset_index(drop=True)
    train.to_csv(A/'inputs/hand_topology_training.csv',index=False)
    frames={d:set(train[train.day.eq(d)].frame_id) for d in DAYS};assert frames[DAYS[0]].isdisjoint(frames[DAYS[1]])
    fs={}
    for split,rows in [('train',train),('oof',q),('test',te)]:
        path=A/f'features/hand_topology_{split}.csv'
        if not path.exists():
            pd.DataFrame([dict(id=r.id,**descriptor(r.frame_id,np.array([getattr(r,c) for c in B])))
                for r in rows.itertuples()]).to_csv(path,index=False)
        fs[split]=pd.read_csv(path).set_index('id').loc[rows.id].reset_index()
    cols=fs['train'].columns.drop('id').tolist();results=[]
    for depth in [2,3]:
        po=np.zeros(len(q));pt=[]
        for day in DAYS:
            fit=train.day.eq(day).to_numpy();held=q.day.ne(day).to_numpy()
            # Equalize source-frame contributions; query perturbations are not
            # independent observations and do not multiply frame confidence.
            w=train.loc[fit,'weight'].to_numpy()/train[fit].groupby('frame_id').id.transform('size').to_numpy()
            w=w/w.mean()
            m=CatBoostClassifier(iterations=180,depth=depth,l2_leaf_reg=25,learning_rate=.035,
                loss_function='Logloss',verbose=False,random_seed=3002,thread_count=2,allow_writing_files=False)
            m.fit(fs['train'].loc[fit,cols],train.loc[fit,'error'],sample_weight=w)
            joblib.dump(dict(model=m,columns=cols,training_frames=sorted(frames[day]),fit_day=day),
                        A/f'models/hand_topology_d{depth}_fit{day}.joblib')
            po[held]=m.predict_proba(fs['oof'].loc[held,cols])[:,1]
            pt.append(m.predict_proba(fs['test'][cols])[:,1])
        fo=q[['id']].assign(p_error=po);ft=te[['id']].assign(p_error=np.mean(pt,axis=0),fold_min=np.minimum(*pt),fold_max=np.maximum(*pt))
        fo.to_csv(A/f'features/topology_d{depth}_oof.csv',index=False);ft.to_csv(A/f'features/topology_d{depth}_test.csv',index=False)
        for cut in [.7,.85]:
            for weight in [.25,.5]:
                outs=[]
                for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
                    features=fs[split];overlap=features[[k+'_overlap' for k in MODELS]].median(axis=1).to_numpy()
                    p=f.p_error.to_numpy();gate=rows.class_name.eq('nude_hands').to_numpy()&(p>=cut)&(overlap>=.25)
                    if split=='test':gate&=f.fold_min.ge(cut).to_numpy()
                    outs.append(base+weight*np.maximum(p-base,0)*gate)
                results.append(save(f'hand_topology_d{depth}_p{cut}_w{weight}',*outs,dict(type='anatomical_coverage',
                    depth=depth,threshold=cut,weight=weight,training='manual synthetic + real queries, opposite day',test_labels_used=False)))
    pd.DataFrame(results).drop(columns='details').to_csv(A/'results/topology_metrics.csv',index=False)
    print(pd.DataFrame(results).drop(columns='details').to_string(index=False))

if __name__=='__main__':run()
