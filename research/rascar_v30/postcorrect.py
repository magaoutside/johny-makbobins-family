"""Reproduce the V30 submission from frozen V22/SAM/DINO outputs.

Usage: python3 postcorrect.py --directory /path/to/rascar_v30
This is postcorrection of cached CV inference, not a raw-image model pipeline.
No hardcoded query IDs, test annotations or manual per-example score overrides.
"""
from pathlib import Path
import argparse,itertools,json,hashlib
import numpy as np,pandas as pd
B=['x1','y1','x2','y2']
BASE_SHA='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'

def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    union=(a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None]-intersection
    return intersection/np.maximum(union,1e-9)

def upper_iou(query,reference,u):
    choices=[]
    for j in range(4):
        lo,hi=reference[j]-u,reference[j]+u
        choices.append(sorted(set([lo,hi]+[float(v) for v in query[[j%2,j%2+2]] if lo<=v<=hi])))
    boxes=np.array(list(itertools.product(*choices)))
    boxes=boxes[(boxes[:,2:]>boxes[:,:2]).all(1)]
    return float(iou(query,boxes).max(initial=0))

def run(directory):
    a=Path(directory);p=a/'inputs';baseline=p/'baseline_v22.csv'
    assert hashlib.sha256(baseline.read_bytes()).hexdigest()==BASE_SHA
    cfg=json.loads((p/'release_rule.json').read_text())
    sub=pd.read_csv(baseline);queries=pd.read_csv(p/'test_queries.csv').set_index('id')
    assert sub.id.is_unique and set(sub.id)==set(queries.index) and len(sub)==362
    sem=pd.read_csv(p/'semantic_test.csv').set_index('id');raw=pd.read_csv(p/'frozen_cup_references.csv')
    refs={}
    for fid,g in raw[raw.confidence.ge(.45)].groupby('frame_id'):
        keep=[]
        for ix,r in g.sort_values('confidence',ascending=False).iterrows():
            if not keep or iou(r[B],g.loc[keep,B]).max()<.5:keep.append(ix)
        refs[fid]=g.loc[keep]
    evidence=[]
    for k,ident in enumerate(sub.id):
        r=queries.loc[ident];before=float(sub.iloc[k].score)
        if r.class_name!=cfg['class_name']:continue
        rr=refs.get(r.frame_id)
        if rr is None:continue
        b=r[B].to_numpy(float);rb=rr[B].to_numpy(float);ov=iou(b,rb)[0];j=ov.argmax()
        max_iou=max(upper_iou(b,x,cfg['tolerance_px']) for x in rb)
        conf=float(rr.iloc[j].confidence);semantic=float(sem.loc[ident,'p_cup'])
        gate=(conf>=cfg['min_reference_confidence'] and semantic>=cfg['min_semantic_score']
              and ov[j]>=cfg['min_raw_iou'] and max_iou<cfg['iou_cut'])
        after=before+cfg['weight']*max(cfg['target']-before,0)*gate
        sub.loc[k,'score']=after
        evidence.append(dict(id=ident,frame_id=r.frame_id,baseline=before,score=after,delta=after-before,
            gate=gate,reference_confidence=conf,semantic=semantic,raw_iou=float(ov[j]),max_iou=max_iou))
    assert np.isfinite(sub.score).all() and sub.score.between(0,1).all()
    path=a/'submission_v30_precision_postcorrection.csv';sub.to_csv(path,index=False)
    ev=pd.DataFrame(evidence);ev.to_csv(a/'release_evidence.csv',index=False)
    print(json.dumps(dict(file=str(path),rows=len(sub),changed=int((abs(ev.delta)>1e-9).sum()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),leaderboard_submitted=False),indent=2))
    return sub,ev

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default=str(Path(__file__).resolve().parent))
    args=parser.parse_args();run(args.directory)
