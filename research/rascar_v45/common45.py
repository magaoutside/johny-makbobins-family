from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,baseline,iou,metrics,B,CL,DAYS,D
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
for folder in ['inputs','features','results/oof','results/submissions','audit','logs','pool_review']:
    (A/folder).mkdir(parents=True,exist_ok=True)

def load_jsonl(kind):
    return {r['frame_id']:r for r in map(json.loads,(ROOT/f'rascar_next_association/features/{kind}.jsonl').read_text().splitlines())}

def anchors(frame,cls,heads,hands):
    """Independent pretrained BPJ anchors, no query or ground truth used."""
    boxes=[];points=[];scores=[]
    d=heads[frame] if cls=='person' else hands[frame]
    for p in d['parts']:
        if p['confidence']<.3 or len(p['associations'])!=1:continue
        body=d['bodies'][p['associations'][0]['body_id']]
        if body['confidence']<.3:continue
        bb=np.array(body['xyxy'] if cls=='person' else p['xyxy'],float)
        pp=(np.array(p['xyxy'][:2])+p['xyxy'][2:])/2
        if boxes and iou(bb,boxes).max()>.65:continue
        boxes.append(bb);points.append(pp);scores.append(min(body['confidence'],p['confidence']))
    return np.array(boxes).reshape(-1,4),np.array(points).reshape(-1,2),np.array(scores)

def v22base(q,te):
    def aligned(path,r):return pd.read_csv(ROOT/path).set_index('id').loc[r.id,'score'].to_numpy()
    oo=aligned('rascar_v22/results/oof/final_reviewed.csv',q)
    oo-=aligned('rascar_v18/results/oof/release_main.csv',q)-aligned('rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q)
    return np.clip(oo,0,1),aligned('rascar_v22/submission_v22_candidate.csv',te)

def compose(base,overlaps):
    t=np.clip((.85-np.array(overlaps))/.5,0,1)
    out=base.copy()
    if len(t)==1:
        ok=np.isfinite(t[0]);out[ok]=.75*base[ok]+.25*t[0,ok]
    else:
        up=(t>=base).all(0);down=(t<=base).all(0);ok=np.isfinite(t).all(0)&(up|down)
        target=np.where(up,np.min(t,0),np.max(t,0));out[ok]=.75*base[ok]+.25*target[ok]
    return out
