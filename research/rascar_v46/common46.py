"""V46: controlled dense-backbone experiment; no subjective test labels."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,baseline,iou,metrics,B,CL,DAYS,D
sys.path.pop(0)  # Do not shadow this experiment's train.py with V41/train.py.
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
for p in ['inputs','features','models','logs','results/oof','results/submissions','audit']:(A/p).mkdir(parents=True,exist_ok=True)
def records():return pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv')
def v22base(q,te):
    def read(p,rows):return pd.read_csv(ROOT/p).set_index('id').loc[rows.id,'score'].to_numpy()
    s=read('rascar_v22/results/oof/final_reviewed.csv',q)
    s-=read('rascar_v18/results/oof/release_main.csv',q)-read('rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q)
    return np.clip(s,0,1),read('rascar_v22/submission_v22_candidate.csv',te)
def geometry_target(overlap):return np.clip((.85-np.asarray(overlap))/.5,0,1)
def compose(base,overlap):
    target=geometry_target(overlap);out=base.copy()
    if target.ndim==1:
        ok=np.isfinite(target);out[ok]=.75*base[ok]+.25*target[ok]
    else:
        up=(target>=base).all(0);down=(target<=base).all(0);ok=np.isfinite(target).all(0)&(up|down)
        conservative=np.where(up,target.min(0),target.max(0));out[ok]=.75*base[ok]+.25*conservative[ok]
    return out
