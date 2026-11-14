"""Exact final score assembly from cached, day-held-out expert outputs.
GPU trained-checkpoint replay is separately recorded in spatial_checkpoint_replay.json.
"""
from pathlib import Path
import numpy as np,pandas as pd,hashlib
A=Path(__file__).resolve().parent;d=np.load(A/'replay_inputs.npz',allow_pickle=False)
b=d['baseline'];c=d['class_name'];n=len(d['new_dense_oof_iou'])
s=b+np.clip(.2*(d['p_full']-b),0,.1)+np.clip(.2*(d['p_crop']-b),-.1,0)
m=c=='nude_hands';s[m]=b[m]+np.clip(.2*(d['p_anchor_adapted'][m]-b[m]),-.1,.1)
s=np.clip(s,0,1)
o,t=d['new_dense_oof_iou'],d['new_dense_test_iou'];v22=d['v22'];dense=v22.copy()
valid=np.isfinite(o);target=np.clip((.85-o)/.5,0,1);dense[:n][valid]+=.25*(target-v22[:n])[valid]
target=np.clip((.85-t)/.5,0,1);delta=target-v22[n:][None,:]
support=np.isfinite(target).all(0)&((delta>=0).all(0)|(delta<=0).all(0))
safe=np.where(np.isfinite(target),target,v22[n:][None,:]);cons=np.where((delta>=0).all(0),safe.min(0),safe.max(0));dense[n:][support]+=.25*(cons-v22[n:])[support]
m=(c=='cup')&np.r_[valid,support];s[m]+=(dense-d['v31'])[m]
m=c=='tray';x=np.clip(b[m],.0001,.9999);s[m]=1/(1+np.exp(-(np.log(x/(1-x))-.6)))
s=np.clip(s,0,1);s=b+np.clip(s-b,-.15,.15)
assert np.allclose(s,d['expected'],rtol=0,atol=1e-14),float(abs(s-d['expected']).max())
assert np.array_equal(np.round(s,12),np.round(d['expected'],12))
out=A/'replayed_submission.csv';pd.DataFrame({'id':d['id'][n:],'score':s[n:]}).to_csv(out,index=False,float_format='%.12f')
sha=hashlib.sha256(out.read_bytes()).hexdigest();assert sha=='1ccc86a345fed7590f0462a2216aa9f07f032df0c68a63decf6cdcde2d8087d6'
print('EXACT_REPLAY_OK',len(s),'scores; CSV SHA256',sha)
