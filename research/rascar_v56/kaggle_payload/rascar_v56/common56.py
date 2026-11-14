from pathlib import Path
import os,sys,json,time,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
average_precision_score=AP
A=Path(__file__).resolve().parent;ROOT=A.parent;D=ROOT/'rascar_data_full'
B=['x1','y1','x2','y2'];CB=['crop_x1','crop_y1','crop_x2','crop_y2'];CL=['person','nude_hands','cup','tray'];DAYS=['2026-10-25','2026-10-26']
for p in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/p).mkdir(parents=True,exist_ok=True)
def inputs():return tuple(pd.read_csv(A/f'inputs/{x}.csv') for x in ['q','t','h'])
def incumbent(q,t):return q.v52.to_numpy().copy(),t.v52.to_numpy().copy()
def baseline(q,t):return q.v31.to_numpy().copy(),t.v31.to_numpy().copy()
def predecessors(q,t):return q.v22.to_numpy().copy(),t.v22.to_numpy().copy()

def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 z=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return z/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-z,1e-9)
def metrics(d,s):
 out={'ap':float(average_precision_score(d.error,s))}
 for col in ['day','class_name']:
  for k in sorted(d[col].unique()):
   m=d[col].eq(k);out[k]=float(average_precision_score(d.loc[m,'error'],np.asarray(s)[m]))
 return out
def score_dense(oo,tt,q,t):
    pre_o,pre_t=predecessors(q,t)
    target_o=np.clip((.85-oo)/.5,0,1);target_t=np.clip((.85-tt)/.5,0,1)
    s=pre_o.copy();valid=np.isfinite(oo);s[valid]+=.25*(target_o[valid]-pre_o[valid])
    delta=target_t-pre_t[None,:];same=((delta>=0).all(0)|(delta<=0).all(0))&np.isfinite(target_t).all(0)
    safe=np.where(np.isfinite(target_t),target_t,pre_t[None,:]);conservative=np.where((delta>=0).all(0),safe.min(0),safe.max(0))
    z=pre_t.copy();z[same]+=.25*(conservative[same]-pre_t[same])
    return s,z

def save_score(name,s,z,extra=None):
    q,t,h=inputs();bo,bt=incumbent(q,t)
    assert len(s)==len(q) and len(z)==len(t)
    assert np.isfinite(s).all() and np.isfinite(z).all()
    assert ((s>=0)&(s<=1)).all() and ((z>=0)&(z<=1)).all()
    q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
    t[['id']].assign(score=z).to_csv(A/f'results/submissions/{name}.csv',index=False)
    return dict(name=name,**metrics(q,s),delta=AP(q.error,s)-AP(q.error,bo),changed_test=int((abs(z-bt)>1e-9).sum()),max_test_change=float(abs(z-bt).max()),**(extra or {}))
def dense_variants(name,oo,tt,extra=None):
    q,t,h=inputs();bo,bt=incumbent(q,t);vo,vt=baseline(q,t)
    so,st=score_dense(oo,tt,q,t);out=[]
    for scope in ['all','person','nude_hands','cup','tray']:
        om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool)
        tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
        for w in [.25,.5,1.]:
            s,z=bo.copy(),bt.copy()
            s[om]+=w*(so-vo)[om]*np.where(q.class_name.eq('person'),1/1.2,1)[om]
            z[tm]+=w*(st-vt)[tm]*np.where(t.class_name.eq('person'),1/1.2,1)[tm]
            out.append(save_score(f'{name}_{scope}_w{int(w*100)}',s,z,dict(scope=scope,weight=w,**(extra or {}))))
    return out
