from pathlib import Path
import os,sys,json,time,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(A/'cache/mpl'))
os.environ.setdefault('HF_HOME',str(A/'cache/hf'))
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,baseline,metrics,iou,CL,B,CB,DAYS,D
sys.path.insert(0,str(ROOT/'rascar_v55'))
from common55 import incumbent
from train import score_dense
sys.path.pop(0)
from sklearn.metrics import average_precision_score as AP
for p in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/p).mkdir(parents=True,exist_ok=True)

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
