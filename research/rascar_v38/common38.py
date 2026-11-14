from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v38'
sys.path.insert(0,str(ROOT/'rascar_v37'))
from common37 import B,CL,DAYS,q,te,h,BASE,iou
for d in ['inputs','features','models','audit','logs','results/oof','results/submissions']:(A/d).mkdir(parents=True,exist_ok=True)

def metric(rows,s):
    out={'ap':float(AP(rows.error,s))}
    for col in ['day','class_name']:
        for key,g in rows.groupby(col):
            mask=rows[col].eq(key);out[key]=float(AP(rows.loc[mask,'error'],np.asarray(s)[mask]))
    return out

def save(name,scores,description):
    out=dict(experiment=name,**metric(q,scores['oof']),description=description,public_ap=None,submitted=False)
    changes=[]
    for split,rows in [('oof',q),('test',te)]:
        s=scores[split];assert len(s)==len(rows) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
        dest=A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}.csv')
        (rows.assign(score=s) if split=='oof' else rows[['id']].assign(score=s)).to_csv(dest,index=False)
        diff=rows.assign(split=split,v31_score=BASE[split],new_score=s,delta=s-BASE[split]);diff=diff[abs(diff.delta)>1e-10];changes.append(diff);out['changed_'+split]=len(diff)
    pd.concat(changes).to_csv(A/f'audit/{name}_changes.csv',index=False)
    print('RESULT',name,round(out['ap'],8),out['changed_oof'],out['changed_test'],flush=True);return out
