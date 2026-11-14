from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v36';V31=ROOT/'rascar_v31'
for d in ['inputs','features','models','logs','results/oof','results/submissions','audit']:(A/d).mkdir(parents=True,exist_ok=True)
B=['x1','y1','x2','y2'];CB=['crop_'+b for b in B];CL=['person','nude_hands','cup','tray'];DAYS=['2026-10-25','2026-10-26']
q=pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv');te=pd.read_csv(ROOT/'rascar_v22/inputs/test_queries.csv')
h=pd.read_csv(ROOT/'rascar_v22/inputs/manual_boxes.csv');ROWS={'oof':q,'test':te}
PATHS={'oof':V31/'results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv','test':V31/'submission_v31_dense_oof098784.csv'}
def align(path,rows):return pd.read_csv(path).set_index('id').loc[rows.id]
BASE={s:align(PATHS[s],r).score.to_numpy() for s,r in ROWS.items()}
def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None]-inter,1e-9)
def metric(rows,s):
 out={'pooled_ap':float(AP(rows.error,s))}
 for field,prefix in [('day','day_'),('class_name','ap_')]:
  for k in rows[field].unique():
   m=rows[field].eq(k).to_numpy();out[prefix+(k[-2:] if field=='day' else k)]=float(AP(rows.loc[m,'error'],s[m]))
 return out
def save(name,scores,policy):
 logs=[]
 for split,rows in ROWS.items():
  s=scores[split];assert s.shape==(len(rows),) and np.isfinite(s).all() and (s>=0).all() and (s<=1).all()
  path=A/f'results/oof/{name}.csv' if split=='oof' else A/f'results/submissions/submission_{name}.csv'
  (rows.assign(score=s) if split=='oof' else rows[['id']].assign(score=s)).to_csv(path,index=False)
  d=rows.copy();d['split']=split;d['v31_score']=BASE[split];d['new_score']=s;d['delta']=s-BASE[split];logs.append(d[abs(d.delta)>1e-10])
 pd.concat(logs,ignore_index=True).to_csv(A/f'audit/{name}_changes.csv',index=False)
 m=metric(q,scores['oof']);m.update(experiment=name,delta_ap=m['pooled_ap']-metric(q,BASE['oof'])['pooled_ap'],changed_oof=int((abs(scores['oof']-BASE['oof'])>1e-10).sum()),changed_test=int((abs(scores['test']-BASE['test'])>1e-10).sum()),policy=policy,public_ap=None,submitted=False)
 print('RESULT',name,round(m['pooled_ap'],8),m['changed_oof'],m['changed_test'],flush=True);return m
