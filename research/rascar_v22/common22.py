from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
A=Path(__file__).resolve().parent;ROOT=A.parent;R=A/'results'
B=['x1','y1','x2','y2'];DAYS=['2026-10-25','2026-10-26'];CL=['person','nude_hands','cup','tray']
q=pd.read_csv(A/'inputs/baseline_oof.csv');te=pd.read_csv(A/'inputs/test_queries.csv')
h=pd.read_csv(A/'inputs/manual_boxes.csv');fr=pd.read_csv(A/'inputs/frames.csv')
def align(path,rows,column=None):
    d=pd.read_csv(path).set_index('id').loc[rows.id]
    return d[column].to_numpy() if column else d
current=align(A/'baseline_lb095417.csv',te,'score')
audit=pd.read_csv(A/'audit/frozen_test_audit.csv').set_index('id').loc[te.id].reset_index()
threshold=json.loads((A/'audit/protocol.json').read_text())['fixed_threshold']
def boolean(x):return pd.Series(x).astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    z=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return z/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-z,1e-9)
def metric(rows,s):
    out={'pooled_ap':float(AP(rows.error,s))}
    for d,g in rows.groupby('day'):out['day_'+d[-2:]]=float(AP(g.error,s[g.index]))
    for c,g in rows.groupby('class_name'):out['ap_'+c]=float(AP(g.error,s[g.index]))
    return out
def register(name,s,t,details=None):
    s=np.asarray(s,float);t=np.asarray(t,float)
    assert s.shape==(198,) and t.shape==(362,) and np.isfinite(s).all() and np.isfinite(t).all()
    assert ((s>=0)&(s<=1)).all() and ((t>=0)&(t<=1)).all()
    q.assign(score=s).to_csv(R/'oof'/f'{name}.csv',index=False)
    dst=R/'submissions'/f'submission_{name}.csv';te[['id']].assign(score=t).to_csv(dst,index=False)
    d=audit.copy();d['baseline_v20']=current;d['new_score']=t;d['delta']=t-current
    good=d.visual_status.eq('plausible_correct');bad=d.visual_status.isin(['clear_error','likely_error'])
    prior=d.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
    direction=np.where(bad,d.delta,np.where(good,-d.delta,0))
    d['diagnostic_pass']=(bad&(t>=threshold))|(good&(t<threshold))
    d.to_csv(R/'reviews'/f'{name}.csv',index=False)
    row=dict(experiment=name,**metric(q,s),details=details or {},public_lb=None,
        priority_improved_vs_v20=int((prior&(direction>.02)).sum()),priority_passes=int((prior&d.diagnostic_pass).sum()),priority_total=int(prior.sum()),
        visual_regressions_gt002=int(((bad&(d.delta<-.02))|(good&(d.delta>.02))).sum()),changed_test_rows=int((abs(t-current)>1e-9).sum()),sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    inherited=align(ROOT/'rascar_v18/results/oof/release_main.csv',q,'score')-align(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q,'score')
    row['without_inherited_oof_only_correction']=metric(q,np.clip(s-inherited,0,1))
    (R/f'{name}.json').write_text(json.dumps(row,indent=2))
    print('RESULT',name,round(row['pooled_ap'],6),'improved/pass/regress',row['priority_improved_vs_v20'],row['priority_passes'],row['visual_regressions_gt002'],flush=True)
    return row
