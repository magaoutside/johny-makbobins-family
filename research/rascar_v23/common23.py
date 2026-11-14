from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP

A=Path(__file__).resolve().parent; ROOT=A.parent; V22=ROOT/'rascar_v22'
B=['x1','y1','x2','y2']; CL=['person','nude_hands','cup','tray']; DAYS=['2026-10-25','2026-10-26']
for folder in ['inputs','models','features','results/oof','results/submissions','results/reviews','audit','logs']:
    (A/folder).mkdir(parents=True,exist_ok=True)
baseline=V22/'submission_v22_candidate.csv'
assert hashlib.sha256(baseline.read_bytes()).hexdigest()=='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
q=pd.read_csv(V22/'results/oof/final_reviewed.csv');te=pd.read_csv(V22/'inputs/test_queries.csv')
h=pd.read_csv(V22/'inputs/manual_boxes.csv');fr=pd.read_csv(V22/'inputs/frames.csv')
def align(path,rows,column=None):
    d=pd.read_csv(path).set_index('id').loc[rows.id]
    return d[column].to_numpy() if column else d
current=align(baseline,te,'score')
inherited=align(ROOT/'rascar_v18/results/oof/release_main.csv',q,'score')-align(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q,'score')
# This is the primary OOF baseline. Keep the historical score as a second column.
base_oof=np.clip(q.score.to_numpy()-inherited,0,1)
audit=align(V22/'results/reviews/final_reviewed.csv',te).reset_index()
threshold=json.loads((V22/'audit/protocol.json').read_text())['fixed_threshold']
def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None]-inter,1e-9)
def metric(rows,s):
    s=np.asarray(s);out={'pooled_ap':float(AP(rows.error,s))}
    for field,prefix in [('day','day_'),('class_name','ap_')]:
        for k in rows[field].unique():
            m=rows[field].eq(k).to_numpy();out[prefix+(k[-2:] if field=='day' else k)]=float(AP(rows.loc[m,'error'],s[m]))
    return out
def register(name,s,t,details):
    s=np.asarray(s,float);t=np.asarray(t,float)
    assert s.shape==(198,) and t.shape==(362,) and np.isfinite(s).all() and np.isfinite(t).all()
    assert ((s>=0)&(s<=1)).all() and ((t>=0)&(t<=1)).all()
    q.assign(score=s,historical_score=np.clip(s+inherited,0,1)).to_csv(A/f'results/oof/{name}.csv',index=False)
    path=A/f'results/submissions/submission_{name}.csv';te[['id']].assign(score=t).to_csv(path,index=False)
    d=audit.copy();d['baseline_v22']=current;d['new_score']=t;d['delta']=t-current
    good=d.visual_status.eq('plausible_correct');bad=d.visual_status.isin(['clear_error','likely_error'])
    d['diagnostic_pass']=(good&(t<threshold))|(bad&(t>=threshold))
    d.to_csv(A/f'results/reviews/{name}.csv',index=False)
    prior=d.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
    row=dict(experiment=name,**metric(q,s),historical=metric(q,np.clip(s+inherited,0,1)),
        changed_test=int((abs(t-current)>1e-9).sum()),changed_oof=int((abs(s-base_oof)>1e-9).sum()),
        visual_regressions_gt002=int(((good&(d.delta>.02))|(bad&(d.delta<-.02))).sum()),
        visual_improvements_gt002=int(((good&(d.delta<-.02))|(bad&(d.delta>.02))).sum()),
        remaining_known=int(((good|bad)&~d.diagnostic_pass).sum()),priority_passes=int((prior&d.diagnostic_pass).sum()),
        details=details,lb=None,submitted=False,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (A/f'results/{name}.json').write_text(json.dumps(row,indent=2))
    print('RESULT',name,round(row['pooled_ap'],7),row['changed_oof'],row['changed_test'],
        'remaining/regress',row['remaining_known'],row['visual_regressions_gt002'],flush=True)
    return row
