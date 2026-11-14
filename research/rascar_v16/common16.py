from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
V16=Path(__file__).resolve().parent;V15=V16/'inputs/v15' if (V16/'inputs/v15').is_dir() else V16.parent/'rascar_v15'
sys.path.insert(0,str(V15))
import common as old
B,CL,DAYS=old.B,old.CL,old.DAYS
q,te,h,fr=old.q.copy(),old.te.copy(),old.h.copy(),old.fr.copy()
iou=old.iou
q['score']=pd.read_csv(V15/'results/oof/final_hands0_pose0.2.csv').set_index('id').loc[q.id,'score'].to_numpy()
current=pd.read_csv(V16/'baseline_submission_0.92822.csv').set_index('id').loc[te.id,'score'].to_numpy()
diagnostic=pd.read_csv(V15/'test_diagnostic_frozen.csv').set_index('id').loc[te.id].reset_index()
diagnostic['v14_score']=diagnostic.score;diagnostic['score']=current
R=V16/'results';R.mkdir(exist_ok=True)
for folder in ['oof','submissions','reviews']:(R/folder).mkdir(exist_ok=True)
def metric(rows,s):
    out={'pooled_ap':float(AP(rows.error,s))}
    for d,g in rows.groupby('day'):out['day_'+d[-2:]]=float(AP(g.error,np.asarray(s)[g.index]))
    for c,g in rows.groupby('class_name'):out['ap_'+c]=float(AP(g.error,np.asarray(s)[g.index]))
    return out
def register(name,s,t,details=None):
    s=np.asarray(s,float);t=np.asarray(t,float)
    assert s.shape==(198,) and t.shape==(362,) and np.isfinite(s).all() and np.isfinite(t).all()
    assert ((s>=0)&(s<=1)).all() and ((t>=0)&(t<=1)).all()
    q.assign(score=s).to_csv(R/'oof'/f'{name}.csv',index=False)
    fn=R/'submissions'/f'submission_{name}.csv';te[['id']].assign(score=t).to_csv(fn,index=False)
    row=dict(experiment=name,**metric(q,s),details=details or {},sha256=hashlib.sha256(fn.read_bytes()).hexdigest(),public_lb=None)
    d=diagnostic.copy();d['new_score']=t;d['delta']=t-current
    m=d.visual_status.isin(['clear_error','plausible_correct'])&~d.revised_after_scores
    row['test_visual_diagnostic_ap_NOT_validation']=float(AP(d.loc[m,'visual_status'].eq('clear_error'),t[m]))
    row['test_regressions_gt002']=int(((d.visual_status.isin(['clear_error','likely_error'])&(d.delta<-.02))|(d.visual_status.eq('plausible_correct')&(d.delta>.02))).sum())
    for label,mask in [('short_people',d.issue.eq('person_torso_only')),('high_plausible_hands',d.class_name.eq('nude_hands')&d.visual_status.eq('plausible_correct')&(d.score>=.8))]:
        row[label+'_mean_delta']=float(d.loc[mask,'delta'].mean())
    d.to_csv(R/'reviews'/f'{name}.csv',index=False)
    (R/f'{name}.json').write_text(json.dumps(row,indent=2))
    print('RESULT',name,*(round(row[k],6) for k in ['pooled_ap','day_25','day_26','test_visual_diagnostic_ap_NOT_validation']),flush=True)
    return row
