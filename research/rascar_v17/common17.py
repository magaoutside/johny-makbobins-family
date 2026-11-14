from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
A=Path(__file__).resolve().parent;V16=A.parent/'rascar_v16';V15=A.parent/'rascar_v15'
sys.path.insert(0,str(V16))
import common16 as prev
V15=prev.V15  # use the immutable, vendored V15 inputs already retained with V16
B,CL,DAYS=prev.B,prev.CL,prev.DAYS
q,te,h,fr=prev.q.copy(),prev.te.copy(),prev.h.copy(),prev.fr.copy()
iou=prev.iou
baseline15=q.score.to_numpy().copy();test15=prev.current.copy()
q['score']=pd.read_csv(V16/'results/oof/final_targeted_short_with_contradiction.csv').set_index('id').loc[q.id,'score'].to_numpy()
current=pd.read_csv(V16/'submission_v16_targeted_short.csv').set_index('id').loc[te.id,'score'].to_numpy()
audit=prev.diagnostic.copy();audit['v15_score']=test15;audit['v16_score']=current
R=A/'results';R.mkdir(exist_ok=True)
for folder in ['oof','submissions','reviews']:(R/folder).mkdir(exist_ok=True)
def align(path,rows,cols=None):
    df=pd.read_csv(path).set_index('id').loc[rows.id]
    return df if cols is None else df[cols].to_numpy()
def metric(rows,s):
    s=np.asarray(s);out={'pooled_ap':float(AP(rows.error,s))}
    for d,g in rows.groupby('day'):out['day_'+d[-2:]]=float(AP(g.error,s[g.index]))
    for c,g in rows.groupby('class_name'):out['ap_'+c]=float(AP(g.error,s[g.index]))
    return out
def register(name,s,t,details=None):
    s=np.asarray(s,float);t=np.asarray(t,float)
    assert s.shape==(198,) and t.shape==(362,) and np.isfinite(s).all() and np.isfinite(t).all()
    assert ((s>=0)&(s<=1)).all() and ((t>=0)&(t<=1)).all()
    q.assign(score=s).to_csv(R/'oof'/f'{name}.csv',index=False)
    path=R/'submissions'/f'submission_{name}.csv';te[['id']].assign(score=t).to_csv(path,index=False)
    d=audit.copy();d['score']=current;d['new_score']=t;d['delta']=t-current;d['delta_vs_v15']=t-test15
    m=d.visual_status.isin(['clear_error','plausible_correct'])&~d.revised_after_scores
    bad=d.visual_status.isin(['clear_error','likely_error']);good=d.visual_status.eq('plausible_correct')
    prior=d.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
    threshold=json.loads((A/'audit_protocol.json').read_text())['fixed_threshold']
    directional=np.where(bad,t-test15,np.where(good,test15-t,0))
    passed=(bad&(t>=threshold))|(good&(t<threshold))
    row=dict(experiment=name,**metric(q,s),details=details or {},public_lb=None,
        test_visual_ap_NOT_validation=float(AP(d.loc[m,'visual_status'].eq('clear_error'),t[m])),
        test_regressions_gt002_vs_v16=int(((bad&(d.delta<-.02))|(good&(d.delta>.02))).sum()),
        priority_significantly_improved_vs_v15=int((prior&(directional>.02)).sum()),
        priority_diagnostic_passes=int((prior&passed).sum()),priority_total=int(prior.sum()),
        changed_test_rows=int((abs(t-current)>1e-9).sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    d['diagnostic_pass']=passed;d.to_csv(R/'reviews'/f'{name}.csv',index=False)
    (R/f'{name}.json').write_text(json.dumps(row,indent=2))
    print('RESULT',name,*(round(row[k],6) for k in ['pooled_ap','day_25','day_26','test_visual_ap_NOT_validation']),
          'priority improved/pass',row['priority_significantly_improved_vs_v15'],row['priority_diagnostic_passes'],
          'regressions',row['test_regressions_gt002_vs_v16'],flush=True)
    return row
