from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score as AP
A=Path(__file__).resolve().parent;V18=A.parent/'rascar_v18';V17=A.parent/'rascar_v17';V16=A.parent/'rascar_v16'
sys.path.insert(0,str(V18));import common18 as old
V15=old.V15;B,CL,DAYS=old.B,old.CL,old.DAYS;iou=old.iou;align=old.align
q,te,h,fr=old.q.copy(),old.te.copy(),old.h.copy(),old.fr.copy()
q['score']=align(V18/'results/oof/release_main.csv',q,'score')
current=align(A/'baseline_lb095259.csv',te,'score');audit=old.audit.copy();audit['v18_score']=current
R=A/'results';R.mkdir(exist_ok=True)
for f in ['oof','submissions','reviews']:(R/f).mkdir(exist_ok=True)
threshold=json.loads((A/'audit_protocol.json').read_text())['fixed_threshold']
def boolean(values):
    """CSV columns can mix True/False with 0; never use truthiness of strings."""
    return pd.Series(values).astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
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
    path=R/'submissions'/f'submission_{name}.csv';te[['id']].assign(score=t).to_csv(path,index=False)
    d=audit.copy();d['score']=current;d['new_score']=t;d['delta']=t-current;d['delta_vs_v15']=t-old.old.test15
    bad=d.visual_status.isin(['clear_error','likely_error']);good=d.visual_status.eq('plausible_correct')
    prior=d.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
    directional=np.where(bad,d.delta,np.where(good,-d.delta,0));direction15=np.where(bad,d.delta_vs_v15,np.where(good,-d.delta_vs_v15,0))
    passed=(bad&(t>=threshold))|(good&(t<threshold));m=d.visual_status.isin(['clear_error','plausible_correct'])&~d.revised_after_scores
    row=dict(experiment=name,**metric(q,s),details=details or {},public_lb=None,
        test_visual_ap_NOT_validation=float(AP(d.loc[m,'visual_status'].eq('clear_error'),t[m])),
        test_regressions_gt002_vs_v18=int(((bad&(d.delta<-.02))|(good&(d.delta>.02))).sum()),
        priority_improved_vs_v18=int((prior&(directional>.02)).sum()),priority_improved_vs_v15=int((prior&(direction15>.02)).sum()),
        priority_diagnostic_passes=int((prior&passed).sum()),priority_total=int(prior.sum()),changed_test_rows=int((abs(t-current)>1e-9).sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    d['diagnostic_pass']=passed;d.to_csv(R/'reviews'/f'{name}.csv',index=False);(R/f'{name}.json').write_text(json.dumps(row,indent=2))
    print('RESULT',name,*(round(row[k],6) for k in ['pooled_ap','day_25','day_26','test_visual_ap_NOT_validation']),'new/pass/regress',row['priority_improved_vs_v18'],row['priority_diagnostic_passes'],row['test_regressions_gt002_vs_v18'],flush=True)
    return row
def duplicate_veto(rows,test):
    split='test' if test else 'oof';g=pd.read_csv(V16/f'hand_rescue_group_guard_{split}.csv').set_index('id').reindex(rows.id)
    names=[V15/'hand_group_test.csv'] if test else [V15/f'hand_group_oof_{d}.csv' for d in DAYS]
    losers=set(pd.concat([pd.read_csv(p) for p in names]).query("reason=='duplicate_loser'").id)
    return rows.id.isin(losers).to_numpy()|g.veto.fillna(False).to_numpy(bool)
