from pathlib import Path
import sys, json, hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,fr,B,CL,DAYS,base_oof,current,audit,threshold,inherited,align,iou,metric,V22
A=ROOT/'rascar_v25'

def register(name,s,t,details):
    s=np.asarray(s,float);t=np.asarray(t,float)
    assert s.shape==(len(q),) and t.shape==(len(te),)
    assert np.isfinite(s).all() and np.isfinite(t).all() and min(s.min(),t.min())>=0 and max(s.max(),t.max())<=1
    q.assign(score=s,historical_score=np.clip(s+inherited,0,1)).to_csv(A/f'results/oof/{name}.csv',index=False)
    path=A/f'results/submissions/submission_{name}.csv';te[['id']].assign(score=t).to_csv(path,index=False)
    d=audit.copy();d['baseline_v22']=current;d['new_score']=t;d['delta']=t-current
    good=d.visual_status.eq('plausible_correct');bad=d.visual_status.isin(['clear_error','likely_error'])
    prior=d.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
    d['diagnostic_pass']=(good&(t<threshold))|(bad&(t>=threshold))
    d.to_csv(A/f'results/reviews/{name}.csv',index=False)
    r=dict(experiment=name,**metric(q,s),historical=metric(q,np.clip(s+inherited,0,1)),
        changed_test=int((abs(t-current)>1e-9).sum()),changed_oof=int((abs(s-base_oof)>1e-9).sum()),
        visual_regressions_gt002=int(((good&(d.delta>.02))|(bad&(d.delta<-.02))).sum()),
        visual_improvements_gt002=int(((good&(d.delta<-.02))|(bad&(d.delta>.02))).sum()),
        remaining_known=int(((good|bad)&~d.diagnostic_pass).sum()),priority_passes=int((prior&d.diagnostic_pass).sum()),
        details=details,lb=None,submitted=False,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (A/f'results/{name}.json').write_text(json.dumps(r,indent=2))
    print('RESULT',name,round(r['pooled_ap'],7),'changes',r['changed_test'],'remaining/regress',r['remaining_known'],r['visual_regressions_gt002'],flush=True)
    return r

def guard(split,rows,cls):
    z=align(V22/f'replay_inputs/{split}.csv',rows)
    yes=lambda s:s.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
    if cls=='nude_hands':
        veto=(z.independent_hand_centers.ge(2).to_numpy()|yes(z.old_duplicate_veto)|yes(z.zoom_duplicate_loser)|
             np.minimum(z.quality_logreg,z.quality_svm).ge(.75).to_numpy()|z.structural_hand.gt(z.baseline+1e-9).to_numpy())
        return ~veto & z.semantic_hand.ge(.8).to_numpy()
    if cls in ['cup','tray']:
        sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_'+cls)
        st=align(ROOT/f'rascar_v21/object_structure_{split}.csv',rows)
        return (sem>=.8)&st.distinct_instances.le(1).to_numpy()&~yes(st.ambiguous_duplicate)
    return np.ones(len(rows),bool)
