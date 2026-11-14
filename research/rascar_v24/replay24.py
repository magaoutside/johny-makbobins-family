"""Reproduce the saved cup candidate and separate hand research variant.

Inputs are frozen CV features and fitted heads, not raw-image detector inference.
Run boundary_pixels.py/evaluate_boundary.py to regenerate the new image heads.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib,json
A=Path(__file__).resolve().parent;R=A/'release';DAYS=['2026-10-25','2026-10-26']

def probability(expert,split,rows):
    predictions=[]
    for fit in DAYS:
        bundle=joblib.load(R/f'{expert}_{fit}.joblib')
        feature_path=R/(f'v23_{split}_{fit}.csv' if expert=='cup_context' else f'v24_{split}.csv')
        x=pd.read_csv(feature_path).set_index('id');x=x[bundle['columns']]
        raw=bundle['model'].predict_proba(x)[:,1]
        logit=np.log(np.clip(raw,1e-5,1-1e-5)/np.clip(1-raw,1e-5,1))[:,None]
        p=bundle['calibrator'].predict_proba(logit)[:,1]
        predictions.append(pd.Series(p,index=x.index).reindex(rows.id).to_numpy())
    if split=='oof':return np.where(rows.day.eq(DAYS[1]),predictions[0],predictions[1]),np.ones(len(rows),bool)
    return np.mean(predictions,axis=0),np.maximum(*predictions)<.5

def replay(split):
    rows=pd.read_csv(R/f'{split}_inputs.csv');base=rows.baseline.to_numpy();cups=[]
    for expert,weight in [('cup_context',.4),('cup_boundary',.5)]:
        p,agreement=probability(expert,split,rows)
        gate=rows.class_name.eq('cup').to_numpy()&rows.cup_guard.eq(1).to_numpy()&(p<.4)&agreement
        cups.append(base-weight*np.maximum(base-p,0)*gate)
    cup=np.minimum(*cups)
    hp,agreement=probability('hand_boundary',split,rows)
    gate=rows.class_name.eq('nude_hands').to_numpy()&rows.hand_guard.eq(1).to_numpy()&rows.geometry_guard.eq(1).to_numpy()&(hp<.4)&agreement
    research=cup-.5*np.maximum(base-hp,0)*gate
    return {'candidate':rows[['id']].assign(score=cup),'hand_research':rows[['id']].assign(score=research)}

if __name__=='__main__':
    checks={}
    for split in ['oof','test']:
        for variant,result in replay(split).items():
            expected=pd.read_csv(R/f'{variant}_{split}_expected.csv')
            assert result.id.equals(expected.id)
            difference=float(abs(result.score-expected.score).max());assert difference<1e-12
            result.to_csv(R/f'{variant}_{split}_replayed.csv',index=False)
            checks[f'{variant}_{split}']=dict(rows=len(result),max_abs_difference=difference)
    (A/'replay_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
