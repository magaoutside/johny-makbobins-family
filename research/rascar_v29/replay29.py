"""Reproduce the two UNPROMOTED diagnostic CSVs from frozen CV features.

This is score replay, not raw-image SAM inference. Refit the empirical geometry
with geometry_experiment.py in the original workspace for the full experiment.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
A=Path(__file__).resolve().parent

def replay(rows,f,weight=.5,penalty=False,test=False):
    base=rows.baseline.to_numpy();delta=f.p_error.to_numpy()-base
    gate=rows.class_name.eq('cup').to_numpy()&rows.semantic_cup.ge(.8).to_numpy()&f.reference_confidence.ge(.65).to_numpy()&f.raw_iou.ge(.2).to_numpy()
    if penalty:
        gate&=f.p_error.ge(.65).to_numpy()
        if test:gate&=f.fold_min.ge(.65).to_numpy()
        delta=np.maximum(delta,0)
    return rows[['id']].assign(score=np.clip(base+weight*delta*gate,0,1))

if __name__=='__main__':
    checks={}
    for split in ['oof','test']:
        rows=pd.read_csv(A/f'inputs/replay_{split}.csv');f=pd.read_csv(A/f'features/one_to_one_{split}.csv')
        assert rows.id.equals(f.id)
        for penalty in [True,False]:
            result=replay(rows,f,penalty=penalty,test=split=='test');name='penalty' if penalty else 'both'
            path=A/(f'results/submissions/submission_cup_uncertainty_one_to_one_{name}_w0.5.csv' if split=='test' else f'results/oof/cup_uncertainty_one_to_one_{name}_w0.5.csv')
            expected=pd.read_csv(path);assert result.id.equals(expected.id)
            diff=float(abs(result.score-expected.score).max());assert diff<1e-12
            checks[f'{split}_{name}']=dict(rows=len(result),max_absolute_difference=diff)
    (A/'replay_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
