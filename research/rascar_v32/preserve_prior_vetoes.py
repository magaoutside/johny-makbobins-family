"""Restore prior duplicate/structure vetoes when composing new geometry heads.

A high geometry match alone cannot invalidate an existing multi-object error.
No ID exceptions; the same old V22 guard is used on OOF and test.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';sys.path.insert(0,str(A))
from consensus import save,q,te,base_oof,current,align,CL
from common25 import guard
import numpy as np,pandas as pd

def main():
    results=[];ledger=[]
    for watershed in [0,1]:
        parent=f'compose_watershed{watershed}_confirmed_hand_protection2';outs=[]
        for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
            p=A/f'results/{"oof" if split=="oof" else "submissions"}/{"" if split=="oof" else "submission_"}{parent}.csv'
            s=align(p,rows).score.to_numpy().copy()
            for cls in CL:
                ix=np.flatnonzero(rows.class_name.eq(cls).to_numpy());g=guard(split,rows.iloc[ix],cls)
                blocked=ix[(s[ix]<base[ix]-1e-9)&~g]
                for i in blocked:ledger.append(dict(parent=parent,split=split,id=rows.iloc[i].id,class_name=cls,unprotected_score=float(s[i]),restored_score=float(base[i])))
                s[blocked]=base[blocked]
            outs.append(s)
        results.append(save(f'compose_watershed{watershed}_prior_vetoes_preserved',*outs))
    pd.DataFrame(results).to_csv(A/'results/prior_veto_experiments.csv',index=False);pd.DataFrame(ledger).to_csv(A/'audit/prior_veto_restorations.csv',index=False)
    (A/'audit/prior_veto_protocol.json').write_text(json.dumps(dict(reason='New geometry alone must not reverse prior duplicate/structure flags.',
        guard_source='rascar_v25/common25.py guard (V22 replay inputs)',new_fitting=False,id_exceptions=False,
        validation_status='Post-hoc development on the existing OOF; not independent confirmation.'),indent=2))
    print(pd.DataFrame(results).to_string(index=False))
if __name__=='__main__':main()
