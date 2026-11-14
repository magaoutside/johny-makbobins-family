from pathlib import Path
import sys,json,hashlib,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,metrics,baseline,iou,B,CB,CL,DAYS,D
sys.path.pop(0)
for p in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/p).mkdir(parents=True,exist_ok=True)

def incumbent(q,t):
    p=ROOT/'rascar_v52/release/submission_v52_candidate_purged_a20.csv'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='d30da6f6f938cba285c01a6e8ddb4fe8d095f38e0ea65ec38a2aa87091705ab5'
    o=pd.read_csv(ROOT/'rascar_v52/results/oof/purged_ensemble_d2_person_a20.csv').set_index('id').loc[q.id,'score'].to_numpy()
    return o,pd.read_csv(p).set_index('id').loc[t.id,'score'].to_numpy()
