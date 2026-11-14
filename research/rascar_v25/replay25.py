"""Portable ranking replay from released frozen CV features, not raw-image inference."""
from pathlib import Path
import sys,json,joblib,numpy as np,pandas as pd
A=Path(__file__).resolve().parent/'release';out=Path(sys.argv[1]) if len(sys.argv)>1 else A/'replayed';out.mkdir(exist_ok=True,parents=True)
DAYS=['2026-10-25','2026-10-26'];models={fit:joblib.load(A/f'models/pose_joint_catboost_{fit}.joblib') for fit in DAYS}
results={}
for split in ['oof','test']:
 d=pd.read_csv(A/f'{split}.csv');base=d.baseline.to_numpy();person=base+.9*(1-base)*d.person_strength.to_numpy()
 if split=='oof':
  f=pd.read_csv(A/'features/pose_joint_oof.csv').set_index('id').loc[d.id];p=np.zeros(len(d))
  for fit,held in zip(DAYS,DAYS[::-1]):
   mask=d.day.eq(held).to_numpy();p[mask]=models[fit].predict_proba(f.to_numpy()[mask])[:,1]
 else:p=np.mean([models[fit].predict_proba(pd.read_csv(A/f'features/pose_joint_test_{fit}.csv').set_index('id').loc[d.id].to_numpy())[:,1] for fit in DAYS],axis=0)
 gate=d.hand_gate.astype(str).str.lower().isin(['1','true']).to_numpy();hand=base+.5*gate*(np.minimum(p,base)-base)
 for tag,score in [('person',person),('hand',hand)]:
  expected=pd.read_csv(A/f'expected/{tag}_{split}.csv');err=float(abs(score-expected.score.to_numpy()).max());assert err<1e-12,(tag,split,err)
  d[['id']].assign(score=score).to_csv(out/f'{tag}_{split}.csv',index=False);results[f'{tag}_{split}']=dict(rows=len(d),max_absolute_error=err)
(out/'verification.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
