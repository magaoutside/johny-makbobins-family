from pathlib import Path
import sys,json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v41'))
from common import inputs,B,CB,CL
for sub in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/sub).mkdir(parents=True,exist_ok=True)
q,t,h=inputs();h=h[h.class_name.isin(CL)].copy();h['id']=['manual_'+str(i) for i in h.index];h['confidence']=1.
box=h[B].to_numpy();wh=np.maximum(box[:,2:]-box[:,:2],16)*2.4;cc=(box[:,2:]+box[:,:2])/2
h[CB]=np.c_[np.maximum(0,np.floor(cc-wh/2)),np.minimum([1280,720],np.ceil(cc+wh/2))]
h.to_csv(A/'inputs/manual_records.csv',index=False)
(A/'inputs/protocol.json').write_text(json.dumps(dict(hypothesis='Frozen OWLv2 object references differ systematically from organizer box conventions. Fit robust class-specific side offsets on manual GT from the opposite day.',manual_crops=len(h),fit='Median normalized GT-reference side offsets, confidence>=0.1, match IoU>=0.35, shrink n/(n+10), clip offsets to +/-0.4; no query error labels in geometry fit.',evaluation='25 -> 26 and 26 -> 25. Test two fit-day corrections must agree in score direction, choose smaller change. Repeated development OOF, not independent holdout.',scopes=CL,weights=[.1,.25],prohibited='No manual test targets, ID rules, pool/test used in fitting.'),indent=2))
print(h.groupby(['day','class_name']).size().to_string())
