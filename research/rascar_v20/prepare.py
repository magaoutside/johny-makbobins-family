"""Freeze V18, pool exclusion ledger, and metadata before V20 experiments."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd

A=Path(__file__).resolve().parent
ROOT=A.parent
for folder in ['inputs','results/oof','results/submissions','results/reviews','models','logs','audit','gpu_payload']:
    (A/folder).mkdir(parents=True,exist_ok=True)
src=ROOT/'rascar_v18/submission_v18_main.csv'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='27d4ce779b8d8d26e6df01c9eccef83f2a54ca99e19cbbe0099b2ca4271dd2bd'
shutil.copy2(src,A/'baseline_lb095259.csv')
shutil.copy2(ROOT/'rascar_v18/results/oof/release_main.csv',A/'inputs/baseline_oof.csv')
shutil.copy2(ROOT/'rascar_v18/audit_all_362_boxes.csv',A/'audit/frozen_test_audit.csv')
shutil.copy2(ROOT/'rascar_v18/audit_protocol.json',A/'audit/protocol.json')
for name in ['manual_boxes.csv','frames.csv','train_queries.csv','test_queries.csv']:
    shutil.copy2(ROOT/'rascar_v17/gpu_payload'/name,A/'inputs'/name)
fr=pd.read_csv(A/'inputs/frames.csv')
manual=pd.read_csv(A/'inputs/manual_boxes.csv')
te=pd.read_csv(A/'inputs/test_queries.csv')
used=fr[fr.frame_id.isin(set(manual.frame_id)|set(te.frame_id))].copy()
assert len(used)==170
used.to_csv(A/'inputs/required_frames.csv',index=False)
P=ROOT/'rascar_v15/previous/rascar_v14/stage2'
mf=pd.read_csv(P/'machine_frames.csv'); mp=pd.read_csv(P/'machine_predictions.csv')
mf['day']=mf.timestamp.str[:10]
stamps=pd.to_datetime(mf.timestamp,utc=True,format='mixed').astype('int64').to_numpy()
gold=pd.to_datetime(fr[fr.split.eq('train')].timestamp,utc=True,format='mixed').astype('int64').to_numpy()
mf['nearest_annotated_seconds']=np.min(abs(stamps[:,None]-gold[None,:]),axis=1)/1e9
mf['eligible_time']=mf.nearest_annotated_seconds.gt(600)&~mf.frame_id.isin(used.frame_id)
mf['hour']=pd.to_datetime(mf.timestamp,utc=True,format='mixed').dt.hour
mf['scored_proposals']=mf.frame_id.map(mp[mp.class_name.ne('table')].groupby('frame_id').size()).fillna(0).astype(int)
mf.to_csv(A/'inputs/pool_frame_ledger.csv',index=False)
mp.to_csv(A/'inputs/machine_predictions.csv',index=False)
# A deterministic first batch, balanced over day/hour and containing scored
# objects. This is a pilot; quality and teacher agreement are checked later.
candidate=mf[mf.eligible_time & mf.scored_proposals.gt(0)].copy()
candidate['selection_key']=candidate.frame_id.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
sample=candidate.sort_values('selection_key').groupby(['day','hour'],sort=True).head(10).sort_values('timestamp')
sample.to_csv(A/'inputs/pool_pilot_frames.csv',index=False)
(A/'plan.json').write_text(json.dumps(dict(
    baseline_lb=.95259,baseline_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    abandoned_experiment='V19; no V19 score modifications included',
    scope=['pose/seat person modes','high precision machine-pool pseudo labels','mask-conditioned crop quality model','YOLO26 fine tuning and sliced inference','cup/tray/high-score valid person audit'],
    folds={'fit_25':'evaluate_26','fit_26':'evaluate_25'},
    pool_days=sorted(mf.day.unique()),pool_frames=len(mf),pool_proposals=len(mp),
    time_exclusion_seconds=600,pool_eligible_frames=int(mf.eligible_time.sum()),pilot_frames=len(sample),
    pseudo_labels_are_official_gt=False,test_visual_labels_used_for_fitting=False,
    keep_all_experiments=True,submit_to_kaggle_automatically=False,
    selection_note='Repeated development OOF plus subjective test diagnostics; no claim of independent hidden-test validation.'
),indent=2))
print('PREPARED',len(sample),'pilot frames',sample.day.value_counts().to_dict(),'eligible',int(mf.eligible_time.sum()))
