from common import *
from scipy.optimize import linear_sum_assignment
from html.parser import HTMLParser

name=(A/'selected_experiment.txt').read_text().strip()
candidate=pd.read_csv(A/'submission_v15.csv')
assert list(candidate.columns)==['id','score'] and candidate.id.is_unique
assert candidate.id.tolist()==te.id.tolist()
assert candidate.score.between(0,1).all() and np.isfinite(candidate.score).all()
for key,path in [('baseline_current_sha256',P/'submission_catboost_person_hands0.5_yolo_cup_tray20.csv'),('baseline_v8_sha256',P/'submission.csv')]:
    expected=json.loads((A/'validation_integrity.json').read_text())[key]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected

recomputed=np.ones(len(q),dtype=int)
for (fid,c),g in q.groupby(['frame_id','class_name']):
    gt=h[(h.frame_id==fid)&(h.class_name==c)]
    if gt.empty: continue
    ov=iou(g[B],gt[B]);reward=np.c_[np.where(ov>=.6,min(ov.shape)+1+ov,0),np.zeros((len(g),len(g)))]
    ii,jj=linear_sum_assignment(-reward)
    for x,y in zip(ii,jj):
        if y<len(gt) and ov[x,y]>=.6:recomputed[g.index[x]]=0
assert np.array_equal(recomputed,q.error.to_numpy())

for fit,filename in [('2026-10-25','hand25'),('2026-10-26','hand26'),('all','handall')]:
    ledger=pd.read_csv(A/'hand_specialist'/fit/'training_ledger.csv')
    assert set(ledger.frame_id)<=set(h.frame_id)
    if fit!='all':assert set(ledger.day)=={fit}
    assert not set(ledger.frame_id)&set(te.frame_id)
    history=pd.read_csv(A/'hand_specialist'/fit/'train/results.csv')
    assert len(history)==35
    manifest=json.loads((A/'remote_manifest.json').read_text())
    assert hashlib.sha256((A/'weights'/f'{filename}_last.pt').read_bytes()).hexdigest()==manifest['weights'][fit]

oof=pd.read_csv(A/'results/oof'/f'{name}.csv').set_index('id').loc[q.id,'score'].to_numpy()
assert abs(metric(q,oof)['pooled_ap']-json.loads((A/'results'/f'{name}.json').read_text())['pooled_ap'])<1e-12
rng=np.random.default_rng(17);deltas=[]
# Frame bootstrap within each day; exploratory uncertainty, not a fresh holdout.
for _ in range(2000):
    ids=[]
    for day in DAYS:
        frames=q.loc[q.day.eq(day),'frame_id'].unique()
        for fid in rng.choice(frames,len(frames),replace=True):ids.extend(q.index[q.frame_id.eq(fid)])
    y=q.error.to_numpy()[ids]
    if len(np.unique(y))<2:continue
    deltas.append(float(AP(y,oof[ids])-AP(y,q.score.to_numpy()[ids])))
out=dict(verified=True,submission_rows=len(candidate),official_labels_match=len(q),
         training_folds_day_isolated=True,weight_hashes_verified=True,epochs_each=35,
         selected=name,sha256=hashlib.sha256((A/'submission_v15.csv').read_bytes()).hexdigest(),
         frame_bootstrap_delta_ci95=np.quantile(deltas,[.025,.975]).tolist(),bootstrap_iterations=len(deltas),
         uncertainty_note='Exploratory: candidates selected using these days and test visuals; CI does not correct selection bias.')
(A/'final_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
