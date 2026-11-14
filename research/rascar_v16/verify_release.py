from common16 import *
import ast,importlib.metadata
from sklearn.metrics import average_precision_score

name='final_targeted_short_with_contradiction'
out=pd.read_csv(R/'submissions'/f'submission_{name}.csv')
assert list(out.columns)==['id','score'] and out.id.is_unique and out.id.tolist()==te.id.tolist()
assert len(out)==362 and out.score.between(0,1).all() and np.isfinite(out.score).all()
assert q.id.is_unique and len(q)==198 and set(q.day)==set(DAYS)
assert not set(q.frame_id)&set(te.frame_id)
records=pd.read_csv(V16/'dino_records.csv');X=np.load(V16/'dino_features.npy',mmap_mode='r')
assert records.id.is_unique and X.shape==(5336,2,1536) and np.isfinite(X).all()
learn=records[records.kind.ne('query_test')]
assert set(learn.frame_id)<=set(h.frame_id) and not set(learn.frame_id)&set(te.frame_id)
assert set(learn.day)==set(DAYS)
for fit,held in zip(DAYS,DAYS[::-1]):
    fit_ids=set(learn[learn.day.eq(fit)].frame_id);held_ids=set(q[q.day.eq(held)].frame_id)
    assert not fit_ids&held_ids
synthetic=records[~records.kind.isin(['query_train','query_test'])]
assert (synthetic.error.to_numpy()==(synthetic.target_iou.to_numpy()<.6)).all()
coverage={}
for task,n in [('heads',121),('hands',194)]:
    c=pd.read_csv(V16/f'wevis_{task}_coverage.csv')
    assert len(c)==n and c.id.is_unique and c.parsed.all() and not c.truncated.any()
    b=pd.read_csv(V16/f'wevis_{task}_references.csv')[B].to_numpy()
    assert (b[:,2:]>b[:,:2]).all() and (b>=0).all() and (b<=np.array([1280,720,1280,720])+1e-4).all()
    coverage[task]={'requests':n,'boxes':len(b),'parse_errors':0,'truncated':0}
for path in V16.glob('*.py'):ast.parse(path.read_text(),filename=str(path))

# Paired bootstrap by frame and stratified by held-out day. Descriptive only:
# this does not account for repeated model/threshold selection on these data.
new=pd.read_csv(R/'oof'/f'{name}.csv').set_index('id').loc[q.id,'score'].to_numpy()
without=pd.read_csv(R/'oof/final_targeted_short.csv').set_index('id').loc[q.id,'score'].to_numpy()
rng=np.random.default_rng(1701);deltas=[]
for _ in range(2000):
    ix=[]
    for day in DAYS:
        frame_ids=q[q.day.eq(day)].frame_id.unique()
        for fid in rng.choice(frame_ids,len(frame_ids),replace=True):ix.extend(np.flatnonzero(q.frame_id.eq(fid)))
    y=q.error.to_numpy()[ix];base=AP(y,q.score.to_numpy()[ix])
    deltas.append([AP(y,new[ix])-base,AP(y,without[ix])-base])
deltas=np.array(deltas)
result={'submission_rows':len(out),'train_query_rows':len(q),'train_query_frames':q.frame_id.nunique(),'test_frames':te.frame_id.nunique(),'feature_training_frames':learn.frame_id.nunique(),'feature_shape':list(X.shape),'cv_day_leakage_check':'passed','baseline_sha256':hashlib.sha256((V16/'baseline_submission_0.92822.csv').read_bytes()).hexdigest(),'inference_coverage':coverage,'bootstrap':{key:{'delta_mean':float(deltas[:,j].mean()),'percentile_95':np.quantile(deltas[:,j],[.025,.975]).tolist()} for j,key in enumerate(['full_rule_set','without_oof_only_contradiction'])},'bootstrap_caveat':'descriptive paired frame bootstrap; excludes model-selection uncertainty'}
(V16/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
