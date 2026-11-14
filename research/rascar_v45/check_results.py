from common45 import *
from sklearn.metrics import average_precision_score as AP
import py_compile,itertools
NAME='nested_cross_class_rank_person_w10_scale_preserved'
q,te,h=inputs();bo,bt=baseline(q,te)
for p in A.glob('*.py'):py_compile.compile(str(p),doraise=True)
paths=sorted((A/'results/submissions').glob('*.csv'))
sample=pd.read_csv(D/'sample_submission.csv')
for p in paths:
    d=pd.read_csv(p);assert list(d.columns)==['id','score'];assert len(d)==362 and d.id.is_unique and set(d.id)==set(sample.id);assert np.isfinite(d.score).all() and d.score.between(0,1).all()
f=pd.read_csv(A/'features/nested_quality.csv');assert np.isfinite(f.drop(columns=['error','overlap']).select_dtypes('number')).all().all()
for fit in DAYS:
    tr=f[f.outer_fit_day.eq(fit)&f.role.eq('meta_train')];ev=f[f.outer_fit_day.eq(fit)&f.role.eq('oof')]
    assert tr.day.eq(fit).all() and ev.day.ne(fit).all();assert set(tr.id)==set(q[q.day.eq(fit)].id);assert set(ev.id)==set(q[q.day.ne(fit)].id)
    assert not set(tr.frame_id)&set(ev.frame_id);assert not set(tr.frame_id)&set(te.frame_id)
for entry in json.loads((A/'models/nested_manifest.json').read_text()):
    fit,held=set(entry['fit_frames']),set(entry['held_frames']);assert not fit&held
    assert all(s[-15:-7].endswith(entry['outer_fit_day'][-2:]) for s in fit|held)
    times=lambda seq:np.array([pd.to_datetime(s[-15:],format='%Y%m%d_%H%M%S').timestamp() for s in seq])
    assert np.abs(times(fit)[:,None]-times(held)[None]).min()>120
for m in json.loads((A/'models/affinity_manifest.json').read_text()):
    assert all(x[-15:-7].endswith(m['fit_day'][-2:]) for x in m['fit_frame_ids'])
oo=pd.read_csv(A/f'results/oof/{NAME}.csv').set_index('id').loc[q.id,'score'].to_numpy();tt=pd.read_csv(A/f'results/submissions/{NAME}.csv').set_index('id').loc[te.id,'score'].to_numpy()
for rows,base,new in [(q,bo,oo),(te,bt,tt)]:
    person=rows.class_name.eq('person').to_numpy();assert np.array_equal(base[~person],new[~person]);assert np.array_equal(np.sort(base[person]),np.sort(new[person]))
# Exact raw official labels using cardinality-first, then IoU matching.
from scipy.optimize import linear_sum_assignment
reproduced=[]
for (_,cls),group in q.groupby(['frame_id','class_name']):
    gt=h[h.frame_id.eq(group.frame_id.iloc[0])&h.class_name.eq(cls)][B].to_numpy(float);ov=iou(group[B],gt);valid=ov>=.6;cost=np.where(valid,1000+ov,0.)
    matches=linear_sum_assignment(-cost) if cost.shape[1] else ([],[]);correct={group.index[i] for i,j in zip(*matches) if valid[i,j]}
    reproduced.extend([(i,int(i not in correct)) for i in group.index])
assert np.array_equal(pd.Series(dict(reproduced)).sort_index().to_numpy(),q.error.to_numpy())
# Additional conditional sensitivity; model selection uncertainty is not covered.
times=pd.to_datetime(q.frame_id.str.extract(r'(\d{8}_\d{6})$')[0],format='%Y%m%d_%H%M%S');block=q.day+'_'+(times.dt.hour//2).astype(str);blocks=sorted(block.unique());leave=[]
for key in blocks:
    mask=block.ne(key);leave.append(dict(excluded=key,frames=int(q[~mask].frame_id.nunique()),delta=float(AP(q.error[mask],oo[mask])-AP(q.error[mask],bo[mask]))))
pd.DataFrame(leave).to_csv(A/'audit/leave_two_hour_block.csv',index=False)
rng=np.random.default_rng(4504);deltas=[]
for _ in range(3000):
    idx=[]
    for day in DAYS:
        keys=sorted(block[q.day.eq(day)].unique());chosen=rng.choice(keys,len(keys),replace=True)
        idx.extend(np.concatenate([np.flatnonzero(block.eq(key)) for key in chosen]))
    idx=np.array(idx);deltas.append(AP(q.error.iloc[idx],oo[idx])-AP(q.error.iloc[idx],bo[idx]))
stats=dict(official_label_matches=len(q),validated_submission_files=len(paths),cross_day_and_inner_purge_verified=True,non_person_scores_unchanged=True,person_score_multiset_unchanged=True,baseline_sha256=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest(),candidate_ap=float(AP(q.error,oo)),candidate_sha256=hashlib.sha256((A/f'results/submissions/{NAME}.csv').read_bytes()).hexdigest(),test_changes=int((abs(tt-bt)>1e-10).sum()),block_bootstrap_ci95=np.quantile(deltas,[.025,.975]).tolist(),leave_two_hour_block_min=min(x['delta'] for x in leave),warning='Repeated development OOF; conditional bootstrap, not independent evidence of public gain; no test GT; quantile transfer is transductive within each population')
(A/'results/integrity_checks.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
