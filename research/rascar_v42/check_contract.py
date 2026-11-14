from base42 import *
import py_compile
from scipy.optimize import linear_sum_assignment
q,te,h=inputs();bo,bt=baseline(q,te);checks=[]
for f in A.glob('*.py'):py_compile.compile(str(f),doraise=True)
checks.append('Python files compile')
r=pd.read_csv(A/'inputs/records.csv');assert r.id.is_unique and r[r.source.eq('test')].target.isna().all();assert set(r[r.source.eq('query')].id)==set(q.id)
for (fid,c),g in q.groupby(['frame_id','class_name']):
 gt=h[h.frame_id.eq(fid)&h.class_name.eq(c)][B].to_numpy();ov=iou(g[B],gt);err=np.ones(len(g),int)
 if ov.size:
  x,y=linear_sum_assignment(-((ov>=.6)*(max(ov.shape)+1)+ov*(ov>=.6)));err[x[ov[x,y]>=.6]]=0
 assert np.array_equal(err,g.error)
checks.append('All 198 official binary labels reproduced by one-to-one IoU>=.6 matching')
assert not set(r.loc[r.source.eq('pool'),'frame_id'])&set(te.frame_id)
for f in (A/'models').glob('*manifest*.json'):
 for m in json.loads(f.read_text()):assert all(pd.to_datetime(fid.split('_')[-2]).strftime('%Y-%m-%d')==m['fit_day'] for fid in m['train_frames'])
checks.append('All trained models use only their fit-day frames; no test labels')
for f in (A/'results').glob('*predictions_s*.csv'):
 d=pd.read_csv(f)
 for variant,g in d.groupby('variant'):
  o=g[g.source.eq('query')];t=g[g.source.eq('test')]
  assert len(o)==198 and o.id.is_unique and set(o.id)==set(q.id) and (o.fit_day!=o.day).all()
  assert len(t)==724 and set(t.id)==set(te.id) and np.isfinite(g.score).all()
checks.append('Every model has 198 cross-day OOF and 2 x 362 test predictions')
files=list((A/'results/submissions').glob('*.csv'))
for f in files:
 d=pd.read_csv(f);assert d.columns.tolist()==['id','score'] and len(d)==362 and d.id.is_unique and set(d.id)==set(te.id) and np.isfinite(d.score).all() and d.score.between(0,1).all()
checks.append(f'{len(files)} research CSV exports pass schema, ID and finite-score checks')
ex=json.loads((A/'features/extraction.json').read_text());assert ex['record_sha256']==hashlib.sha256((A/'inputs/records.csv').read_bytes()).hexdigest()
checks.append('Feature cache matches input-record checksum')
out=dict(passed=True,checks=checks,baseline_v31_sha256=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest(),leaderboard_uploaded=False)
(A/'results/integrity_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
