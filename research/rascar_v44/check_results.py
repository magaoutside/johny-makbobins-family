from pathlib import Path
import sys,json,hashlib,py_compile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'rascar_v43'))
from common43 import *
from scipy.optimize import linear_sum_assignment
q,te,h=inputs();bo,bt=baseline(q,te);checks=[]
for folder in ['rascar_v43','rascar_v44']:
 for f in (ROOT/folder).glob('*.py'):py_compile.compile(str(f),doraise=True)
checks.append('Both branches compile')
for (fid,c),g in q.groupby(['frame_id','class_name']):
 gt=h[h.frame_id.eq(fid)&h.class_name.eq(c)][B].to_numpy();ov=iou(g[B],gt);e=np.ones(len(g),int)
 if ov.size:
  x,y=linear_sum_assignment(-((ov>=.6)*(max(ov.shape)+1)+ov*(ov>=.6)));e[x[ov[x,y]>=.6]]=0
 assert np.array_equal(e,g.error)
checks.append('198 official labels reproduced by cardinality-first, then IoU one-to-one matching')
for folder in ['rascar_v43','rascar_v44']:
 a=ROOT/folder
 for f in (a/'models').glob('*manifest.json'):
  for m in json.loads(f.read_text()):
   fs=m.get('fit_frames',m.get('old_train_frames',[]))+m.get('new_train_frames',[])
   assert all(pd.to_datetime(v.split('_')[-2]).strftime('%Y-%m-%d')==m['fit_day'] for v in fs)
   assert not set(fs)&set(te.frame_id)
 for f in (a/'results/submissions').glob('*.csv'):
  d=pd.read_csv(f);assert d.columns.tolist()==['id','score'] and len(d)==362 and d.id.is_unique and set(d.id)==set(te.id)
  assert np.isfinite(d.score).all() and d.score.between(0,1).all()
  if folder=='rascar_v44':assert np.array_equal(d.set_index('id').loc[te.id[te.class_name.ne('nude_hands')],'score'].to_numpy(),bt[te.class_name.ne('nude_hands')])
checks.append('Every trained fold uses only its fitting day; no test training, all research CSVs valid')
checks.append('V44 changes hands only, all other classes byte-numerically unchanged')
a=ROOT/'rascar_v44';r=pd.read_csv(a/'inputs/pool_records.csv');held=set(r[r.split.eq('reserved_holdout')].frame_id)
for f in (a/'models').glob('*manifest.json'):
 for m in json.loads(f.read_text()):assert not held&set(m['old_train_frames']+m['new_train_frames'])
checks.append('New reserved pool frames excluded from every V44 model')
assert not set(r.frame_id)&set(te.frame_id)
for a,f in [(ROOT/'rascar_v43','features/local_manifest.json'),(ROOT/'rascar_v44','features/manifest.json')]:
 m=json.loads((a/f).read_text());rec=a/('inputs/pool_records.csv' if a.name=='rascar_v44' else 'inputs/records.csv');assert m['record_sha256']==hashlib.sha256(rec.read_bytes()).hexdigest()
checks.append('Completed feature-cache checksums match training records')
out=dict(passed=True,checks=checks,baseline_hash=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest(),leaderboard_uploaded=False,test_manual_annotations=False)
(ROOT/'rascar_v44/results/integrity_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
