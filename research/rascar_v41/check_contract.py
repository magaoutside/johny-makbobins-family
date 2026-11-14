"""Data isolation and export integrity checks, not model-performance assertions."""
from common import *
import py_compile
q,te,h=inputs();bo,bt=baseline(q,te);checks=[]
for path in A.glob('*.py'):py_compile.compile(str(path),doraise=True)
checks.append('All top-level Python files compile')
rec=pd.read_csv(A/'inputs/records.csv');assert rec.id.is_unique
assert rec.loc[rec.split.eq('test'),['error','quality','presence']].isna().all().all()
review=pd.read_csv(A/'pool_expansion/visual_review.csv');approved=pd.read_csv(A/'pool_expansion/approved_geometry.csv')
assert not set(review.frame_id)&set(te.frame_id)
assert approved.decision.isin(['weak_geometry','class_absent']).all()
assert (review.decision.eq('weak_geometry').sum(),review.decision.eq('class_absent').sum())==(42,2)
checks.append('New visual labels are exclusively pool images; 42 weak boxes + 2 negatives')
for path in (A/'models').glob('*expanded*_manifest.json'):
 for m in json.loads(path.read_text()):
  assert not m['automatic_test_ids'] and m['manual_test_labels_used'] is False
  assert all(day(f)==m['fit_day'] for f in m['train_frames'])
checks.append('All inductive expanded-pool models use fit-day frames only')
for path in (A/'models').glob('*transductive_manifest.json'):
 for m in json.loads(path.read_text()):
  assert m['claims_oof'] is False and m['manual_test_labels_used'] is False
  assert set(m['automatic_test_ids'])<=set(te.id)
checks.append('Final test adaptation has no claimed OOF and no manual test labels')
for path in (A/'features').glob('*expanded*_predictions.csv'):
 d=pd.read_csv(path)
 for variant,g in d.groupby('variant'):
  o=g[g.split.eq('query')];assert len(o)==198 and o.id.is_unique and set(o.id)==set(q.id)
  assert (o.day!=o.fit_day).all()
  t=g[g.split.eq('test')];assert len(t)==724 and set(t.id)==set(te.id)
checks.append('Every expanded model has 198 cross-day predictions and 2 x 362 test predictions')
files=list((A/'results/submissions').glob('*.csv'))+list(A.glob('submission*.csv'))
for path in files:
 d=pd.read_csv(path);assert d.columns.tolist()==['id','score'] and len(d)==362 and d.id.is_unique
 assert set(d.id)==set(te.id) and np.isfinite(d.score).all() and d.score.between(0,1).all()
candidate=pd.read_csv(A/'submission_v41_pool_person_3seed.csv').set_index('id').loc[te.id].score.to_numpy()
assert np.allclose(candidate[te.class_name.ne('person')],bt[te.class_name.ne('person')],rtol=0,atol=1e-15)
checks.append(f'{len(files)} CSV exports have valid IDs, schema, finite scores and bounds; main candidate changes only person')
result=dict(passed=True,checks=checks,baseline_sha256=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest(),main_submission_sha256=hashlib.sha256((A/'submission_v41_pool_person_3seed.csv').read_bytes()).hexdigest())
(A/'results/integrity_checks.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
