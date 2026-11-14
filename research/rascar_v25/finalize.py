from common25 import *
import shutil
from sklearn.metrics import average_precision_score
# Research hand candidate: keeps the best V22 as baseline, no person/cup/tray changes.
name='pose_joint_catboost_confirmed_rescue_w0.5'
shutil.copyfile(A/f'results/submissions/submission_{name}.csv',A/'submission_v25_pose_conservative_research.csv')
rr=[]
for path in (A/'results').glob('*.json'):
 z=json.loads(path.read_text())
 if isinstance(z,dict) and 'experiment' in z:rr.append(z)
flat=[{k:v for k,v in z.items() if k not in ['details','historical']} for z in rr]
pd.DataFrame(flat).sort_values('pooled_ap',ascending=False).to_csv(A/'results/experiment_registry.csv',index=False)
bench=[]
for model in ['rtmw','wilor_detector','rtmwx_bgr_padwide','rtmwx_rgb_padwide']:
 d=pd.read_csv(A/f'results/{model}_manual_detection.csv')
 for source,g in d.groupby('source'):
  bench.append(dict(model=model,source=source,n=len(g),mean_iou=g.best_iou.mean(),hits_iou06=int(g.best_iou.ge(.6).sum()),recall_iou06=g.best_iou.ge(.6).mean()))
pd.DataFrame(bench).to_csv(A/'results/pose_detection_benchmark.csv',index=False)
R=A/'release';R.mkdir(exist_ok=True)
for folder in ['models','features','expected']:(R/folder).mkdir(exist_ok=True)
for fit in DAYS:
 shutil.copyfile(A/f'models/pose_joint_catboost_{fit}.joblib',R/f'models/pose_joint_catboost_{fit}.joblib')
 shutil.copyfile(A/f'features/pose_joint_test_{fit}.csv',R/f'features/pose_joint_test_{fit}.csv')
shutil.copyfile(A/'features/pose_joint_oof.csv',R/'features/pose_joint_oof.csv')
for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
 f=align(A/f'features/person_fixed_tables_{split}.csv',rows)
 jf=align(A/f'features/pose_joint_{split}.csv',rows)
 gate=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')&jf.iou_max.ge(.6).to_numpy()&jf.iou_min.ge(.3).to_numpy()
 rows[['id','class_name']].assign(day=rows.day if 'day' in rows else '',baseline=base,person_strength=f.strength.to_numpy(),hand_gate=gate).to_csv(R/f'{split}.csv',index=False)
 for tag,exp in [('person','person_hidden_legs_fixed_tables'),('hand',name)]:
  score=align(A/f'results/{"oof" if split=="oof" else "submissions"}/{exp if split=="oof" else "submission_"+exp}.csv',rows,'score')
  rows[['id']].assign(score=score).to_csv(R/f'expected/{tag}_{split}.csv',index=False)
# Paired frame bootstrap describes score uncertainty but not search-selection bias.
s=align(A/f'results/oof/{name}.csv',q,'score');rng=np.random.default_rng(25);ids=q.frame_id.unique();deltas=[]
for _ in range(2000):
 ix=np.concatenate([np.flatnonzero(q.frame_id.eq(fid)) for fid in rng.choice(ids,len(ids),replace=True)])
 if q.error.iloc[ix].nunique()<2:continue
 deltas.append(average_precision_score(q.error.iloc[ix],s[ix])-average_precision_score(q.error.iloc[ix],base_oof[ix]))
summary=dict(baseline={'lb':.95610,**metric(q,base_oof)},person_probe=json.loads((A/'results/person_hidden_legs_fixed_tables.json').read_text()),hand_research=json.loads((A/f'results/{name}.json').read_text()),cup_simple_day_selected=json.loads((A/'results/cup_nested_day_selected.json').read_text()),cup_guarded_day_selected=json.loads((A/'results/cup_verified_day_selected.json').read_text()),experiments=len(rr),models=len(list((A/'models').glob('*.joblib'))),gpu_benchmark=bench,hand_gain_ci95=np.quantile(deltas,[.025,.975]).tolist(),hand_bootstrap_note='paired frame bootstrap, does not account for model/parameter search',new_lb=None,submitted=False)
(A/'summary.json').write_text(json.dumps(summary,indent=2))
# Every experiment has a complete, bounded submission with the original row order.
for p in (A/'results/submissions').glob('*.csv'):
 d=pd.read_csv(p);assert list(d.columns)==['id','score'];assert d.id.tolist()==te.id.tolist();assert d.id.is_unique and len(d)==362;assert np.isfinite(d.score).all() and d.score.between(0,1).all(),p
assert hashlib.sha256((V22/'submission_v22_candidate.csv').read_bytes()).hexdigest()=='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
checks=dict(experiment_submissions_checked=len(list((A/'results/submissions').glob('*.csv'))),baseline_v22_unchanged=True,main='No replacement of confirmed V22',person_other_classes_unchanged=bool(np.array_equal(align(A/'submission_v25_person_hidden_legs_test.csv',te,'score')[te.class_name.ne('person')],current[te.class_name.ne('person')])),hand_other_classes_unchanged=bool(np.array_equal(align(A/'submission_v25_pose_conservative_research.csv',te,'score')[te.class_name.ne('nude_hands')],current[te.class_name.ne('nude_hands')])))
(A/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks));print('SUMMARY',len(rr),summary['models'],summary['hand_gain_ci95'])
