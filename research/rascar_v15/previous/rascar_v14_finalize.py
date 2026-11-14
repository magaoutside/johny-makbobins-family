
from pathlib import Path
import json,hashlib,sys,zipfile,shutil
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
E=Path('/kaggle/working/rascar_v14');O=Path('/kaggle/working/oof_validation_v11');F=E/'final_audit';F.mkdir(exist_ok=True)
base=pd.read_csv(E/'stage1/oof/v8_baseline.csv');sample=pd.read_csv(O/'sample_submission.csv');metricpaths=[E/'stage1/experiment_metrics.csv',E/'stage2/crop_experiment_metrics.csv',E/'stage3/spatial_experiment_metrics.csv',E/'stage4/seat_experiment_metrics.csv',E/'stage5/pool_experiment_metrics.csv'];tables=[]
for p in metricpaths:
 if p.exists():tables.append(pd.read_csv(p).assign(source=str(p.relative_to(E))))
if (E/'stage2/portfolio.json').exists():tables.append(pd.DataFrame(json.loads((E/'stage2/portfolio.json').read_text())).assign(source='stage2/portfolio.json'))
metrics=pd.concat(tables,ignore_index=True);metrics.to_csv(F/'all_experiment_metrics.csv',index=False)
# Validate every exported candidate, preserve identical alternatives as separate trials.
files=list(Path('/kaggle/working').glob('submission_*.csv'))+list(E.glob('stage*/submission_*.csv'))+list((E/'stage1/candidates').glob('submission_*.csv'));manifest=[]
for p in files:
 t=pd.read_csv(p);assert list(t.columns)==['id','score'];assert len(t)==362 and t.id.is_unique and set(t.id)==set(sample.id);assert np.isfinite(t.score).all() and t.score.between(0,1).all();manifest.append(dict(file=str(p.relative_to('/kaggle/working')),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=len(t),public_lb=None))
(F/'submission_manifest.json').write_text(json.dumps(manifest,indent=2));assert hashlib.sha256(Path('/kaggle/working/submission.csv').read_bytes()).hexdigest()=='45cafde407eb41bfd061f4418681111bea82c1942da5c07844004351514ec883'

from sklearn.base import clone
import joblib
X=pd.read_csv(E/'stage1/raw_frozen_train_features.csv').set_index('id').loc[base.id];foldchecks=[]
for fit,held in [('2026-10-25','2026-10-26'),('2026-10-26','2026-10-25')]:
 a=base.day.eq(fit).to_numpy();b=base.day.eq(held).to_numpy();old=joblib.load(E/f'stage1/models/catboost_d2_fit_{fit}.joblib');poison=base.error.to_numpy().copy();poison[b]=1-poison[b];again=clone(old);again.fit(X[a],poison[a]);delta=float(np.max(abs(old.predict_proba(X[b])-again.predict_proba(X[b]))));assert delta<1e-12;foldchecks.append(dict(fit_day=fit,held_day=held,held_labels_flipped=True,maximum_prediction_change=delta,fit_frames=int(base.loc[a,'frame_id'].nunique()),held_frames=int(base.loc[b,'frame_id'].nunique())))
(F/'label_leakage_checks.json').write_text(json.dumps(foldchecks,indent=2));print('LABEL_LEAKAGE_CHECKS',json.dumps(foldchecks),flush=True)

# Paired, day-stratified frame bootstrap; descriptive uncertainty after model search.
selected=['catboost_d2_blend0.5','yolo_cup_tray_w0.2_conf0.25','person_edge_down_0.25','catboost_person_hands0.5_yolo_cup_tray20'];ci=[];rng=np.random.default_rng(17);draws=[]
for _ in range(2000):
 inds=[]
 for day,g in base.groupby('day'):
  for fid in rng.choice(g.frame_id.unique(),size=g.frame_id.nunique(),replace=True):inds.extend(g.index[g.frame_id.eq(fid)].tolist())
 draws.append(np.array(inds))
for name in selected:
 matches=list(E.glob('stage*/'+name+'_oof.csv'))+list((E/'stage1/oof').glob(name+'.csv'))
 if not matches:continue
 z=pd.read_csv(matches[0]);ss=z.set_index('id').loc[base.id,'score'].to_numpy();delta=[]
 for ix in draws:
  if base.error.iloc[ix].nunique()<2:continue
  delta.append(average_precision_score(base.error.iloc[ix],ss[ix])-average_precision_score(base.error.iloc[ix],base.score.iloc[ix]))
 ci.append(dict(experiment=name,delta=float(average_precision_score(base.error,ss)-average_precision_score(base.error,base.score)),lo=float(np.quantile(delta,.025)),hi=float(np.quantile(delta,.975)),positive_fraction=float(np.mean(np.array(delta)>0)),warning='Exploratory selected candidates; interval does not account for searching many models or unseen-day shift'))
pd.DataFrame(ci).to_csv(F/'paired_frame_bootstrap.csv',index=False)
# Preserve all code including failed attempts in older immutable Kaggle versions.
versions={'v1':368908411,'v2':368909344,'v3':368910883,'v4':368913716,'v5':368914502,'v6':368915968,'v7':368918080,'v8':368919659};(F/'historical_versions.json').write_text(json.dumps(versions,indent=2));
lines=['# Rascar V14 experiments','', 'Best verified public LB remains 0.90481. All new candidates: public LB unknown, not submitted.','', 'Validation: train 2026-10-25 predict 2026-10-26 and reverse; pooled AP plus each day. 198 scored inputs, 35 frames, only 2 development days. Tables are context and are excluded from ranking labels. Query boxes are unchanged.','', 'All hyperparameter trials, negative results and candidate CSVs are retained. Selecting candidates on these same two days makes the maximum score optimistic; this is not a guarantee of LB improvement.','', '| Experiment | OOF AP | Day 25 | Day 26 |','|---|---:|---:|---:|']
for r in metrics.sort_values('pooled_ap',ascending=False).to_dict('records'):lines.append('| '+r['experiment']+' | '+' | '.join(f'{r.get(c,float("nan")):.6f}' for c in ['pooled_ap','day_25','day_26'])+' |')
lines += ['', '## Checks',f'Validated {len(manifest)} candidate CSVs, each with exactly the 362 sample IDs, finite scores in [0,1]. Original V8 submission hash unchanged.', '', '## Evidence and limitations','Supervised fusion uses raw frozen-model geometry, not leaked two-fold supervised model predictions. PCA/scalers/quality heads and scene priors are fit on the fitting day only. The historical V8 baseline itself was developed using these two days.','Machine candidates require independent detector agreement, image-quality checks, temporal exclusion and visual review. Accepted labels remain silver; they are not gold validation labels. Only a cup/tray specialist may use reviewed crops, preventing unlabeled people/hands from becoming false background.','Frozen DINO spatial head sees a 3x3 box grid and four outer boundary bands. Synthetic examples supplement real fit-day error labels; no held-day synthetic examples enter fitting.','Paired frame bootstrap is descriptive and cannot correct the two-day distribution mismatch or model-selection optimism.']
(F/'REPORT.md').write_text('\n'.join(lines));print('FINAL_METRICS',metrics.sort_values('pooled_ap',ascending=False).head(15).to_json(orient='records'),flush=True);print('BOOTSTRAP',json.dumps(ci),flush=True);print('VALIDATED_CANDIDATES',len(manifest),flush=True)

if (E/'stage6/pool_status.json').exists():
 status=json.loads((E/'stage6/pool_status.json').read_text());lines+=['','## Machine-pool status',json.dumps(status,indent=2),'Expanded-data YOLO retraining was NOT executed because no candidate passed review in the available partial sample and the full dataset could not be accessed. The existing final four-class YOLO was trained for 80 epochs on all 84 manual frames in V3; weights are retained in V5.']
(F/'REPORT.md').write_text(chr(10).join(lines))

_best=pd.read_csv(E/'stage2/catboost_person_hands0.5_yolo_cup_tray20_oof.csv').set_index('id').loc[base.id].reset_index();_classrows=[]
for _cls,_g in base.groupby('class_name'):
 _i=_g.index;_classrows.append(dict(class_name=_cls,rows=len(_i),errors=int(_g.error.sum()),baseline_ap=float(average_precision_score(_g.error,_g.score)),candidate_ap=float(average_precision_score(_g.error,_best.score.iloc[_i]))))
pd.DataFrame(_classrows).to_csv(F/'by_class_comparison.csv',index=False)
lines+=['','## Best candidate by class','| Class | Queries | Errors | V8 AP | Candidate AP |','|---|---:|---:|---:|---:|']
for _r in _classrows:lines.append('| '+_r['class_name']+' | '+str(_r['rows'])+' | '+str(_r['errors'])+' | '+format(_r['baseline_ap'],'.6f')+' | '+format(_r['candidate_ap'],'.6f')+' |')
_audit=_best.copy();_scores=_audit.score.to_numpy();_labels=_audit.error.to_numpy();_audit['ranking_inversions']=[int(np.sum(_scores[_labels!=_labels[_j]]>=_scores[_j])) if _labels[_j]==1 else int(np.sum(_scores[_labels!=_labels[_j]]<=_scores[_j])) for _j in range(len(_audit))];_audit.sort_values('ranking_inversions',ascending=False).to_csv(F/'best_candidate_ranking_errors.csv',index=False)
print('CLASS_COMPARISON',json.dumps(_classrows),flush=True)
(F/'REPORT.md').write_text(chr(10).join(lines))

from pathlib import Path
import json,ast,base64,gzip,zipfile,hashlib,shutil
W=Path('/kaggle/working');E=W/'rascar_v14';out=E/'final_audit';out.mkdir(exist_ok=True)
# Provide readable source beside the notebook and keep every prior version intact.
nbpath=W/'__notebook__.ipynb'
if nbpath.exists():
 nb=json.loads(nbpath.read_text());cells=[]
 for cell in nb['cells']:
  if cell['cell_type']!='code':cells.append(cell);continue
  source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source'];readable=None
  try:
   tree=ast.parse(source)
   for node in ast.walk(tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):readable=gzip.decompress(base64.b64decode(node.args[0].value)).decode();break
  except Exception:pass
  # Decode for review; source executes in original notebook in isolated subprocesses where specified.
  cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True)))
 nb['cells']=cells;(W/'source_v14_all_experiments.ipynb').write_text(json.dumps(nb));shutil.copy2(nbpath,W/'execution_v14_all_experiments.ipynb')
# Small, portable source/metrics archive. Images, model weights and embeddings stay separately in Kaggle Output.
with zipfile.ZipFile(W/'rascar_v14_code_metrics_submissions.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in W.rglob('*'):
  if not p.is_file() or p.suffix not in ['.py','.json','.csv','.md','.ipynb','.yaml','.txt']:continue
  if p.stat().st_size>15_000_000 or any(x in p.parts for x in ['v8_replay','v7_replay','dataset','test_images','pool_images','images']):continue
  z.write(p,p.relative_to(W))
print('DELIVERABLES_READY',len(list(W.glob('submission_*.csv'))),'root submissions',flush=True)
