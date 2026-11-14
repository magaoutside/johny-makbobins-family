"""Fixed ablations; descriptive development OOF, not a fresh holdout.
No test labels or per-ID rules. Every CSV is a research artifact, not an upload.
"""
from common import *
import argparse
from scipy.stats import rankdata
p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
q,te,h=inputs();bo,bt=baseline(q,te);models={};quality={};rows=[];predictions={}
def aligned(df):
 o=df[df.split.eq('query')].set_index('id').loc[q.id,'score'].to_numpy()
 t=df[df.split.eq('test')].groupby('id').score.mean().loc[te.id].to_numpy()
 assert np.isfinite(o).all() and np.isfinite(t).all()
 return o,t
for key in ['clip','vit_b','vit_l']:
 path=A/f'features/{key}_predictions_s4101.csv'
 if not path.exists():
  if args.partial:continue
  raise FileNotFoundError(path)
 d=pd.read_csv(path)
 for variant,g in d.groupby('variant'):
  models[key+'_'+variant]=aligned(g)
  if variant=='finetuned':
   quality[key]=aligned(g.assign(score=np.clip((.85-g.predicted_iou)/.5,0,1)))
 path=A/f'features/{key}_geometry_predictions.csv'
 if path.exists():models[key+'_geometry']=aligned(pd.read_csv(path))
yp=[A/f'features/yolo_fit{x}_predictions.csv' for x in ['25','26']]
if all(x.exists() for x in yp):models['yolo']=aligned(pd.concat([pd.read_csv(x) for x in yp]))
elif not args.partial:raise FileNotFoundError('YOLO predictions missing')
if (A/'features/clip_text_predictions.csv').exists():models['clip_text_presence']=aligned(pd.read_csv(A/'features/clip_text_predictions.csv'))
def save(name,o,t,kind):
 assert len(o)==198 and len(t)==362 and np.isfinite(o).all() and np.isfinite(t).all()
 assert ((o>=0)&(o<=1)).all() and ((t>=0)&(t<=1)).all()
 row=dict(experiment=name,kind=kind,**metrics(q,o),changed_test=int((abs(t-bt)>1e-8).sum()),max_test_delta=float(abs(t-bt).max()))
 rows.append(row);predictions[name]=(o,t)
 q.assign(score=o).to_csv(A/f'results/oof/{name}.csv',index=False)
 te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
 return row
save('v31_baseline',bo,bt,'baseline')
for name,(o,t) in models.items():save(name,o,t,'standalone')
for name,(o,t) in quality.items():save(name+'_iou',o,t,'standalone_quality')
ensembles={}
for variant in ['frozen','finetuned','geometry']:
 names=[k+'_'+variant for k in ['clip','vit_b','vit_l']]
 if all(x in models for x in names):
  ensembles['vision_'+variant]=tuple(np.mean([models[x][s] for x in names],axis=0) for s in [0,1])
  if 'yolo' in models:
   ensembles['four_'+variant]=tuple(np.mean([models[x][s] for x in names+['yolo']],axis=0) for s in [0,1])
   for remove in ['clip','vit_b','vit_l','yolo']:
    ns=[x for x in names+['yolo'] if x!=remove and x!=remove+'_'+variant]
    ensembles['without_'+remove+'_'+variant]=tuple(np.mean([models[x][s] for x in ns],axis=0) for s in [0,1])
for name,(o,t) in ensembles.items():save(name,o,t,'ensemble_standalone')
# Predeclared blend weights and model families, no threshold search on test.
for name,(o,t) in {**models,**ensembles}.items():
 for weight in [.05,.1,.2]:save(name+f'_blend{weight}',(1-weight)*bo+weight*o,(1-weight)*bt+weight*t,'blend')
reg=pd.DataFrame(rows);reg['delta']=reg.ap-reg.iloc[0].ap
for day in DAYS:reg['delta_'+day[-2:]]=reg[day]-reg.iloc[0][day]
reg['both_days_nonnegative']=(reg.delta_25>=-1e-12)&(reg.delta_26>=-1e-12)
reg.to_csv(A/'results/experiments.csv',index=False)
print(reg.sort_values('ap',ascending=False)[['experiment','ap','delta','delta_25','delta_26','changed_test']].head(15).to_string(index=False),flush=True)
# Paired frame bootstrap, within day, for the four-family blends and leaders.
candidates=list(reg[reg.experiment.str.startswith('four_')&reg.kind.eq('blend')].experiment)
candidates+=list(reg[reg.kind.eq('blend')].nlargest(4,'ap').experiment)
rng=np.random.default_rng(4101);boots={x:[] for x in set(candidates)};idx=[q.index[q.day.eq(d)].to_numpy() for d in DAYS]
frames=[q.loc[ix,'frame_id'].unique() for ix in idx];groups={f:np.flatnonzero(q.frame_id.eq(f)) for fs in frames for f in fs}
for _ in range(2000):
 ii=np.concatenate([np.concatenate([groups[f] for f in rng.choice(fs,len(fs),replace=True)]) for fs in frames]);y=q.error.to_numpy()[ii]
 base=average_precision_score(y,bo[ii])
 for x in boots:boots[x].append(float(average_precision_score(y,predictions[x][0][ii])-base))
stability=[]
for name,arr in boots.items():
 o,t=predictions[name];loo=[]
 for f in q.frame_id.unique():
  m=~q.frame_id.eq(f);loo.append(float(average_precision_score(q.loc[m,'error'],o[m])-average_precision_score(q.loc[m,'error'],bo[m])))
 stability.append(dict(experiment=name,paired_frame_bootstrap_low=float(np.quantile(arr,.025)),high=float(np.quantile(arr,.975)),fraction_positive=float(np.mean(np.array(arr)>0)),leave_one_frame_out_min=min(loo),leave_one_frame_out_max=max(loo)))
pd.DataFrame(stability).to_csv(A/'results/stability.csv',index=False)
# User-discussed IDs are diagnostics only, never targets or selection conditions.
viewer=json.loads((ROOT/'rascar_test_viewer/data.js').read_text().removeprefix('window.RASCAR_DATA = ').rstrip(';\n'))
rank={d['id']:d['rank'] for d in viewer['detections']};watch=[]
for name in candidates:
 o,t=predictions[name];g=te.copy();g['rank_v31']=g.id.map(rank);g['v31']=bt;g['new_score']=t;g['delta']=t-bt;g['experiment']=name
 watch.append(g[g.rank_v31.isin([41,53,63,64,67,77,78,82,87,117])])
if watch:pd.concat(watch).to_csv(A/'audit/discussed_examples.csv',index=False)
print('Saved',len(reg),'experiments. These are development OOF estimates; no LB upload.',flush=True)
