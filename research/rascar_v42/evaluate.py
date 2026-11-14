"""Standalone score first, fixed blends second; no fitted stacking on V31 OOF."""
from base42 import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');args=p.parse_args()
q,te,h=inputs();bo,bt=baseline(q,te);models={};rows=[];preds={}
for path in sorted((A/'results').glob('*predictions_s*.csv')):
 d=pd.read_csv(path)
 for variant,g in d.groupby('variant'):
  o=g[g.source.eq('query')].set_index('id').loc[q.id].score.to_numpy();t=g[g.source.eq('test')].groupby('id').score.mean().loc[te.id].to_numpy();assert len(o)==198 and len(t)==362 and np.isfinite(o).all() and np.isfinite(t).all();seed=g.seed.iloc[0];models[f'{variant}_s{seed}']=(o,t)
if not args.partial:assert all(f'{v}_s4201' in models for v in ['tight','dual','dual_pool'])
def save(name,o,t,kind):
 row=dict(experiment=name,kind=kind,**metrics(q,o),changed_test=int((abs(t-bt)>1e-8).sum()),max_test_delta=float(abs(t-bt).max()));rows.append(row);preds[name]=(o,t)
 q.assign(score=o).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
save('v31_baseline',bo,bt,'baseline')
for name,(o,t) in models.items():
 save(name,o,t,'standalone')
 for w in [.05,.1,.2]:save(f'{name}_blend{w}',(1-w)*bo+w*o,(1-w)*bt+w*t,'blend')
reg=pd.DataFrame(rows);reg['delta']=reg.ap-reg.iloc[0].ap
for day in DAYS:reg['delta_'+day[-2:]]=reg[day]-reg.iloc[0][day]
reg.to_csv(A/'results/experiments.csv',index=False)
print(reg[['experiment','kind','ap','delta','delta_25','delta_26']].sort_values('ap',ascending=False).to_string(index=False),flush=True)
viewer=json.loads((ROOT/'rascar_test_viewer/data.js').read_text().removeprefix('window.RASCAR_DATA = ').rstrip(';\n'));rank={x['id']:x['rank'] for x in viewer['detections']};watch=te.copy();watch['rank_v31']=watch.id.map(rank);watch['v31']=bt
for name,(_,t) in models.items():watch[name]=t
watch.to_csv(A/'audit/all_test_scores.csv',index=False);watch[watch.rank_v31.isin([41,53,63,64,67,77,78,82,87,117])].to_csv(A/'audit/discussed_scores.csv',index=False)
print('EVALUATION DONE',len(models),'models',flush=True)
