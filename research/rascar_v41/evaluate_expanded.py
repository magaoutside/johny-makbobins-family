"""Fixed-weight pool ablations; transductive exports deliberately have no AP."""
from common import *
q,te,h=inputs();bo,bt=baseline(q,te);models={};rows=[];preds={}
def aligned(d):
 o=d[d.split.eq('query')].set_index('id').loc[q.id,'score'].to_numpy()
 t=d[d.split.eq('test')].groupby('id').score.mean().loc[te.id].to_numpy()
 assert len(o)==198 and len(t)==362 and np.isfinite(o).all() and np.isfinite(t).all()
 return o,t
def save(name,o,t,kind):
 rows.append(dict(experiment=name,kind=kind,**metrics(q,o),changed_test=int((abs(t-bt)>1e-8).sum()),max_test_delta=float(abs(t-bt).max())))
 preds[name]=(o,t);q.assign(score=o).to_csv(A/f'results/oof/{name}.csv',index=False)
 te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
save('expanded_v31_baseline',bo,bt,'baseline')
for key in ['vit_b','vit_l']:
 models[key+'_original']=aligned(pd.read_csv(A/f'features/{key}_geometry_predictions.csv'))
 d=pd.read_csv(A/f'features/{key}_expanded_predictions.csv')
 for variant,g in d.groupby('variant'):models[key+'_'+variant]=aligned(g)
for variant in ['original','clean_continue','expanded_pool']:
 models['BL_'+variant]=tuple(np.mean([models[key+'_'+variant][j] for key in ['vit_b','vit_l']],axis=0) for j in [0,1])
clip=aligned(pd.read_csv(A/'features/clip_geometry_predictions.csv'))
yolo=aligned(pd.concat([pd.read_csv(A/f'features/yolo_fit{d}_predictions.csv') for d in ['25','26']]))
for variant in ['clean_continue','expanded_pool']:
 models['four_'+variant]=tuple(np.mean([models[key+'_'+variant][j] for key in ['vit_b','vit_l']]+[clip[j],yolo[j]],axis=0) for j in [0,1])
for name,(o,t) in models.items():
 save(name+'_standalone',o,t,'standalone')
 for weight in [.1,.2]:
  save(name+f'_blend{weight}',(1-weight)*bo+weight*o,(1-weight)*bt+weight*t,'blend')
 # Single-class ablations only: no ID-specific rules, no combinatorial search.
 for cls in CL:
  weight=.2;mo=q.class_name.eq(cls).to_numpy();mt=te.class_name.eq(cls).to_numpy()
  save(name+f'_{cls}_blend{weight}',np.where(mo,(1-weight)*bo+weight*o,bo),np.where(mt,(1-weight)*bt+weight*t,bt),'class_ablation')
reg=pd.DataFrame(rows);reg['delta']=reg.ap-reg.iloc[0].ap
for day in DAYS:reg['delta_'+day[-2:]]=reg[day]-reg.iloc[0][day]
reg['both_days_nonnegative']=(reg.delta_25>=-1e-12)&(reg.delta_26>=-1e-12)
reg.to_csv(A/'results/expanded_experiments.csv',index=False)
print(reg.sort_values('ap',ascending=False)[['experiment','ap','delta','delta_25','delta_26']].head(22).to_string(index=False),flush=True)
names=list(reg[reg.kind.eq('blend')].nlargest(4,'ap').experiment)+list(reg[reg.kind.eq('class_ablation')].nlargest(3,'ap').experiment)
rng=np.random.default_rng(4151);boots={x:[] for x in set(names)};frames=[q.loc[q.day.eq(d),'frame_id'].unique() for d in DAYS];groups={f:np.flatnonzero(q.frame_id.eq(f)) for fs in frames for f in fs}
for _ in range(2000):
 ii=np.concatenate([np.concatenate([groups[f] for f in rng.choice(fs,len(fs),replace=True)]) for fs in frames]);y=q.error.to_numpy()[ii];b=average_precision_score(y,bo[ii])
 for x in boots:boots[x].append(float(average_precision_score(y,preds[x][0][ii])-b))
st=[]
for name,arr in boots.items():
 o,t=preds[name];loo=[]
 for f in q.frame_id.unique():
  m=~q.frame_id.eq(f);loo.append(float(average_precision_score(q.loc[m,'error'],o[m])-average_precision_score(q.loc[m,'error'],bo[m])))
 st.append(dict(experiment=name,low=float(np.quantile(arr,.025)),high=float(np.quantile(arr,.975)),fraction_positive=float(np.mean(np.array(arr)>0)),leave_one_frame_out_min=min(loo),leave_one_frame_out_max=max(loo)))
pd.DataFrame(st).to_csv(A/'results/expanded_stability.csv',index=False)
# Full-data final models: no OOF value can be assigned to these exact models.
trans=[]
for key in ['vit_b','vit_l']:
 d=pd.read_csv(A/f'features/{key}_transductive_predictions.csv');tt={}
 for variant,g in d.groupby('variant'):
  t=g.set_index('id').loc[te.id,'score'].to_numpy();tt[variant]=t
  for w in [.1,.2]:
   name=f'{key}_{variant}_UNVALIDATED_blend{w}';s=(1-w)*bt+w*t
   te[['id']].assign(score=s).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
   trans.append(dict(experiment=name,ap=None,claims_oof=False,changed_test=int((abs(s-bt)>1e-8).sum()),max_test_delta=float(abs(s-bt).max())))
 delta=tt['final_auto_test']-tt['final_no_test']
 te.assign(no_test=tt['final_no_test'],auto_test=tt['final_auto_test'],delta=delta).to_csv(A/f'audit/{key}_automatic_test_adaptation_changes.csv',index=False)
pd.DataFrame(trans).to_csv(A/'results/transductive_exports.csv',index=False)
print('EXPANDED EVALUATION DONE',flush=True)
