"""Package fixed 20% person-only ablations and audited research candidates."""
from common import *
q,te,h=inputs();bo,bt=baseline(q,te);rows=[];models={};all_predictions={}
def aligned(d):
 return d[d.split.eq('query')].set_index('id').loc[q.id].score.to_numpy(),d[d.split.eq('test')].groupby('id').score.mean().loc[te.id].to_numpy()
def person_blend(o,t):return np.where(q.class_name.eq('person'),.8*bo+.2*o,bo),np.where(te.class_name.eq('person'),.8*bt+.2*t,bt)
def record(name,o,t,kind):
 rows.append(dict(experiment=name,kind=kind,**metrics(q,o),delta=float(average_precision_score(q.error,o)-average_precision_score(q.error,bo))))
 all_predictions[name]=(o,t)
 q.assign(score=o).to_csv(A/f'results/oof/{name}.csv',index=False)
 te[['id']].assign(score=t).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
record('v41_reference_v31',bo,bt,'baseline')
for group in ['clean','all','old']:
 accum=[]
 for seed in [4142,4143,4144]:
  suffix='' if seed==4142 and group!='old' else f'_s{seed}_{"old" if group=="old" else "all"}'
  d=pd.read_csv(A/f'features/vit_l_expanded{suffix}_predictions.csv');d=d[d.variant.eq('clean_continue' if group=='clean' else 'expanded_pool')]
  pair=aligned(d);accum.append(pair);record(f'v41_{group}_person_seed{seed}',*person_blend(*pair),'seed')
 models[group]=tuple(np.mean([x[j] for x in accum],axis=0) for j in [0,1])
 record(f'v41_{group}_person_3seed',*person_blend(*models[group]),'three_seed_ensemble')
d=pd.read_csv(A/'features/vit_l_foldadapt_predictions.csv')
for variant,g in d.groupby('variant'):record('v41_'+variant+'_person',*person_blend(*aligned(g)),'transductive_simulation' if variant=='fold_auto' else 'adaptation_control')
reg=pd.DataFrame(rows);reg.to_csv(A/'results/final_comparison.csv',index=False)
# Stratified, paired frame bootstrap; uncertainty is conditional on this reused
# development set and does not account for historical model selection.
comparisons=[('v41_old_person_3seed','v41_reference_v31'),('v41_old_person_3seed','v41_clean_person_3seed'),('v41_all_person_3seed','v41_old_person_3seed'),('v41_fold_auto_person','v41_fold_control_person')]
rng=np.random.default_rng(4152);frames=[q.loc[q.day.eq(d),'frame_id'].unique() for d in DAYS];groups={f:np.flatnonzero(q.frame_id.eq(f)) for fs in frames for f in fs};boots={x:[] for x in comparisons}
for _ in range(2000):
 ii=np.concatenate([np.concatenate([groups[f] for f in rng.choice(fs,len(fs),replace=True)]) for fs in frames]);y=q.error.to_numpy()[ii]
 for name,reference in comparisons:boots[(name,reference)].append(float(average_precision_score(y,all_predictions[name][0][ii])-average_precision_score(y,all_predictions[reference][0][ii])))
stats=[]
for (name,reference),v in boots.items():
 loo=[]
 for f in q.frame_id.unique():
  m=~q.frame_id.eq(f);loo.append(float(average_precision_score(q.loc[m,'error'],all_predictions[name][0][m])-average_precision_score(q.loc[m,'error'],all_predictions[reference][0][m])))
 stats.append(dict(experiment=name,reference=reference,low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),fraction_positive=float(np.mean(np.array(v)>0)),leave_one_frame_out_min=min(loo),leave_one_frame_out_max=max(loo)))
pd.DataFrame(stats).to_csv(A/'results/final_stability.csv',index=False)
# Give only the stable, fixed-seed research candidate a short top-level filename.
name='v41_old_person_3seed';o,t=all_predictions[name]
for src,dst in [(name,'submission_v41_pool_person_3seed.csv'),('v41_all_person_3seed','submission_v41_expanded_pool_person_3seed.csv')]:
 te[['id']].assign(score=all_predictions[src][1]).to_csv(A/dst,index=False)
final=pd.read_csv(A/'features/vit_l_transductive_predictions.csv')
for variant,g in final.groupby('variant'):
 s=g.set_index('id').loc[te.id].score.to_numpy();s=np.where(te.class_name.eq('person'),.8*bt+.2*s,bt)
 te[['id']].assign(score=s).to_csv(A/f'results/submissions/submission_v41_{variant}_person_UNVALIDATED.csv',index=False)
metadata=dict(selected_experiment=name,selection='Exploratory: 20% correction for person only; average all 3 fixed seeds and both fit days on test. No test labels or ID-specific rules.',validation=metrics(q,o),public_ap=None,baseline_public_ap_reported_by_user=.96239,baseline_submission_sha256=hashlib.sha256((ROOT/'rascar_v31/submission_v31_dense_oof098784.csv').read_bytes()).hexdigest(),weights=dict(v31=.8,vit_l_geometry=.2),class_scope=['person'],fit_seeds=[4142,4143,4144],newly_reviewed_pool_in_selected_candidate=False,old_weak_geometry_boxes=154,test_pseudo_labels_in_selected_candidate=False,changed_test=int((abs(t-bt)>1e-8).sum()),max_test_delta=float(abs(t-bt).max()),leaderboard_uploaded=False,uncertainty='Repeated development OOF; bootstrap CI includes zero; hidden test labels unknown.')
(A/'candidate_metadata.json').write_text(json.dumps(metadata,indent=2))
viewer=json.loads((ROOT/'rascar_test_viewer/data.js').read_text().removeprefix('window.RASCAR_DATA = ').rstrip(';\n'));position={x['id']:x['position'] for x in viewer['detections']}
te.assign(v31=bt,candidate=t,delta=t-bt,viewer_position=te.id.map(position)).to_csv(A/'audit/final_candidate_changes.csv',index=False)
print(reg.to_string(index=False));print(pd.DataFrame(stats).to_string(index=False));print(json.dumps(metadata,indent=2))
