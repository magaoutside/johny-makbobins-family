from common22 import *
import sys,shutil
sys.path.insert(0,str(ROOT/'rascar_v20'))
from evaluate_sahi import old_veto
P=A/'replay_inputs';P.mkdir(exist_ok=True)
pred=pd.read_csv(A/'gpu_results/spatial/predictions.csv')
for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
 part='oof' if split=='oof' else 'submissions';prefix='' if split=='oof' else 'submission_'
 oldcup=align(ROOT/f'rascar_v21/results/{part}/{prefix}cup_instance_quality_w0.5.csv',rows,'score')
 newcup=align(R/part/f'{prefix}dinov2_inside_svm_cup_guarded_w0.3.csv',rows,'score')
 hand=align(ROOT/f'rascar_v21/results/{part}/{prefix}hand_quality_both_w0.75.csv',rows,'score')
 f=align(A/f'zoom_targeted_{split}.csv',rows);sem=align(ROOT/f'rascar_v16/semantic_tight_{split}.csv',rows,'p_nude_hands')
 dd=pred[pred.representation.eq('spatial')&pred.split.eq('held' if split=='oof' else 'test')]
 quality=dd.groupby(['id','algorithm']).probability.mean().unstack().loc[rows.id]
 geo=align(ROOT/f'rascar_v18/hand_geometry_features_{split}.csv',rows)
 x=rows.copy();x['baseline']=base;x['old_cup']=oldcup;x['new_cup']=newcup;x['structural_hand']=hand;x['semantic_hand']=sem;x['old_duplicate_veto']=old_veto(rows,split=='test');x['quality_logreg']=quality.logreg.to_numpy();x['quality_svm']=quality.svm.to_numpy();x['independent_hand_centers']=geo.center_instances.to_numpy()
 for c in ['assigned','instances','overlap','confidence','coherence','folds','duplicate_loser']:x['zoom_'+c]=f[c].to_numpy()
 x.to_csv(P/f'{split}.csv',index=False)
results=[json.loads(p.read_text()) for p in R.glob('*.json')]
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in results]).sort_values('pooled_ap',ascending=False).to_csv(R/'experiment_registry.csv',index=False)
base=register('baseline_v20',q.score.to_numpy(),current,dict(lb=.95417))
prior=register('baseline_v21',align(ROOT/'rascar_v21/results/oof/release_main.csv',q,'score'),align(ROOT/'rascar_v21/submission_v21_main.csv',te,'score'),dict(lb=None))
final=json.loads((R/'final_reviewed.json').read_text());control=json.loads((R/'validation_reviewed.json').read_text())
# Stratified paired resampling by frame. This does not correct for model selection.
inherited=align(ROOT/'rascar_v18/results/oof/release_main.csv',q,'score')-align(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q,'score')
bs=np.clip(q.score.to_numpy()-inherited,0,1);vs=np.clip(align(R/'oof/final_reviewed.csv',q,'score')-inherited,0,1)
rng=np.random.default_rng(2214);deltas=[]
for _ in range(3000):
 weight=np.zeros(len(q))
 for day,g in q.groupby('day'):
  frames=g.frame_id.unique();sample=rng.choice(frames,len(frames),replace=True);counts=pd.Series(sample).value_counts();weight[g.index]=g.frame_id.map(counts).fillna(0)
 deltas.append(AP(q.error,vs,sample_weight=weight)-AP(q.error,bs,sample_weight=weight))
ci=dict(delta=float(AP(q.error,vs)-AP(q.error,bs)),ci95=np.quantile(deltas,[.025,.975]).tolist(),fraction_positive=float(np.mean(np.array(deltas)>0)),bootstrap_repetitions=3000,unit='frame, resampled separately within each day',limitation='development estimate, not adjusted for experiment selection; not a forecast of LB')
(R/'paired_frame_bootstrap.json').write_text(json.dumps(ci,indent=2))
review=pd.read_csv(R/'reviews/final_reviewed.csv');priority=review[review.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])]
byclass=priority.groupby('class_name').agg(total=('id','size'),passes=('diagnostic_pass','sum'));byclass.to_csv(R/'priority_by_class.csv')
checks=[]
for day in DAYS:
 ledger=pd.read_csv(A/f'gpu_results/hand_training/{day}/training_ledger.csv')
 assert set(ledger.day)=={day};assert not set(ledger.frame_id)&set(te.frame_id)
 assert not set(ledger.frame_id)&set(q[q.day.ne(day)].frame_id)
 checks.append(dict(fit=day,unique_frames=ledger.frame_id.nunique(),sources=ledger.groupby('source').frame_id.nunique().to_dict(),other_day_or_test_training_overlap=0))
(R/'split_integrity.json').write_text(json.dumps(checks,indent=2))
assert hashlib.sha256((A/'baseline_lb095417.csv').read_bytes()).hexdigest()=='f9b07bfcb4e92d85bcac64635c0303e2e650a6310cfe47b746d1b9c149c0babe'
summary=dict(baseline_v20=base,prior_v21=prior,candidate=final,validation_control=control,bootstrap=ci,models_trained=50,experiments_saved=len(results),leaderboard_submitted=False,known_lb=.95417)
(A/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(dict(experiments=len(results),bootstrap=ci,priority_by_class=byclass.to_dict(),changed=final['changed_test_rows']),indent=2))
