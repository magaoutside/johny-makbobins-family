from common27 import *
import shutil,zipfile
import render23

def main():
    outputs={};(A/'replay_inputs').mkdir(exist_ok=True)
    for split,rows,base in [('oof',q,base_oof),('test',te,current)]:
        path=lambda name:A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}.csv')
        assembled=align(path('assembled_control'),rows,'score')
        merged=align(path('contained_people_w0.6'),rows,'score')
        weak=align(path('weak_temporal_fusion'),rows,'score')
        control=np.clip(assembled+merged-base,0,1)
        candidate=np.clip(control+weak-base,0,1)
        outputs[split]=dict(control=control,candidate=candidate)
        rows.assign(baseline_v22=base,assembled=assembled,merged_delta=merged-base,weak_hand_delta=weak-base)[['id','class_name','baseline_v22','assembled','merged_delta','weak_hand_delta']].to_csv(A/f'replay_inputs/{split}.csv',index=False)
    summaries={}
    for name in ['control','candidate']:
        summaries[name]=register('release_'+name,outputs['oof'][name],outputs['test'][name],dict(type='V27_release',
            components=['reviewed V24 cup ensemble','reviewed V26 hands and trays','V27 contained-person group penalty']+(['V27 weak-hand temporal corroboration'] if name=='candidate' else []),
            validation_limitation='The two newly closed audit cases have no matching main-OOF activation. Auxiliary geometric checks and visual development audit support them; OOF gain mainly comes from previously saved cup experts.'))
        shutil.copyfile(A/f'results/submissions/submission_release_{name}.csv',A/f'submission_v27_{name}.csv')
    render23.A=A
    d=pd.read_csv(A/'results/reviews/release_candidate.csv')
    changed=d[d.delta.abs()>1e-9].sort_values('review_no');changed.to_csv(A/'audit/candidate_changes.csv',index=False)
    render23.render(changed,'candidate_changes')
    original=(audit.visual_status.eq('plausible_correct')&(current>=threshold))|(audit.visual_status.isin(['clear_error','likely_error'])&(current<threshold))
    closed=d.loc[original&d.diagnostic_pass,['id','review_no','class_name','issue','baseline_v22','new_score']]
    closed.to_csv(A/'audit/new_diagnostic_closures.csv',index=False)
    failed=d[original&~d.diagnostic_pass];failed.to_csv(A/'audit/remaining_frozen_mismatches.csv',index=False)
    render23.render(failed,'remaining_cases')
    # Paired resampling of whole frames within each day; descriptive only, does
    # not account for extensive model selection on this same validation set.
    rng=np.random.default_rng(2705);deltas=[];score=outputs['oof']['candidate']
    for _ in range(2000):
        indices=[]
        for day in DAYS:
            frames=q.loc[q.day.eq(day),'frame_id'].unique()
            for fid in rng.choice(frames,len(frames),replace=True):indices.extend(np.flatnonzero(q.frame_id.eq(fid)))
        ii=np.asarray(indices);deltas.append(float(AP(q.error.to_numpy()[ii],score[ii])-AP(q.error.to_numpy()[ii],base_oof[ii])))
    bootstrap=dict(samples=2000,unit='frame, stratified by day',mean_delta=float(np.mean(deltas)),ci95=np.quantile(deltas,[.025,.975]).tolist(),
        descriptive_only=True,does_not_adjust_for_model_selection=True)
    (A/'results/paired_frame_bootstrap.json').write_text(json.dumps(bootstrap,indent=2))
    registry=[]
    for p in (A/'results').glob('*.json'):
        r=json.loads(p.read_text())
        if 'experiment' in r:registry.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
    pd.DataFrame(registry).sort_values('pooled_ap',ascending=False).to_csv(A/'results/experiment_registry.csv',index=False)
    checks=[]
    for p in (A/'results/submissions').glob('*.csv'):
        df=pd.read_csv(p);assert df.id.tolist()==te.id.tolist() and df.id.is_unique and df.score.between(0,1).all() and np.isfinite(df.score).all();checks.append(p.name)
    assert hashlib.sha256((V22/'submission_v22_candidate.csv').read_bytes()).hexdigest()=='44b09d4cff99fcac1e42b2f9d245ae8967be1437a6571f704cf4b3cdd2698c2c'
    summary=dict(baseline_lb=.95610,baseline_oof=metric(q,base_oof),**summaries,
        variants=len(registry),final_model_bundles=len(list((A/'models').glob('*.joblib'))),gpu=json.loads((A/'gpu/DONE.json').read_text()),
        new_frozen_closures=closed.review_no.tolist(),closed_fraction_of_v22_remaining=len(closed)/int(original.sum()),
        unresolved_hard_hands=failed[failed.class_name.eq('nude_hands')].review_no.tolist(),
        submitted=False,lb=None,bootstrap=bootstrap)
    (A/'summary.json').write_text(json.dumps(summary,indent=2))
    (A/'checks.json').write_text(json.dumps(dict(checked_submissions=len(checks),baseline_unchanged=True,ids_and_scores_valid=True,no_submission_to_lb=True),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
