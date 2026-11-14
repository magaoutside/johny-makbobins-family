"""Frame-level paired checks, frozen-file integrity and explicitly provisional CSVs."""
from experiments import *
from sklearn.metrics import average_precision_score as AP
import shutil

def ap(y,s):
    # Exact sklearn tie behavior, including bootstrap ties from repeated frames.
    return float(AP(y,s))

def main():
    tables=[pd.read_csv(A/f'results/{name}.csv') for name in ['experiments','boundary_experiments','semantic_experiments']]
    ex=pd.concat(tables,ignore_index=True);ex.to_csv(A/'results/all_experiments.csv',index=False)
    selected=['person_confirm_floor0.2_restore0.5','person_confirm_floor0.4_restore0.5',
              'nude_hands_boundary_identity0_w0.1','hand_semantic_c0.1_p0.15_w0.5']
    frames=q.frame_id.unique();frameids={f:np.flatnonzero(q.frame_id.eq(f).to_numpy()) for f in frames}
    rng=np.random.default_rng(3418);boot=[]
    # Resample whole frames within each day; keep day counts fixed.
    for _ in range(1500):
        ix=[]
        for day in DAYS:
            ff=q.loc[q.day.eq(day),'frame_id'].unique()
            ix += [frameids[f] for f in rng.choice(ff,len(ff),replace=True)]
        boot.append(np.concatenate(ix))
    y=q.error.to_numpy();base=BASE['oof'];summary=[];loo=[]
    for name in selected:
        s=align(A/f'results/oof/{name}.csv',q,'score')
        bt=np.array([ap(y[ix],s[ix])-ap(y[ix],base[ix]) for ix in boot])
        ll=[]
        for frame in frames:
            m=~q.frame_id.eq(frame).to_numpy();delta=ap(y[m],s[m])-ap(y[m],base[m]);ll.append(delta)
            loo.append(dict(experiment=name,omitted_frame=frame,delta_ap=delta))
        # Count changed error/correct ordering, separate within and between days.
        good=bad=within_good=within_bad=0
        for i in np.flatnonzero(y==1):
            for j in np.flatnonzero(y==0):
                old=np.sign(base[i]-base[j]);new=np.sign(s[i]-s[j])
                good+=new>old;bad+=new<old
                if q.iloc[i].day==q.iloc[j].day:
                    within_good+=new>old;within_bad+=new<old
        row=dict(experiment=name,**metric(q,s),delta_ap=ap(y,s)-ap(y,base),
            bootstrap_delta_low=float(np.quantile(bt,.025)),bootstrap_delta_high=float(np.quantile(bt,.975)),
            bootstrap_replicates=len(bt),leave_one_frame_delta_min=float(min(ll)),leave_one_frame_delta_max=float(max(ll)),
            improved_error_correct_pairs=int(good),worsened_error_correct_pairs=int(bad),
            within_day_improved_pairs=int(within_good),within_day_worsened_pairs=int(within_bad),
            caveat='Post-selection, repeatedly reused development OOF; interval does not correct selection bias or predict public AP.')
        summary.append(row)
    pd.DataFrame(summary).to_csv(A/'results/frame_stability.csv',index=False)
    pd.DataFrame(loo).to_csv(A/'audit/leave_one_frame_out.csv',index=False)
    print(pd.DataFrame(summary)[['experiment','pooled_ap','day_25','day_26','leave_one_frame_delta_min','bootstrap_delta_low','bootstrap_delta_high']].to_string(index=False))
    # Final descriptive figures for the semantic maximum: does its gain survive
    # removal of the single hardest elbow frame?
    target=q[q.id.eq('bf3cb1ea632500f7')].frame_id.iloc[0]
    d=pd.DataFrame(loo);print('SEMANTIC_WITHOUT_ELBOW_FRAME',d[d.experiment.eq(selected[-1])&d.omitted_frame.eq(target)].to_dict('records'))
    checks=[]
    for _,r in ex.iterrows():
        p=A/f'results/submissions/submission_{r.experiment}.csv';t=pd.read_csv(p)
        assert list(t.columns)==['id','score'] and t.id.is_unique and set(t.id)==set(te.id)
        assert len(t)==362 and np.isfinite(t.score).all() and t.score.between(0,1).all()
        s=align(A/f'results/oof/{r.experiment}.csv',q,'score');assert abs(ap(y,s)-r.pooled_ap)<1e-12
        checks.append(dict(experiment=r.experiment,rows=len(t),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    protocol=json.loads((A/'results/protocol.json').read_text());assert score_hashes()==protocol['frozen_hashes']
    manifest=json.loads((A/'models/semantic_manifest.json').read_text())
    for m in manifest:
        assert all(f.split('_')[-2]==m['fit_day'].replace('-','') for f in m['fit_frames'])
    for name,out in [(selected[0],'submission_v34_person_control'),(selected[-1],'submission_v34_semantic_probe')]:
        src=A/f'results/submissions/submission_{name}.csv';dst=A/(out+'.csv');shutil.copyfile(src,dst)
        metrics=ex[ex.experiment.eq(name)].iloc[0].to_dict()
        metadata=dict(file=dst.name,experiment=name,metrics={k:metrics[k] for k in ['pooled_ap','day_25','day_26']},
            public_ap=None,submitted=False,release_status='research_control_only_not_promoted_over_V31',
            best_confirmed_public_ap=.96239,best_confirmed_file=str(V31_TEST.relative_to(ROOT)),
            changed_test=int(metrics['changed_test']),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
            reason='No within-day AP gain' if 'person' in out else 'OOF gain vanishes without one elbow frame; fresh semantic check flags a real hand; test changes already high-error-score boxes.')
        dst.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    (A/'results/integrity.json').write_text(json.dumps(dict(experiments=len(ex),checks=checks,
        frozen_files_unchanged=True,semantic_fit_day_isolation=True,leaderboard_upload=False),indent=2))

if __name__=='__main__':main()
