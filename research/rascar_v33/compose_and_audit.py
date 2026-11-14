"""Combine disjoint positive/negative signals; audit frame stability and files."""
from evaluate import *
from sklearn.metrics import average_precision_score as AP

def main():
    parent_s=align(ROOT/'rascar_v32/results/oof/compose_watershed1_confirmed_hand_protection2.csv',q,'score')
    parent_t=align(ROOT/'rascar_v32/submission_v32_research_oof098800.csv',te,'score')
    neg='v32_pool_regional_absence_w0.25';ns=align(A/f'results/oof/{neg}.csv',q,'score');nt=align(A/f'results/submissions/submission_{neg}.csv',te,'score')
    entries=[];chosen={'v32_frozen':parent_s,neg:ns}
    for label in ['v33_conservative_geometry_and_low_score_protection','v33_consistent_geometry_and_low_score_protection','v33_positive_confirmation_only','v32_pool_regional_balanced_boundary_conservative_w0.25','v32_pool_regional_balanced_rare_absence_w0.25']:
        chosen[label]=align(A/f'results/oof/{label}.csv',q,'score')
    for frac in [.5,1.]:
        pos=f'v32_positive_geometry_restore{frac}';ps=align(A/f'results/oof/{pos}.csv',q,'score');pt=align(A/f'results/submissions/submission_{pos}.csv',te,'score')
        ss=np.clip(ns+ps-parent_s,0,1);tt=np.clip(nt+pt-parent_t,0,1);name=f'v33_pool_absence_plus_confirmation{frac}'
        q.assign(score=ss).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=tt).to_csv(A/f'results/submissions/submission_{name}.csv',index=False)
        entries.append(dict(experiment=name,**metric(q,ss),changed_oof=int((abs(ss-parent_s)>1e-9).sum()),changed_test=int((abs(tt-parent_t)>1e-9).sum())))
        chosen[pos]=ps;chosen[name]=ss
    pd.DataFrame(entries).to_csv(A/'results/composition_experiments.csv',index=False)
    # Whole-frame paired bootstrap within each day; descriptive after selection.
    rng=np.random.default_rng(3310);groups={d:[np.flatnonzero(q.frame_id.eq(f).to_numpy()) for f in q.loc[q.day.eq(d),'frame_id'].unique()] for d in DAYS}
    samples=[np.concatenate([np.concatenate([g[j] for j in rng.integers(0,len(g),len(g))]) for g in groups.values()]) for _ in range(2000)]
    y=q.error.to_numpy();pb=np.array([AP(y[ix],parent_s[ix]) for ix in samples]);stability=[]
    for name,s in chosen.items():
        boot=np.array([AP(y[ix],s[ix]) for ix in samples])-pb
        loo=[AP(y[~q.frame_id.eq(f)],s[~q.frame_id.eq(f)])-AP(y[~q.frame_id.eq(f)],parent_s[~q.frame_id.eq(f)]) for f in q.frame_id.unique()]
        stability.append(dict(experiment=name,**metric(q,s),delta_ap_vs_v32=AP(y,s)-AP(y,parent_s),bootstrap_lo=float(np.quantile(boot,.025)),bootstrap_hi=float(np.quantile(boot,.975)),bootstrap_positive_fraction=float((boot>0).mean()),leave_one_frame_delta_min=min(loo),leave_one_frame_delta_max=max(loo)))
    pd.DataFrame(stability).to_csv(A/'results/frame_stability.csv',index=False)
    manifest=[]
    for path in sorted((A/'results/oof').glob('*.csv')):
        s=align(path,q,'score');sub=A/f'results/submissions/submission_{path.stem}.csv';d=pd.read_csv(sub);assert len(d)==362 and d.id.is_unique and list(d.id)==list(te.id)
        assert np.isfinite(s).all() and np.isfinite(d.score).all() and (d.score.between(0,1)).all() and ((s>=0)&(s<=1)).all()
        manifest.append(dict(experiment=path.stem,oof_ap=float(AP(y,s)),test_rows=len(d),sha256=hashlib.sha256(sub.read_bytes()).hexdigest()))
    (A/'results/submission_integrity.json').write_text(json.dumps(dict(experiments_checked=len(manifest),files=manifest,leaderboard_upload=False,bootstrap_caveat='Post-selection descriptive sensitivity, not an independent holdout or leaderboard forecast.'),indent=2))
    print(pd.DataFrame(entries).to_string(index=False));print(pd.DataFrame(stability).to_string(index=False));print('CHECKED',len(manifest))
if __name__=='__main__':main()
