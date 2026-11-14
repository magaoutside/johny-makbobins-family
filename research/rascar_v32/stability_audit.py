"""Paired, day-stratified frame bootstrap and exact ranking diagnostics.

Exploratory uncertainty only: this does not undo tuning on these same 35 frames.
Each bootstrap samples whole frames, keeping all queries from each sampled frame.
"""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parent.parent; A=ROOT/'rascar_v32'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,base_oof,align,metric

def main():
    selected={
      'V22_primary':base_oof,
      'V31_dense_wide':align(ROOT/'rascar_v31/results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv',q).score.to_numpy(),
      'V32_watershed_hands':align(A/'results/oof/watershed_fused_cut0.35_nude_hands_penalty_only_w0.25.csv',q).score.to_numpy(),
      'V32_qfl_point':align(A/'results/oof/qfl_point_cut0.5_all_penalty_only_w0.25.csv',q).score.to_numpy(),
      'V32_qfl_distribution':align(A/'results/oof/qfl_distribution_cut0.5_all_penalty_only_w0.1.csv',q).score.to_numpy(),
      'V32_composed':align(A/'results/oof/compose_watershed1_confirmed_hand_protection0.csv',q).score.to_numpy(),
    }
    y=q.error.to_numpy(int); n=len(y); rng=np.random.default_rng(3202)
    frames={d:[np.flatnonzero(q.frame_id.eq(f).to_numpy()) for f in q.loc[q.day.eq(d),'frame_id'].unique()] for d in sorted(q.day.unique())}
    samples=[]
    for _ in range(2000):
        ix=np.concatenate([np.concatenate([ff[k] for k in rng.integers(0,len(ff),size=len(ff))]) for ff in frames.values()])
        assert len(np.unique(y[ix]))==2
        samples.append(ix)
    base_boot=np.array([average_precision_score(y[ix],base_oof[ix]) for ix in samples])
    v31_boot=np.array([average_precision_score(y[ix],selected['V31_dense_wide'][ix]) for ix in samples])
    summary=[];loo=[];inversions=[];case=[]
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    def pair_quality(s):
        delta=s[pos,None]-s[None,neg]
        return (delta>0).astype(float)+.5*(delta==0)
    oldpairs=pair_quality(base_oof)
    for name,s in selected.items():
        ap=average_precision_score(y,s)
        boot=np.array([average_precision_score(y[ix],s[ix]) for ix in samples]);diff=boot-base_boot
        row=dict(experiment=name,**metric(q,s),delta_ap=ap-average_precision_score(y,base_oof),
                 bootstrap_delta_lo=float(np.quantile(diff,.025)),bootstrap_delta_hi=float(np.quantile(diff,.975)),
                 bootstrap_fraction_positive=float((diff>0).mean()),bootstrap_replicates=len(samples))
        row.update(delta_vs_v31=ap-average_precision_score(y,selected['V31_dense_wide']),
                   bootstrap_vs_v31_lo=float(np.quantile(boot-v31_boot,.025)),
                   bootstrap_vs_v31_hi=float(np.quantile(boot-v31_boot,.975)),
                   bootstrap_vs_v31_fraction_positive=float((boot>v31_boot).mean()))
        local=[]
        for f in q.frame_id.unique():
            keep=~q.frame_id.eq(f).to_numpy();delta=average_precision_score(y[keep],s[keep])-average_precision_score(y[keep],base_oof[keep])
            loo.append(dict(experiment=name,removed_frame=f,n_removed=int((~keep).sum()),delta_ap=delta));local.append(delta)
        row.update(leave_one_frame_delta_min=float(min(local)),leave_one_frame_delta_max=float(max(local)))
        newpairs=pair_quality(s);improvement=newpairs-oldpairs
        row.update(error_correct_pairs_improved=int((improvement>0).sum()),error_correct_pairs_worsened=int((improvement<0).sum()))
        for a,b in zip(*np.nonzero(improvement)):
            i,j=pos[a],neg[b]
            inversions.append(dict(experiment=name,error_id=q.iloc[i].id,correct_id=q.iloc[j].id,
                error_class=q.iloc[i].class_name,correct_class=q.iloc[j].class_name,
                direction='improved' if improvement[a,b]>0 else 'worsened',
                old_error=float(base_oof[i]),old_correct=float(base_oof[j]),new_error=float(s[i]),new_correct=float(s[j])))
        # Tie-aware exact decomposition of 1-AP: every tied score has same precision.
        order=np.argsort(-s,kind='stable');cum=np.cumsum(y[order]);loss=np.zeros(n)
        for val in np.unique(s):
            tied=np.flatnonzero(s[order]==val);precision=cum[tied[-1]]/(tied[-1]+1)
            loss[order[tied]]=y[order[tied]]*(1-precision)/y.sum()
        assert abs(loss.sum()-(1-ap))<1e-12
        for i,r in q.iterrows():
            case.append(dict(experiment=name,id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,error=int(y[i]),score=float(s[i]),ap_loss=float(loss[i])))
        summary.append(row)
    pd.DataFrame(summary).to_csv(A/'results/frame_stability.csv',index=False)
    pd.DataFrame(loo).to_csv(A/'audit/leave_one_frame_out.csv',index=False)
    pd.DataFrame(inversions).to_csv(A/'audit/rank_pair_changes.csv',index=False)
    pd.DataFrame(case).to_csv(A/'audit/ap_loss_after_experiments.csv',index=False)
    (A/'audit/stability_protocol.json').write_text(json.dumps(dict(n_queries=n,n_frames=int(q.frame_id.nunique()),
        frames_per_day={d:len(v) for d,v in frames.items()},seed=3202,bootstrap_replicates=2000,
        caveat='Exploratory, post-selection sensitivity. Not an independent holdout or prediction of leaderboard score.',
        grouping='Whole frames resampled within day; repeated frames keep all their queries.',
        pair_counts='Pairwise ranking diagnostic; not equal to Average Precision.'),indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
