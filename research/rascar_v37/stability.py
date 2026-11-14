"""Paired frame diagnostics of frozen cross-day predictions; no refitting.

Bootstrap and leave-one-frame-out do NOT remove repeated model selection bias.
Both final-fit and daily-consensus test policies share this OOF vector.
"""
from common37 import *

NAME='refit_full_fit_median_person_guarded'

def main():
    s=pd.read_csv(A/f'results/oof/{NAME}.csv').set_index('id').loc[q.id].score.to_numpy()
    base=BASE['oof'];y=q.error.to_numpy();frames=q.frame_id.unique();rng=np.random.default_rng(3701)
    leave=[]
    for f in frames:
        m=q.frame_id.ne(f).to_numpy()
        leave.append(dict(removed_frame=f,delta_ap=float(AP(y[m],s[m])-AP(y[m],base[m]))))
    pd.DataFrame(leave).to_csv(A/'results/person_leave_one_frame_out.csv',index=False)
    boot=[]
    groups={f:np.flatnonzero(q.frame_id.eq(f)) for f in frames}
    dayframes={d:q.loc[q.day.eq(d),'frame_id'].unique() for d in DAYS}
    for rep in range(2000):
        # Preserve the day composition, resample whole frames within each day.
        ids=np.concatenate([groups[f] for d in DAYS for f in rng.choice(dayframes[d],len(dayframes[d]),replace=True)])
        boot.append(AP(y[ids],s[ids])-AP(y[ids],base[ids]))
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    before=np.sign(base[positive,None]-base[negative][None,:]);after=np.sign(s[positive,None]-s[negative][None,:])
    pairs=[]
    for i,j in zip(*np.nonzero(before!=after)):
        pi,ni=positive[i],negative[j]
        pairs.append(dict(error_id=q.iloc[pi].id,correct_id=q.iloc[ni].id,
            error_class=q.iloc[pi].class_name,correct_class=q.iloc[ni].class_name,
            error_frame=q.iloc[pi].frame_id,correct_frame=q.iloc[ni].frame_id,
            same_day=bool(q.iloc[pi].day==q.iloc[ni].day),before=int(before[i,j]),after=int(after[i,j]),improved=bool(after[i,j]>before[i,j])))
    pd.DataFrame(pairs).to_csv(A/'audit/person_oof_pair_changes.csv',index=False)
    ts=pd.read_csv(A/f'results/submissions/submission_{NAME}.csv').set_index('id').loc[te.id].score.to_numpy();tb=BASE['test']
    upper=np.triu(np.ones((len(te),len(te)),dtype=bool),1)
    reorders=upper&(np.sign(ts[:,None]-ts[None,:])!=np.sign(tb[:,None]-tb[None,:]))
    changed=np.abs(ts-tb)>1e-10
    out=dict(experiment=NAME,baseline=metric(q,base),candidate=metric(q,s),delta_ap=float(AP(y,s)-AP(y,base)),
        bootstrap=dict(unit='frame within day',replicates=2000,seed=3701,percentile95=np.quantile(boot,[.025,.975]).tolist(),
                       fraction_positive=float(np.mean(np.array(boot)>0)),fraction_negative=float(np.mean(np.array(boot)<0))),
        leave_one_frame_out=dict(min_delta=min(x['delta_ap'] for x in leave),max_delta=max(x['delta_ap'] for x in leave),
                                negative=sum(x['delta_ap'] < -1e-12 for x in leave),zero=sum(abs(x['delta_ap'])<=1e-12 for x in leave)),
        oof_error_correct_pairs_changed=len(pairs),oof_pairs_improved=sum(p['improved'] for p in pairs),
        oof_pairs_worsened=sum(not p['improved'] for p in pairs),
        test=dict(changed=int(changed.sum()),changed_frames=int(te.loc[changed,'frame_id'].nunique()),
                  max_abs_delta=float(abs(ts-tb).max()),pairs_reordered=int(reorders.sum()),
                  increases=int((ts-tb>1e-10).sum()),decreases=int((ts-tb< -1e-10).sum())),
        caveat='Resampling fixed selected OOF, not an unbiased confidence interval after many experiments; not a validation of final both-day fit.')
    (A/'results/person_stability.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
