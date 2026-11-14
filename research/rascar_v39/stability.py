"""Conditional frame bootstrap and ranking diagnostics, not a fresh holdout."""
from common39 import *

def main():
    names=['v39_set_geometry_person_RESEARCH','v39_set_long_identity_geometry_person_RESEARCH',
           'v39_set_long_identity_majority_geometry_all_RESEARCH',
           'v39_set_long_identity_majority_geometry_all_RESEARCH_presence_guard_reference_guard']
    rng=np.random.default_rng(39001);groups=[g for _,g in q.groupby('day')];boots=[]
    for _ in range(2000):
        ids=[]
        for g in groups:
            frames=g.frame_id.unique()
            for f in rng.choice(frames,len(frames),replace=True):ids.extend(g.index[g.frame_id.eq(f)])
        boots.append(np.array(ids))
    y=q.error.to_numpy();base=BASE['oof'];base_ap=float(AP(y,base));base_boot=[float(AP(y[ix],base[ix])) for ix in boots];out=[];pair_rows=[]
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    old_pairs=np.sign(base[pos,None]-base[neg][None])
    for name in names:
        s=pd.read_csv(A/f'results/oof/{name}.csv').set_index('id').loc[q.id].score.to_numpy()
        delta=float(AP(y,s)-base_ap);boot=np.array([AP(y[ix],s[ix])-b for ix,b in zip(boots,base_boot)])
        leave=[]
        for f in q.frame_id.unique():
            take=q.frame_id.ne(f);leave.append(float(AP(y[take],s[take])-AP(y[take],base[take])))
        new_pairs=np.sign(s[pos,None]-s[neg][None]);changed=np.argwhere(new_pairs!=old_pairs);better=0;worse=0;cross=0
        for a,b in changed:
            i,j=int(pos[a]),int(neg[b]);improved=bool(new_pairs[a,b]>old_pairs[a,b]);better+=improved;worse+=not improved;cross+=q.iloc[i].day!=q.iloc[j].day
            pair_rows.append(dict(experiment=name,error_id=q.iloc[i].id,correct_id=q.iloc[j].id,improved=improved,same_day=q.iloc[i].day==q.iloc[j].day))
        out.append(dict(experiment=name,delta_ap=delta,bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),bootstrap_fraction_positive=float((boot>1e-12).mean()),leave_one_frame_delta_min=min(leave),leave_one_frame_delta_max=max(leave),improved_error_correct_pairs=better,worsened_error_correct_pairs=worse,cross_day_changed_pairs=int(cross),changed_pairs=len(changed)))
    (A/'results/stability.json').write_text(json.dumps(dict(bootstrap_samples=2000,unit='whole frame resampled within each day',warning='Conditional development-set uncertainty only. Does not account for model/rule selection, training variability, repeated use of validation, or unseen-day shift. Pair counts are not AP increments.',experiments=out),indent=2));pd.DataFrame(pair_rows).to_csv(A/'audit/changed_ranking_pairs.csv',index=False)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
