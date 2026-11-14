"""Exploratory two-model semantic absence guard. No ID rules or test targets."""
from infer import *
sys.path.insert(0,str(ROOT/'rascar_v46'))
from common46 import baseline,metrics
def main():
    q,te,h=inputs();bo,bt=baseline(q,te);dense=pd.read_csv(ROOT/'rascar_v48/features/meta_features.csv');owl=[]
    for split in ['oof','test','pool']:
        path=A/f'features/{split}.jsonl'
        if path.exists():owl.extend(dict(id=d['id'],split=split,day=d['day'],frame_id=d['frame_id'],positive_local=d['features']['positive_local']) for d in map(json.loads,path.read_text().splitlines()))
    owl=pd.DataFrame(owl).set_index('id');oo=dense[dense.role.eq('oof')].set_index('id').loc[q.id];om=q.class_name.eq('nude_hands').to_numpy()&(oo.small_control_coverage.to_numpy()<.1)&(owl.loc[q.id,'positive_local'].to_numpy()<.1);ready=set(te.id)<=set(owl.index);tm=None
    if ready:
        ts=[z.set_index('id').loc[te.id] for _,z in dense[dense.role.eq('test')].groupby('outer_fit_day')];cov=np.max([z.small_control_coverage.to_numpy() for z in ts],axis=0);tm=te.class_name.eq('nude_hands').to_numpy()&(cov<.1)&(owl.loc[te.id,'positive_local'].to_numpy()<.1)
    out=[]
    for w in [.1,.25,.5]:
        s=bo.copy();s[om]+=w*(1-bo[om]);name=f'dual_absence_hands_w{int(w*100):02}';q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
        if ready:
            t=bt.copy();t[tm]+=w*(1-bt[tm]);te[['id']].assign(score=t).to_csv(A/f'results/submissions/{name}.csv',index=False)
        out.append(dict(name=name,**metrics(q,s),flagged_oof=int(om.sum()),flagged_correct=int((om&q.error.eq(0).to_numpy()).sum()),flagged_test=int(tm.sum()) if ready else None))
    q.assign(flagged=om,coverage=oo.small_control_coverage.to_numpy(),owl_positive=owl.loc[q.id,'positive_local'].to_numpy()).to_csv(A/'audit/dual_absence_oof.csv',index=False)
    if ready:te.assign(flagged=tm,coverage=cov,owl_positive=owl.loc[te.id,'positive_local'].to_numpy(),old_score=bt).to_csv(A/'audit/dual_absence_test.csv',index=False)
    if (A/'features/pool.jsonl').exists():
        p=pd.read_csv(A/'features/pool_dense.csv');p=p[p.day.ne(p.fit_day)].copy();p['owl_positive']=p.id.map(owl.positive_local);p['flagged']=(p.coverage<.1)&(p.owl_positive<.1);p.to_csv(A/'audit/dual_absence_pool.csv',index=False);known=p[p.presence.notna()];stats=dict(labelled=len(known),true_hands=int(known.presence.eq(1).sum()),nonhands=int(known.presence.eq(0).sum()),true_hands_flagged=int((known.presence.eq(1)&known.flagged).sum()),nonhands_flagged=int((known.presence.eq(0)&known.flagged).sum()),warning='Reused stratified 36-crop presence-only diagnostic, not exact error/AP GT; several nonhands are the same pink-material motif.')
        (A/'audit/dual_absence_pool_summary.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
    pd.DataFrame(out).to_csv(A/'results/dual_absence_experiments.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':main()
