"""Post-hoc diagnostic: retain V31's person-score marginal, change its order.

One label-free monotone quantile transfer per population (OOF or test), no
per-day/category threshold fitting or test labels. Exploratory, not independent.
"""
from common45 import *
q,te,h=inputs();bo,bt=baseline(q,te);summary=[]
for model in ['logreg','cross_class_rank']:
    name=f'nested_{model}_person_w10';out=[]
    for split,rows,base in [('oof',q,bo),('submissions',te,bt)]:
        score=pd.read_csv(A/f'results/{split}/{name}.csv').set_index('id').loc[rows.id,'score'].to_numpy();ix=np.flatnonzero(rows.class_name.eq('person'));new=base.copy();order=ix[np.argsort(score[ix],kind='stable')];new[order]=np.sort(base[ix]);out.append(new)
        frame=rows.assign(score=new) if split=='oof' else rows[['id']].assign(score=new);frame.to_csv(A/f'results/{split}/{name}_scale_preserved.csv',index=False)
    oo,tt=out;p=q.error.to_numpy()==1;n=~p;before=bo[p,None]<bo[None,n];after=oo[p,None]<oo[None,n]
    summary.append(dict(name=name+'_scale_preserved',**metrics(q,oo),repaired_pairs=int((before&~after).sum()),broken_pairs=int((~before&after).sum()),changed_oof=int((abs(oo-bo)>1e-10).sum()),changed_test=int((abs(tt-bt)>1e-10).sum()),note='Post-hoc label-free marginal preservation; repeated OOF; test GT not used'))
pd.DataFrame(summary).to_csv(A/'results/scale_experiments.csv',index=False);print(pd.DataFrame(summary).to_string(index=False))
