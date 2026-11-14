from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v43'))
from common43 import *
A=ROOT/'rascar_v44'
q,te,h=inputs();bo,bt=baseline(q,te)
def aligned(path,rows):return pd.read_csv(path).set_index('id').loc[rows.id,'score'].to_numpy()
baseq=aligned(ROOT/'rascar_v22/results/oof/final_reviewed.csv',q)
baseq-=aligned(ROOT/'rascar_v18/results/oof/release_main.csv',q)-aligned(ROOT/'rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q)
baseq=np.clip(baseq,0,1);baset=aligned(ROOT/'rascar_v22/submission_v22_candidate.csv',te)
summ=[]
def save(name,oo,tt,notes):
    assert np.isfinite(oo).all() and np.isfinite(tt).all() and ((tt>=0)&(tt<=1)).all()
    assert np.array_equal(tt[te.class_name.ne('nude_hands')],bt[te.class_name.ne('nude_hands')])
    q.assign(score=oo).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=tt).to_csv(A/f'results/submissions/{name}.csv',index=False)
    row=dict(name=name,**metrics(q,oo),notes=notes)
    for label,base,v in [('oof',bo,oo),('test',bt,tt)]:
        row['changed_'+label]=int((abs(v-base)>1e-10).sum());ix=np.argsort(-base,kind='stable');sv=v[ix];row['inversions_'+label]=int(np.triu(sv[:,None]<sv[None,:],1).sum())
    summ.append(row)
save('V31',bo,bt,'Confirmed Public .96239')
for variant in ['control','silver']:
    path=A/f'results/{variant}_predictions.csv'
    if not path.exists():continue
    d=pd.read_csv(path)
    if d.fit_day.nunique()!=2:continue
    o=d[d.split.eq('oof')&d.day.ne(d.fit_day)].set_index('id').reindex(q.id);ts=[g.set_index('id').reindex(te.id) for _,g in d[d.split.eq('test')].groupby('fit_day')]
    ov=o.overlap.to_numpy();ot=np.stack([t.overlap.to_numpy() for t in ts]);op=np.clip((.85-ov)/.5,0,1);tp=np.clip((.85-ot)/.5,0,1)
    om=np.isfinite(op)&q.class_name.eq('nude_hands').to_numpy()
    # Identical V31 recipe: replace only the hand dense component at weight .25.
    dd=tp-baset[None];gate=np.isfinite(tp).all(0)&((dd>=0).all(0)|(dd<=0).all(0))&te.class_name.eq('nude_hands').to_numpy()
    conservative=np.where((dd>=0).all(0),np.min(tp,0),np.max(tp,0))
    so=bo.copy();st=bt.copy();hand=q.class_name.eq('nude_hands').to_numpy();ht=te.class_name.eq('nude_hands').to_numpy();so[hand]=baseq[hand];st[ht]=baset[ht]
    so[om]=baseq[om]+.25*(op[om]-baseq[om]);st[gate]=baset[gate]+.25*(conservative[gate]-baset[gate])
    save(f'{variant}_replace_hand',so,st,'Original V31 rule and .25 weight; only its hand head replaced')
    for fraction in [.25,.50]:save(f'{variant}_blend_replace{int(fraction*100)}',bo+fraction*(so-bo),bt+fraction*(st-bt),'Conservative fraction of head replacement; OOF and test use same fraction')
    if variant=='silver':
        guarded_o=bo.copy();guarded_t=bt.copy()
        guarded_o[om]=so[om];guarded_t[gate]=st[gate]
        save('silver_supported_blend50',bo+.5*(guarded_o-bo),bt+.5*(guarded_t-bt),'Only update where new reference is available; both test branches must agree. Missing or conflicting new reference keeps V31, instead of removing old dense evidence.')
        values=[]
        for df,base,s in [(q,bo,so),(te,bt,st)]:
            v=base.copy()
            for day in df.day.unique():
                ix=np.flatnonzero(df.day.eq(day)&df.class_name.eq('nude_hands'))
                # No labels: preserve each day's existing hand-score distribution,
                # changing only within-class order. Added after first diagnostics.
                order=ix[np.argsort(s[ix],kind='stable')];v[order]=np.sort(base[ix])
            values.append(v)
        save('silver_rank_preserved_EXPLORATORY',*values,'Post-hoc unsupervised day/class quantile transfer; explicitly exploratory')
pd.DataFrame(summ).to_csv(A/'results/experiments.csv',index=False)
print(pd.DataFrame(summ)[['name','ap','2026-10-25','2026-10-26','nude_hands','changed_test','inversions_test']].to_string(index=False))
