from common45 import *
q,te,h=inputs();bo,bt=baseline(q,te);bq,bte=v22base(q,te)
d=pd.read_csv(A/'features/decomposition.csv')
old={s:[pd.read_csv(ROOT/f'rascar_v31/features/linear_mask_cut0.5_{tail}_evidence.csv').set_index('id').loc[rows.id].overlap.to_numpy() for tail in tails] for s,rows,tails in [('oof',q,['oof']),('test',te,['test_fit25','test_fit26'])]}
assert np.max(abs(compose(bq,old['oof'])-bo))<1e-12
assert np.max(abs(compose(bte,old['test'])-bt))<1e-12
summary=[]
def save(name,oo,tt,notes):
    q.assign(score=oo).to_csv(A/f'results/oof/{name}.csv',index=False)
    te[['id']].assign(score=tt).to_csv(A/f'results/submissions/{name}.csv',index=False)
    p=q.error.to_numpy()==1;n=~p
    before=bo[p,None]<bo[None,n];after=oo[p,None]<oo[None,n]
    row=dict(name=name,**metrics(q,oo),delta=float(metrics(q,oo)['ap']-metrics(q,bo)['ap']),changed_oof=int((abs(oo-bo)>1e-10).sum()),changed_test=int((abs(tt-bt)>1e-10).sum()),repaired_pairs=int((before&~after).sum()),broken_pairs=int((~before&after).sum()),notes=notes)
    summary.append(row)
save('V31',bo,bt,'Confirmed Public .96239')
for method in d.method.unique():
    sub=d[d.method.eq(method)]
    for scope in ['person','nude_hands','both']:
        values={};pred=[]
        for split,rows,base,current in [('oof',q,bq,bo),('test',te,bte,bt)]:
            revised=[]
            for k,ov in enumerate(old[split]):
                f=sub[sub.split.eq(split)]
                if split=='test':f=f[f.fit_day.eq(DAYS[k])]
                f=f.set_index('id').reindex(rows.id);gate=f.modified.eq(True).to_numpy()
                if scope!='both':gate&=rows.class_name.eq(scope).to_numpy()
                new=ov.copy();new[gate]=f.overlap.to_numpy()[gate];revised.append(new)
            values[split]=compose(base,revised)
        for weight in [.10,.25]:
            # Fraction of original dense replacement; keep same V31 formula.
            fraction=weight/.25
            save(f'{method}_{scope}_w{int(weight*100):02}',bo+fraction*(values['oof']-bo),bt+fraction*(values['test']-bt),'Only independently anchored components split; no query coordinates in decoder optimization')
pd.DataFrame(summary).to_csv(A/'results/experiments.csv',index=False)
print(pd.DataFrame(summary).sort_values('ap',ascending=False).to_string(index=False))
