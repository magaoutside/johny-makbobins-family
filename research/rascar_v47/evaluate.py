from point_instance import *
from sklearn.metrics import average_precision_score as AP
def main():
    q,te,h=inputs();bo,bt=baseline(q,te);bq,bte=v22base(q,te);results=[]
    for path in (A/'features').glob('*_evidence.csv'):
        f=pd.read_csv(path);key=path.stem.removesuffix('_evidence');o=f[f.split.eq('oof')].set_index('id').reindex(q.id);tt=[g.set_index('id').reindex(te.id) for _,g in f[f.split.eq('test')].groupby('fit_day')]
        ov=o.overlap.to_numpy();tv=np.stack([g.overlap.to_numpy() for g in tt]);oo=compose(bq,[ov]);tt=compose(bte,tv)
        targets=np.clip((.85-tv)/.5,0,1);valid_test=np.isfinite(tv).all(0)&(((targets>=bte).all(0))|((targets<=bte).all(0)))
        for cls in ['all','person','nude_hands']:
            om=q.class_name.isin(CLASS).to_numpy() if cls=='all' else q.class_name.eq(cls).to_numpy();tm=te.class_name.isin(CLASS).to_numpy() if cls=='all' else te.class_name.eq(cls).to_numpy()
            om&=np.isfinite(ov);tm&=valid_test
            for weight in [.25,.5,1.]:
                s=bo.copy();t=bt.copy();s[om]+=weight*(oo[om]-bo[om]);t[tm]+=weight*(tt[tm]-bt[tm]);name=f'{key}_{cls}_mix{int(weight*100)}'
                q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/{name}.csv',index=False);results.append(dict(name=name,**metrics(q,s),delta=AP(q.error,s)-AP(q.error,bo),changed_test=int((abs(t-bt)>1e-9).sum())))
    df=pd.DataFrame(results).sort_values('ap',ascending=False);df.to_csv(A/'results/experiments.csv',index=False);print(df.to_string(index=False))
if __name__=='__main__':main()
