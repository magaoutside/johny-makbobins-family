"""Fixed V31 geometry recipe, and fixed 25/50% interpolation with V31."""
from common46 import *
from sklearn.metrics import average_precision_score as AP
def main():
    q,te,h=inputs();bo,bt=baseline(q,te);bq,bte=v22base(q,te);results=[]
    for path in sorted((A/'features').glob('*_evidence.csv')):
        key=path.stem.removesuffix('_evidence');f=pd.read_csv(path)
        o=f[f.split.eq('oof')].set_index('id').loc[q.id];t=[g.set_index('id').loc[te.id] for _,g in f[f.split.eq('test')].groupby('fit_day')]
        oo=compose(bq,o.overlap.to_numpy());tt=compose(bte,np.stack([g.overlap for g in t]))
        for scope in ['all']+CL:
            om=np.ones(len(q),bool) if scope=='all' else q.class_name.eq(scope).to_numpy();tm=np.ones(len(te),bool) if scope=='all' else te.class_name.eq(scope).to_numpy()
            for weight in [.25,.5,1.]:
                so=bo.copy();st=bt.copy();so[om]+=weight*(oo[om]-bo[om]);st[tm]+=weight*(tt[tm]-bt[tm]);name=f'{key}_{scope}_mix{int(weight*100)}'
                q.assign(score=so).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False)
                e=q.error.to_numpy()==1;n=~e;before=bo[e,None]<bo[None,n];after=so[e,None]<so[None,n]
                results.append(dict(name=name,**metrics(q,so),delta=float(AP(q.error,so)-AP(q.error,bo)),repaired=int((before&~after).sum()),broken=int((~before&after).sum()),changed_oof=int((abs(so-bo)>1e-9).sum()),changed_test=int((abs(st-bt)>1e-9).sum())))
        if key=='small_control_s3101':
            assert np.max(abs(oo-bo))<1e-12,np.max(abs(oo-bo));assert np.max(abs(tt-bt))<1e-12,np.max(abs(tt-bt))
    df=pd.DataFrame(results).sort_values('ap',ascending=False);df.to_csv(A/'results/experiments.csv',index=False);print(df.to_string(index=False))
if __name__=='__main__':main()
