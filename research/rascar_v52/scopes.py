"""Post-hoc ablation: class-scoped correction versus class-scale preservation."""
from train import *
def main():
    q,t,h=inputs();bo,bt=baseline(q,t);out=[]
    for stem in ['ensemble_base23_siglip2_cb2','ensemble_base23_prototypes_cb4','ensemble_base23_text_cb2','ensemble_semantic_prototypes_cb2','base23_nested_cb2']:
        p=pd.read_csv(A/f'features/{stem}_predictions.csv');oo=p[p.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy();tt=p[p.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
        for scope in ['all']+CL:
            om=np.ones(len(q),bool) if scope=='all' else q.class_name.eq(scope).to_numpy();tm=np.ones(len(t),bool) if scope=='all' else t.class_name.eq(scope).to_numpy()
            for alpha in [.1,.2]:
                s=bo.copy();z=bt.copy();s[om]=(bo[om]+alpha*oo[om])/(1+alpha);z[tm]=(bt[tm]+alpha*tt[tm])/(1+alpha)
                for preserve in [False,True]:
                    so=s.copy();st=z.copy()
                    if preserve:
                        for rows,score,base in [(q,so,bo),(t,st,bt)]:
                            for c in CL if scope=='all' else [scope]:
                                ix=np.flatnonzero(rows.class_name.eq(c));score[ix[np.argsort(score[ix],kind='stable')]]=np.sort(base[ix])
                    name=f'scope_{stem}_{scope}_a{int(alpha*100)}'+('_scale_preserved' if preserve else '');q.assign(score=so).to_csv(A/f'results/oof/{name}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{name}.csv',index=False);out.append(dict(name=name,scope=scope,alpha=alpha,preserve=preserve,**metrics(q,so),changed_test=int((abs(st-bt)>1e-9).sum())))
    pd.DataFrame(out).to_csv(A/'results/scoped_experiments.csv',index=False);print(pd.DataFrame(out).sort_values('ap',ascending=False).head(20).to_string(index=False))
if __name__=='__main__':main()
