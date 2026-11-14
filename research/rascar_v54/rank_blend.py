"""Change within-class ordering while preserving incumbent class score scales.

The map uses predictions only, separately for the OOF pool and test pool. It never
uses labels, and the exact same postprocessing is applied to augmentation controls.
"""
from common54 import *
from sklearn.metrics import average_precision_score as AP


def reorder(base, auxiliary, mask, alpha):
    out=base.copy(); ix=np.flatnonzero(mask)
    order=np.argsort(base[ix]+alpha*auxiliary[ix],kind='stable')
    out[ix[order]]=np.sort(base[ix])
    return out


def main():
    q,t,h=inputs();bo,bt=incumbent(q,t);results=[]
    names=['visual_d4_all_w000_actual_confirm','visual_d4_all_w100_actual_confirm',
           'all_d4_all_w000_actual_confirm','all_d4_no_gt_w100_actual_confirm',
           'visual_d4_all_w100_repeat_parent_controls','all_d4_no_gt_w100_repeat_parent_controls']
    for model in names:
        p=pd.read_csv(A/f'features/{model}_predictions.csv')
        oo=p[p.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
        tt=p[p.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
        for scope in ['person','nude_hands']:
            om=q.class_name.eq(scope).to_numpy();tm=t.class_name.eq(scope).to_numpy()
            for alpha in [.05,.1,.2]:
                s=reorder(bo,oo,om,alpha);z=reorder(bt,tt,tm,alpha)
                name=f'rank_{model}_{scope}_a{int(alpha*100):02}'
                q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
                t[['id']].assign(score=z).to_csv(A/f'results/submissions/{name}.csv',index=False)
                results.append(dict(name=name,model=model,scope=scope,alpha=alpha,**metrics(q,s),delta=AP(q.error,s)-AP(q.error,bo),
                                    changed_oof=int((abs(s-bo)>1e-10).sum()),changed_test=int((abs(z-bt)>1e-10).sum())))
    d=pd.DataFrame(results).sort_values('ap',ascending=False);d.to_csv(A/'results/rank_experiments.csv',index=False)
    print(d.to_string(index=False))


if __name__=='__main__':main()
