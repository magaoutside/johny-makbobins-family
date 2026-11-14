"""Use V33 heads only when object identity and geometry agree; rebase on V31.

No absence signal and no test labels. A low-quality estimate is allowed to
increase suspicion only when the head's optimistic IoU remains below .6.
"""
from experiments import *

def evidence(rows, split, fit):
    ds=[align(ROOT/f'rascar_v33/features/side_{kind}_fit{fit[-2:]}.csv',rows)
        for kind in ['pool_regional_balanced','pool_regional_balanced_rare']]
    refs=[d[['ref_'+c for c in B]].to_numpy(float) for d in ds]
    mutual=np.array([iou(a,b)[0,0] for a,b in zip(*refs)])
    valid=np.logical_and.reduce([d.valid.to_numpy(bool)&d.presence.ge(.9).to_numpy() for d in ds])
    valid &= mutual>=.65
    allowed=np.ones(len(rows),bool)
    for cls in CL:
        ids=np.flatnonzero(rows.class_name.eq(cls).to_numpy())
        if len(ids): allowed[ids]=guard(split,rows.iloc[ids],cls)
    good=valid & np.logical_and.reduce([d.ref_iou.ge(.65).to_numpy() for d in ds]) & allowed
    bad=valid & np.logical_and.reduce([d.optimistic_iou.lt(.6).to_numpy() for d in ds])
    # A mask/side-head identity check prevents averaging different objects.
    tail='oof' if split=='oof' else f'test_fit{fit[-2:]}'
    mask=align(ROOT/f'rascar_v31/features/linear_mask_cut0.5_{tail}_evidence.csv',rows)
    mr=mask[['ref_'+c for c in B]].to_numpy(float)
    agree=np.logical_and.reduce([np.array([iou(a,b)[0,0]>=.5 for a,b in zip(r,mr)]) for r in refs])
    return ds,refs,good,bad,agree

def main():
    signals={};ledger=[]
    for split,rows in ROWS.items():
        out={}
        for source in ['boundary','quality']:
            for strict in [False,True]:
                delta=np.zeros(len(rows))
                fits=DAYS
                if split=='oof':
                    for fit in fits:
                        ix=np.flatnonzero(rows.day.ne(fit).to_numpy());ds,refs,good,bad,agree=evidence(rows.iloc[ix],split,fit)
                        if strict:good&=agree;bad&=agree
                        vals=np.stack([np.clip((.85-d.ref_iou.to_numpy())/.5,0,1) if source=='boundary'
                                       else 1-d.predicted_quality.to_numpy() for d in ds])
                        diff=vals-BASE[split][None,ix]
                        inc=np.maximum(diff.min(0),0)*bad;dec=np.minimum(diff.max(0),0)*good
                        delta[ix]=inc+dec
                else:
                    vals=[];goods=[];bads=[];allrefs=[]
                    for fit in fits:
                        ds,refs,good,bad,agree=evidence(rows,split,fit)
                        if strict:good&=agree;bad&=agree
                        goods.append(good);bads.append(bad);allrefs+=refs
                        vals += [np.clip((.85-d.ref_iou.to_numpy())/.5,0,1) if source=='boundary'
                                 else 1-d.predicted_quality.to_numpy() for d in ds]
                    agree=np.ones(len(rows),bool)
                    for i in range(len(allrefs)):
                        for j in range(i):agree&=np.array([iou(a,b)[0,0]>=.65 for a,b in zip(allrefs[i],allrefs[j])])
                    good=np.logical_and.reduce(goods)&agree;bad=np.logical_and.reduce(bads)&agree
                    diff=np.stack(vals)-BASE[split][None,:]
                    delta=np.maximum(diff.min(0),0)*bad+np.minimum(diff.max(0),0)*good
                out[(source,strict)]=delta
                for i in np.flatnonzero(abs(delta)>1e-10):
                    ledger.append(dict(split=split,id=rows.iloc[i].id,class_name=rows.iloc[i].class_name,
                                       source=source,strict=strict,raw_delta=float(delta[i])))
        signals[split]=out
    results=[]
    for source in ['boundary','quality']:
        for strict in [False,True]:
            for cls in ['person','nude_hands','cup']:
                for weight in [.1,.25]:
                    scores={split:np.clip(BASE[split]+weight*signals[split][(source,strict)]*
                                          rows.class_name.eq(cls).to_numpy(),0,1) for split,rows in ROWS.items()}
                    results.append(save(f'{cls}_{source}_identity{int(strict)}_w{weight}',scores,'boundary_expert',
                      'Two same-day heads on OOF, four on test. Presence>=.9, mutual IoU>=.65; penalty needs optimistic IoU<.6; rescue needs query IoU>=.65 and old semantic/duplicate guard.'))
    pd.DataFrame(results).to_csv(A/'results/boundary_experiments.csv',index=False)
    pd.DataFrame(ledger).to_csv(A/'audit/boundary_signal.csv',index=False)

if __name__=='__main__':main()
