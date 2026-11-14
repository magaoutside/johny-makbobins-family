"""Fixed seed averages; frame-clustered uncertainty and exact ranking changes."""
from common56 import *

def robust(q,base,new,n=3000):
    rng=np.random.default_rng(56056);y=q.error.to_numpy();days=[]
    for day in DAYS:
        part=q[q.day.eq(day)];days.append([np.flatnonzero(q.frame_id.eq(f)) for f in part.frame_id.unique()])
    dif=[]
    for _ in range(n):
        ix=np.concatenate([np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))]) for groups in days])
        if len(np.unique(y[ix]))==2:dif.append(AP(y[ix],new[ix])-AP(y[ix],base[ix]))
    loo=[]
    for frame in q.frame_id.unique():
        m=~q.frame_id.eq(frame).to_numpy();loo.append(dict(frame=frame,delta=AP(y[m],new[m])-AP(y[m],base[m])))
    pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
    old=base[pos,None]<=base[None,neg];now=new[pos,None]<=new[None,neg]
    pair=[]
    for a,b in zip(*np.where(old!=now)):
        i,j=pos[a],neg[b];pair.append(dict(error_id=q.iloc[i].id,correct_id=q.iloc[j].id,error_frame=q.iloc[i].frame_id,correct_frame=q.iloc[j].frame_id,error_class=q.iloc[i].class_name,correct_class=q.iloc[j].class_name,fixed=bool(old[a,b]),error_before=base[i],error_after=new[i],correct_before=base[j],correct_after=new[j]))
    return dict(bootstrap_n=n,bootstrap_95=np.quantile(dif,[.025,.975]).tolist(),bootstrap_positive_fraction=float(np.mean(np.array(dif)>0)),loo_min=min(x['delta'] for x in loo),loo_max=max(x['delta'] for x in loo),loo_positive=sum(x['delta']>0 for x in loo),loo_frames=len(loo),pairs_fixed=int((old&~now).sum()),pairs_broken=int((~old&now).sum()),wrong_pairs_before=int(old.sum()),wrong_pairs_after=int(now.sum())),pd.DataFrame(loo),pd.DataFrame(pair)

def predictions(names,q,t):
    oo=[];tt=[]
    for name in names:
        d=pd.read_csv(A/f'features/{name}_predictions.csv')
        assert d[d.split.eq('oof')].id.nunique()==len(q)
        oo.append(d[d.split.eq('oof')].set_index('id').loc[q.id,'p_error'].to_numpy())
        tt.append(d[d.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy())
    return np.mean(oo,axis=0),np.mean(tt,axis=0)

def main():
    q,t,h=inputs();bo,bt=incumbent(q,t);records=[];manifest=[]
    for family in ['anchor','base13']:
        names=[f'tabiclv1_{family}_n8_s5601',f'tabiclv1_{family}_n16_s5602',f'tabiclv1_{family}_n16_s5603']
        oo,tt=predictions(names,q,t)
        pd.concat([q[['id']].assign(split='oof',p_error=oo),t[['id']].assign(split='test',p_error=tt)]).to_csv(A/f'features/v1_{family}_mean3.csv',index=False)
        for scope in ['all','person','nude_hands']:
            om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool);tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
            for alpha in [.1,.2,.4]:
                s,z=bo.copy(),bt.copy();s[om]=(s[om]+alpha*oo[om])/(1+alpha);z[tm]=(z[tm]+alpha*tt[tm])/(1+alpha)
                for rank in [False,True] if scope!='all' else [False]:
                    sn,zn=s.copy(),z.copy();name=f'v1_{family}_mean3_{scope}_a{int(alpha*100)}'+('_rank' if rank else '')
                    if rank:
                        oi=np.flatnonzero(om);ti=np.flatnonzero(tm)
                        sn[oi[np.argsort(s[om],kind='stable')]]=np.sort(bo[om]);zn[ti[np.argsort(z[tm],kind='stable')]]=np.sort(bt[tm])
                    record=save_score(name,sn,zn,dict(family=family,scope=scope,alpha=alpha,rank=rank))
                    if family=='anchor' and scope=='nude_hands':
                        stats,loo,pairs=robust(q,bo,sn);record.update(stats)
                        loo.to_csv(A/f'audit/{name}_loo.csv',index=False);pairs.to_csv(A/f'audit/{name}_pairs.csv',index=False)
                        t.assign(before=bt,after=zn,delta=zn-bt).sort_values('delta',key=abs,ascending=False).to_csv(A/f'audit/{name}_test_changes.csv',index=False)
                    records.append(record)
        manifest.append(dict(family=family,models=names,aggregation='equal arithmetic mean of 3 seeds; test averages 2 day fits'))
    pd.DataFrame(records).sort_values('ap',ascending=False).to_csv(A/'results/ensemble_v1.csv',index=False)
    (A/'audit/ensemble_v1_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(records).sort_values('ap',ascending=False).head(18).to_string(index=False))

if __name__=='__main__':main()
