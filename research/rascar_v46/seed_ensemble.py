"""Equal mean over three initialization seeds; tray scope chosen post-hoc."""
from common46 import *
from train import decode
def main():
    rec=records().set_index('id');rows=[]
    for fit in DAYS:
        data=[np.load(A/f'features/siglip2_s{s}_fit{fit[-2:]}_maps.npz') for s in [3101,3102,3103]];ids=data[0]['ids'];assert all(np.array_equal(d['ids'],ids) for d in data);probs=np.mean([d['maps'].astype(np.float32) for d in data],axis=0)
        for j,ident in enumerate(ids):
            r=rec.loc[ident].copy();r['id']=ident
            if r.class_name!='tray':continue
            rows.append(dict(id=ident,frame_id=r.frame_id,day=r.day,fit_day=fit,split=r.split,source='query' if r.split=='oof' else 'test',class_name=r.class_name,**decode(probs[j],r)))
    f=pd.DataFrame(rows);f.to_csv(A/'features/siglip2_seedmean_tray.csv',index=False);q,te,h=inputs();bo,bt=baseline(q,te);bq,bte=v22base(q,te);o=f[f.split.eq('oof')].set_index('id').reindex(q.id);t=[g.set_index('id').reindex(te.id) for _,g in f[f.split.eq('test')].groupby('fit_day')];so=compose(bq,o.overlap.to_numpy());st=compose(bte,np.stack([g.overlap for g in t]));out=[]
    for weight in [.25,.5,1.]:
        om=q.class_name.eq('tray');tm=te.class_name.eq('tray');s=bo.copy();tt=bt.copy();s[om]+=weight*(so[om]-bo[om]);tt[tm]+=weight*(st[tm]-bt[tm]);name=f'siglip2_seedmean_tray_mix{int(weight*100)}';q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False);te[['id']].assign(score=tt).to_csv(A/f'results/submissions/{name}.csv',index=False);out.append(dict(name=name,**metrics(q,s),changed_test=int((abs(tt-bt)>1e-9).sum())))
    pd.DataFrame(out).to_csv(A/'results/seed_ensemble_experiments.csv',index=False);print(pd.DataFrame(out).to_string(index=False))
if __name__=='__main__':main()
