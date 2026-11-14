"""Frame weighting / Ordered boosting controls for correlated small-data queries."""
from train import *
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);rows,banks,groups,_=load_bank();q,t,h=inputs();bo,bt=baseline(q,t);out=[];manifest=[]
    for conf in ['base13','base23','base23_siglip2','base23_prototypes','base23_nested','all_frozen_emb_all']:
        gr,emb=configurations()[conf];cols=list(dict.fromkeys(c for g in gr+['embed_'+e for e in emb] for c in groups[g]))+['class_name']
        for depth in [2,4]:
            for mode in ['ordered','frame_weighted','group_bagging']:
                name=f'{conf}_{mode}_d{depth}';pp=[]
                for fit in DAYS:
                    tr=rows[rows.split.eq('oof')&rows.day.eq(fit)].sort_values('frame_id');ev=rows[~rows.index.isin(tr.index)];X=banks[fit][cols];kw={};w=np.ones(len(tr))
                    if mode=='ordered':kw['boosting_type']='Ordered'
                    elif mode=='frame_weighted':w=1/tr.groupby('frame_id').id.transform('size').to_numpy();w*=len(w)/w.sum()
                    else:kw.update(bootstrap_type='Bernoulli',subsample=.8,sampling_unit='Group')
                    model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False,**kw)
                    if mode=='group_bagging':
                        from catboost import Pool
                        pool=Pool(X.loc[tr.index],tr.error,cat_features=['class_name'],group_id=tr.frame_id)
                        model.fit(pool)
                    else:model.fit(X.loc[tr.index],tr.error,cat_features=['class_name'],sample_weight=w)
                    prob=model.predict_proba(X.loc[ev.index])[:,1];model.save_model(str(A/f'models/{name}_fit{fit[-2:]}.cbm'));pp.append(ev[['id','frame_id','day','class_name','split']].assign(p_error=prob,fit_day=fit));manifest.append(dict(name=name,fit_day=fit,training_ids=tr.id.tolist(),features=cols,mode=mode,depth=depth))
                pp=pd.concat(pp);pp.to_csv(A/f'features/{name}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
                for a in [.05,.1,.2,.5,1000000.]:
                    tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);exp=name+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,model=name,mode=mode,alpha=a,**metrics(q,s)))
                pd.DataFrame(out).to_csv(A/'results/frame_experiments.csv',index=False);(A/'models/frame_manifest.json').write_text(json.dumps(manifest,indent=2));print(name,'standalone',AP(q.error,oo),'blend',pd.DataFrame(out[-5:]).iloc[:4].ap.max(),flush=True)
if __name__=='__main__':main()
