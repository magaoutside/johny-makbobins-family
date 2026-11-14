"""Native CatBoost vector features and direct ranking objective controls."""
from train import *
from catboost import CatBoostRanker
from scipy.special import expit
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);rows,banks,groups,transforms=load_bank();q,te,h=inputs();bo,bt=baseline(q,te);results=[];manifests=[]
    configurations=[]
    for e in EMB:configurations.append((f'native_{e}',['base23'],[e],'roi'))
    configurations += [('native_all',['base23'],EMB,'roi'),('native_all_context',['base23'],EMB,'context'),('native_siglip_dino',['base23'],['siglip2','dino_small'],'roi'),('native_clip_dino',['base23'],['clip','vit_l448'],'roi'),('native_only_all',[],EMB,'roi')]
    tasks=[(name,gr,emb,view,'native') for name,gr,emb,view in configurations]
    for name in ['base13','base23','base23_siglip2','base23_nested','all_frozen']:
        gr,emb=globals()['configurations']()[name]
        for loss in ['pair','map']:tasks.append((name+'_'+loss,gr,emb,'pca',loss))
    for label,gr,emb,view,kind in tasks:
        for depth in [2,4]:
            name=f'{label}_d{depth}';pp=[]
            for fit in DAYS:
                tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;cols=list(dict.fromkeys(c for g in gr for c in groups[g]));X=banks[fit][cols+['class_name']].copy();ec=[]
                for e in emb:
                    if kind=='native':
                        x=np.load(A/f'features/{e}.npy');dim=x.shape[1]//3;v=x[:,dim:2*dim] if view=='roi' else x[:,:dim];v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);X[e+'_vector']=list(v);ec.append(e+'_vector')
                    else:
                        for c in groups['embed_'+e]:X[c]=banks[fit][c]
                if kind=='native':
                    model=CatBoostClassifier(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr],rows.loc[tr,'error'],cat_features=['class_name'],embedding_features=ec);prob=model.predict_proba(X.loc[ev])[:,1]
                else:
                    loss='PairLogit' if kind=='pair' else 'YetiRank:mode=MAP';model=CatBoostRanker(iterations=250 if depth==2 else 400,depth=depth,learning_rate=.03,l2_leaf_reg=10 if depth==2 else 20,loss_function=loss,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(X.loc[tr],rows.loc[tr,'error'],cat_features=['class_name'],group_id=np.zeros(int(tr.sum()),dtype=int));prob=expit(model.predict(X.loc[ev]))
                model.save_model(str(A/f'models/{name}_fit{fit[-2:]}.cbm'));pp.append(rows.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=prob,fit_day=fit));manifests.append(dict(model=name,fit_day=fit,features=X.columns.tolist(),embedding_columns=ec,kind=kind,training_ids=rows.loc[tr,'id'].tolist(),held_day_in_fit=False,test_in_fit=False,ranking_group='all training-day queries' if kind!='native' else None,rank_blend_transform='fixed sigmoid, not calibrated probability' if kind!='native' else None))
            pp=pd.concat(pp);pp.to_csv(A/f'features/{name}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=np.stack([g.set_index('id').loc[te.id].p_error.to_numpy() for _,g in pp[pp.split.eq('test')].groupby('fit_day')]).mean(0)
            for a in [.05,.1,.2,.5,1000000.]:
                tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);t=tt if a>100 else (bt+a*tt)/(1+a);exp=name+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);te[['id']].assign(score=t).to_csv(A/f'results/submissions/{exp}.csv',index=False);results.append(dict(name=exp,model=name,depth=depth,kind=kind,alpha=a,**metrics(q,s)))
            pd.DataFrame(results).to_csv(A/'results/native_rank_experiments.csv',index=False);(A/'models/native_rank_manifest.json').write_text(json.dumps(manifests,indent=2));print(name,'standalone',AP(q.error,oo),'blend',pd.DataFrame(results[-5:]).iloc[:4].ap.max(),flush=True)
    print(pd.DataFrame(results).sort_values('ap',ascending=False).head(15).to_string(index=False))
if __name__=='__main__':main()
