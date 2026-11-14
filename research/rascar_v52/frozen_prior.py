"""Learn CatBoost corrections to a FROZEN SAM/WeVis prior, never V31 OOF as a fit feature."""
from train import *
from scipy.special import expit,logit
from sklearn.metrics import log_loss
from catboost import Pool
def main():
    warnings.filterwarnings('ignore',category=pd.errors.PerformanceWarning);rows,banks,groups,_=load_bank();q,t,h=inputs();bo,bt=baseline(q,t);out=[];manifest=[]
    for conf in ['base23','base23_siglip2','all_frozen_emb_all']:
        gr,emb=configurations()[conf];cols=list(dict.fromkeys(c for g in gr+['embed_'+e for e in emb] for c in groups[g]))+['class_name']
        for prior in ['sam','sam_wevis']:
            for it in [100,250]:
                name=f'prior_{conf}_{prior}_i{it}';pp=[]
                for fit in DAYS:
                    tr=rows.split.eq('oof')&rows.day.eq(fit);ev=~tr;X=banks[fit][cols];s=banks[fit].sam_score.to_numpy() if prior=='sam' else banks[fit][['sam_score','wevis_score']].mean(axis=1).to_numpy();base=logit(np.clip(s,.001,.999));pool=Pool(X.loc[tr],rows.loc[tr,'error'],cat_features=['class_name'],baseline=base[tr]);model=CatBoostClassifier(iterations=it,depth=2,learning_rate=.03,l2_leaf_reg=20,random_seed=23,thread_count=3,verbose=False,allow_writing_files=False);model.fit(pool)
                    train_p=expit(model.predict(X.loc[tr],prediction_type='RawFormulaVal')+base[tr]);actual=log_loss(rows.loc[tr,'error'],train_p);reported=model.evals_result_['learn']['Logloss'][-1]
                    residual=model.predict(X.loc[ev],prediction_type='RawFormulaVal');with_prior=model.predict(Pool(X.loc[ev],cat_features=['class_name'],baseline=base[ev]),prediction_type='RawFormulaVal');assert np.max(abs(with_prior-residual-base[ev]))<1e-5
                    p=expit(model.predict(X.loc[ev],prediction_type='RawFormulaVal')+base[ev]);pp.append(rows.loc[ev,['id','frame_id','day','class_name','split']].assign(p_error=p,fit_day=fit));model.save_model(str(A/f'models/{name}_fit{fit[-2:]}.cbm'));manifest.append(dict(name=name,fit_day=fit,training_ids=rows.loc[tr,'id'].tolist(),features=cols,prior=prior,prior_frozen=True,V31_used_in_fit=False,baseline_predict_contract_error=actual-reported))
                pp=pd.concat(pp);pp.to_csv(A/f'features/{name}_predictions.csv',index=False);oo=pp[pp.split.eq('oof')].set_index('id').loc[q.id].p_error.to_numpy();tt=pp[pp.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
                for a in [.05,.1,.2,.5,1000000.]:
                    tag='standalone' if a>100 else f'a{int(a*100):02}';s=oo if a>100 else (bo+a*oo)/(1+a);st=tt if a>100 else (bt+a*tt)/(1+a);exp=name+'_'+tag;q.assign(score=s).to_csv(A/f'results/oof/{exp}.csv',index=False);t[['id']].assign(score=st).to_csv(A/f'results/submissions/{exp}.csv',index=False);out.append(dict(name=exp,model=name,alpha=a,**metrics(q,s)))
                pd.DataFrame(out).to_csv(A/'results/prior_experiments.csv',index=False);(A/'models/prior_manifest.json').write_text(json.dumps(manifest,indent=2));print(name,AP(q.error,oo),flush=True)
if __name__=='__main__':main()
