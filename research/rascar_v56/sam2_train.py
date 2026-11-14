"""Fold-clean calibration of new frozen SAM2 features, no in-sample stacked scores."""
from common56 import *
sys.path.insert(0,str(A/'vendor'));sys.path.insert(0,str(ROOT/'rascar_v53'))
from experiment import FeatureBank
from tabicl import TabICLClassifier
from catboost import CatBoostClassifier
import torch,argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',default=['full_base','full_anchor','crop_base','crop_anchor','both_base','both_anchor']);args=p.parse_args()
    torch.set_num_threads(2);q,t,h=inputs();bank=FeatureBank();bo,bt=incumbent(q,t);results=[];manifest=[];start=time.time()
    new={}
    for mode in ['full','crop']:
        path=A/f'features/sam2_{mode}.csv'
        if path.exists():
            d=pd.read_csv(path).set_index('id')
            if len(d)==len(bank.rows):new[mode]=d.loc[bank.rows.id].reset_index(drop=True)
    for family in args.families:
        view,base=family.split('_');modes=['full','crop'] if view=='both' else [view]
        if any(m not in new for m in modes):continue
        sam=pd.concat([new[m] for m in modes],axis=1)
        if view=='both' and base=='anchor':sam=sam[[c for c in sam if not any(x in c for x in ['_mean','_std','logit','uncertainty','components','empty'])]]
        for algo in ['cb2','cb4','icl1']:
            name=f'sam2_{family}_{algo}';pred=[]
            for fit in DAYS:
                use=bank.rows.split.eq('oof')&bank.rows.day.eq(fit);ev=~use;X,g=bank.build(use,fit)
                cols=g['base13'] if base=='base' else g['base23']+g['gt']
                X=pd.concat([X[cols+['class_name']].reset_index(drop=True),sam],axis=1).fillna(0);cols=X.columns.tolist();ps=[]
                for seed in ([5601] if algo=='icl1' else [5601,5602,5603]):
                    if algo=='icl1':model=TabICLClassifier(n_estimators=16,model_path=str(A/'models/tabicl-classifier-v1.1-20250506.ckpt'),allow_auto_download=False,device='cpu',n_jobs=2,batch_size=2,random_state=seed,use_amp=False)
                    else:model=CatBoostClassifier(iterations=250,depth=int(algo[-1]),learning_rate=.03,l2_leaf_reg=10,random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False,cat_features=['class_name'])
                    model.fit(X.loc[use],bank.rows.loc[use,'error'].astype(int));ps.append(model.predict_proba(X.loc[ev])[:,1])
                    path=A/f'models/{name}_s{seed}_fit{fit[-2:]}'
                    if algo=='icl1':model.save(str(path)+'.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=False)
                    else:model.save_model(str(path)+'.cbm')
                pred.append(bank.rows.loc[ev,['id','split']].assign(fit_day=fit,p_error=np.mean(ps,axis=0)))
                manifest.append(dict(name=name,fit_day=fit,cols=cols,train_ids=bank.rows.loc[use,'id'].tolist()))
            d=pd.concat(pred);d.to_csv(A/f'features/{name}_predictions.csv',index=False)
            oo=d[d.split.eq('oof')].set_index('id').loc[q.id,'p_error'].to_numpy();tt=d[d.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy();stand=AP(q.error,oo)
            for scope in ['all','person','nude_hands','cup']:
                om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool);tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
                for alpha in [.1,.2,.4]:
                    s,z=bo.copy(),bt.copy();s[om]=(s[om]+alpha*oo[om])/(1+alpha);z[tm]=(z[tm]+alpha*tt[tm])/(1+alpha)
                    results.append(save_score(f'{name}_{scope}_a{int(alpha*100)}',s,z,dict(family=family,algo=algo,scope=scope,alpha=alpha,standalone_ap=stand)))
            print('SAM2FIT',name,'standalone',stand,'best',max(x['ap'] for x in results[-12:]),'seconds',round(time.time()-start,1),flush=True)
            pd.DataFrame(results).sort_values('ap',ascending=False).to_csv(A/f'results/sam2_{args.families[0]}.csv',index=False)
            (A/f'models/sam2_manifest_{args.families[0]}.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(results).sort_values('ap',ascending=False).head(15).to_string(index=False))

if __name__=='__main__':main()
