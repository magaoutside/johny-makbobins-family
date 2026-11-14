"""Quantile regression of GT geometry; CDF at official IoU cutoff gives error score."""
from common56 import *
sys.path.insert(0,str(A/'vendor'));sys.path.insert(0,str(ROOT/'rascar_v53'))
from experiment import FeatureBank
from tabicl import TabICLRegressor
from scipy.special import expit
import torch

def main():
    torch.set_num_threads(2);q,t,h=inputs();bank=FeatureBank();bo,bt=incumbent(q,t);results=[];manifest=[]
    quality=[]
    for _,r in q.iterrows():
        ov=float(iou(r[B],h.loc[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name),B]).max(initial=0))
        quality.append(min(ov,.599) if r.error else ov)
    quality=pd.Series(quality,index=q.id);alphas=np.arange(.01,1,.01);start=time.time()
    for family in ['base13','base23','anchor']:
        pred=[];name=f'tabreg_{family}'
        for fit in DAYS:
            tr=bank.rows.split.eq('oof')&bank.rows.day.eq(fit);ev=~tr;X,g=bank.build(tr,fit)
            cols=g['base13'] if family=='base13' else g['base23']+(g['gt'] if family=='anchor' else [])
            cols=cols+['class_name'];model=TabICLRegressor(n_estimators=8,model_path=str(A/'models/tabicl-regressor-v2-20260212.ckpt'),allow_auto_download=False,device='cpu',n_jobs=2,batch_size=2,random_state=5601,use_amp=False)
            model.fit(X.loc[tr,cols],quality.loc[bank.rows.loc[tr,'id']].to_numpy())
            estimates=model.predict(X.loc[ev,cols],output_type=['mean','quantiles'],alphas=alphas.tolist())
            quantiles=np.sort(estimates['quantiles'],axis=1);cdf=np.array([np.interp(.6,z,alphas,left=.005,right=.995) for z in quantiles])
            pred.append(bank.rows.loc[ev,['id','split']].assign(fit_day=fit,cdf=cdf,mean_error=expit((.6-np.clip(estimates['mean'],0,1))/.1)))
            model.save(str(A/f'models/{name}_fit{fit[-2:]}.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False)
            manifest.append(dict(name=name,fit_day=fit,columns=cols,train_ids=bank.rows.loc[tr,'id'].tolist(),target='max_GT_IoU, capped at .599 for officially unmatched duplicates',seed=5601))
            print('TABREG',family,fit,'seconds',round(time.time()-start,1),flush=True)
        d=pd.concat(pred);d.to_csv(A/f'features/{name}_predictions.csv',index=False)
        for mode in ['cdf','mean_error']:
            oo=d[d.split.eq('oof')].set_index('id').loc[q.id,mode].to_numpy();tt=d[d.split.eq('test')].groupby('id')[mode].mean().loc[t.id].to_numpy()
            stand=AP(q.error,oo);print('STANDALONE',name,mode,stand,flush=True)
            for scope in ['all','person','nude_hands']:
                om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool);tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
                for alpha in [.05,.1,.2,.4]:
                    s,z=bo.copy(),bt.copy();s[om]=(s[om]+alpha*oo[om])/(1+alpha);z[tm]=(z[tm]+alpha*tt[tm])/(1+alpha)
                    results.append(save_score(f'{name}_{mode}_{scope}_a{int(alpha*100)}',s,z,dict(family='tabreg',features=family,mode=mode,scope=scope,alpha=alpha,standalone_ap=stand)))
        pd.DataFrame(results).sort_values('ap',ascending=False).to_csv(A/'results/tabreg.csv',index=False);(A/'models/tabreg_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(results).sort_values('ap',ascending=False).head(15).to_string(index=False))

if __name__=='__main__':main()
