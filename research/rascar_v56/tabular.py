"""New pretrained tabular inference, on fixed cross-day feature banks."""
from common56 import *
sys.path.insert(0,str(A/'vendor'))
sys.path.insert(0,str(ROOT/'rascar_v53'))
from experiment import FeatureBank
from tabicl import TabICLClassifier
import torch,argparse,warnings

def main():
    p=argparse.ArgumentParser();p.add_argument('--families',nargs='+',default=['base13','base23','anchor','geometry','owl']);p.add_argument('--seed',type=int,default=5601);p.add_argument('--estimators',type=int,default=8);p.add_argument('--version',default='v2',choices=['v1','v2']);args=p.parse_args()
    torch.set_num_threads(2);warnings.filterwarnings('ignore',category=FutureWarning)
    bank=FeatureBank();q,t,h=inputs();bo,bt=incumbent(q,t);results=[];manifest=[]
    configs={'base13':['base13'],'base23':['base23'],'anchor':['base23','gt'],'geometry':['base23','gt','geometry_derived'],'owl':['base23','gt','owl','clip_text']}
    start=time.time()
    for family in args.families:
        name=f'tabicl{args.version}_{family}_n{args.estimators}_s{args.seed}';pred=[]
        checkpoint='tabicl-classifier-v2-20260212.ckpt' if args.version=='v2' else 'tabicl-classifier-v1.1-20250506.ckpt'
        for fit in DAYS:
            use=bank.rows.split.eq('oof')&bank.rows.day.eq(fit);ev=~use
            X,groups=bank.build(use,fit);cols=list(dict.fromkeys([c for key in configs[family] for c in groups[key]]+['class_name']))
            assert not {'id','frame_id','day','error','score'}&set(cols)
            model=TabICLClassifier(n_estimators=args.estimators,model_path=str(A/'models'/checkpoint),allow_auto_download=False,device='cpu',n_jobs=2,batch_size=2,random_state=args.seed,use_amp=False,verbose=False)
            model.fit(X.loc[use,cols],bank.rows.loc[use,'error'].astype(int))
            prob=model.predict_proba(X.loc[ev,cols])[:,1]
            d=bank.rows.loc[ev,['id','split']].assign(fit_day=fit,p_error=prob);pred.append(d)
            X[cols].to_csv(A/f'features/{family}_fit{fit[-2:]}_bank.csv',index=False)
            model.save(str(A/f'models/{name}_fit{fit[-2:]}.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False)
            manifest.append(dict(name=name,fit_day=fit,features=cols,train_ids=bank.rows.loc[use,'id'].tolist(),checkpoint=checkpoint,n_estimators=args.estimators,seed=args.seed,device='cpu'))
            print('TABICL',name,fit,'features',len(cols),'seconds',round(time.time()-start,1),flush=True)
        pred=pd.concat(pred);pred.to_csv(A/f'features/{name}_predictions.csv',index=False)
        oo=pred[pred.split.eq('oof')].set_index('id').loc[q.id,'p_error'].to_numpy();tt=pred[pred.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
        stand=metrics(q,oo);print('STANDALONE',name,json.dumps(stand),flush=True)
        for scope in ['all','person','nude_hands','cup','tray']:
            om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool);tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
            for alpha in [.05,.1,.2,.4]:
                s,z=bo.copy(),bt.copy();s[om]=(s[om]+alpha*oo[om])/(1+alpha);z[tm]=(z[tm]+alpha*tt[tm])/(1+alpha)
                results.append(save_score(f'{name}_{scope}_a{int(alpha*100)}',s,z,dict(family='tabicl',features=family,scope=scope,alpha=alpha,standalone_ap=stand['ap'],seed=args.seed)))
        (A/f'models/tabicl{args.version}_manifest_s{args.seed}.json').write_text(json.dumps(manifest,indent=2));pd.DataFrame(results).sort_values('ap',ascending=False).to_csv(A/f'results/tabicl{args.version}_s{args.seed}.csv',index=False)
        print('BEST',pd.DataFrame(results).sort_values('ap',ascending=False).head(5).to_string(index=False),flush=True)

if __name__=='__main__':main()
