"""TabICL on bounded, parent-balanced samples of exact-matching bbox augmentations."""
from common56 import *
sys.path.insert(0,str(A/'vendor'))
from tabicl import TabICLClassifier
import torch,argparse

def main():
    p=argparse.ArgumentParser();p.add_argument('--counts',type=int,nargs='+',default=[0,4,12]);p.add_argument('--families',nargs='+',default=['visual','dense']);p.add_argument('--seed',type=int,default=5601);args=p.parse_args()
    torch.set_num_threads(2)
    q,t,h=inputs();bo,bt=incumbent(q,t);rows=pd.read_csv(ROOT/'rascar_v54/inputs/records.csv');groups=json.loads((ROOT/'rascar_v54/features/groups.json').read_text());results=[];manifest=[];start=time.time()
    banks={fit:pd.read_pickle(ROOT/f'rascar_v54/features/bank_fit{fit[-2:]}.pkl') for fit in DAYS}
    geometry=[c for c in groups['geometry'] if not c.startswith(('owl05_','owl20_'))]
    for family in args.families:
        cols=geometry+groups['proto']+(groups['dense_small'] if family=='dense' else [])+['class_name']
        assert len(cols)<100 and not {'id','day','frame_id','error','score'}&set(cols)
        for count in args.counts:
            name=f'tabaug_{family}_k{count}_s{args.seed}';pred=[]
            for fit in DAYS:
                original=rows.split.eq('oof')&rows.day.eq(fit);tr=list(rows.index[original]);rng=np.random.default_rng(args.seed)
                if count:
                    candidates=rows[rows.split.eq('augmentation')&rows.day.eq(fit)&rows.family.ne('toward_gt')]
                    for parent,g in candidates.groupby('parent_id',sort=True):
                        # Equal count per parent, random without consulting the held day.
                        tr.extend(rng.choice(g.index,min(count,len(g)),replace=False).tolist())
                tr=np.array(tr);ev=rows.split.eq('test')|(rows.split.eq('oof')&rows.day.ne(fit))
                assert set(rows.loc[tr,'frame_id']).isdisjoint(rows.loc[ev,'frame_id'])
                model=TabICLClassifier(n_estimators=8,model_path=str(A/'models/tabicl-classifier-v2-20260212.ckpt'),allow_auto_download=False,device='cpu',n_jobs=2,batch_size=2,random_state=args.seed,use_amp=False)
                model.fit(banks[fit].loc[tr,cols],rows.loc[tr,'error'].astype(int));prob=model.predict_proba(banks[fit].loc[ev,cols])[:,1]
                pred.append(rows.loc[ev,['id','split']].assign(fit_day=fit,p_error=prob))
                model.save(str(A/f'models/{name}_fit{fit[-2:]}.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False)
                manifest.append(dict(name=name,fit_day=fit,features=cols,training_ids=rows.loc[tr,'id'].tolist(),train_frames=sorted(rows.loc[tr,'frame_id'].unique()),seed=args.seed,original_rows=int(original.sum()),augmented_rows=len(tr)-int(original.sum())))
                print('TABAUG',name,fit,'train',len(tr),'sec',round(time.time()-start,1),flush=True)
            pred=pd.concat(pred);pred.to_csv(A/f'features/{name}_predictions.csv',index=False)
            oo=pred[pred.split.eq('oof')].set_index('id').loc[q.id,'p_error'].to_numpy();tt=pred[pred.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy();stand=metrics(q,oo)
            print('STANDALONE',name,json.dumps(stand),flush=True)
            for scope in ['all','person','nude_hands']:
                om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool);tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
                for alpha in [.05,.1,.2,.4]:
                    s,z=bo.copy(),bt.copy();s[om]=(s[om]+alpha*oo[om])/(1+alpha);z[tm]=(z[tm]+alpha*tt[tm])/(1+alpha)
                    results.append(save_score(f'{name}_{scope}_a{int(alpha*100)}',s,z,dict(family='tabaug',features=family,count=count,scope=scope,alpha=alpha,standalone_ap=stand['ap'])))
            pd.DataFrame(results).sort_values('ap',ascending=False).to_csv(A/f'results/tabaug_s{args.seed}.csv',index=False)
            (A/f'models/tabaug_manifest_s{args.seed}.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(results).sort_values('ap',ascending=False).head(15).to_string(index=False),flush=True)

if __name__=='__main__':main()
