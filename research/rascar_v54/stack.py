"""Use augmentation predictions as nested features alongside the V52 anchor.

Only frozen geometry and train-subset GT prototypes feed the inner auxiliary
models. Occupancy heads are excluded here to avoid an extra nesting dependency.
"""
from common54 import *
from features import prototype_features
from train import fitting_rows, configurations
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as AP
import time
sys.path.insert(0,str(ROOT/'rascar_v53'))
from experiment import FeatureBank
from common import baseline


def auxiliary(rows):
    path=A/'features/nested_augmentation.csv'
    if path.exists():return pd.read_csv(path)
    static=pd.read_pickle(A/'features/static.pkl'); manifests=json.loads((ROOT/'rascar_v48/models/inner_manifest.json').read_text())
    output=[]; ledger=[]
    for fit in DAYS:
        for budget in [0,1]:
            name=f'visual_d4_all_w{budget*100:03}'
            spec=configurations()[name]; tr_all,weights=fitting_rows(rows,fit,spec)
            for m in [m for m in manifests if m['key']=='small_control' and m['outer_fit_day']==fit]:
                allowed=set(m['fit_frames']); held=set(m['held_frames']); assert allowed.isdisjoint(held)
                tr=tr_all & rows.frame_id.isin(allowed); ev=rows.split.eq('oof') & rows.frame_id.isin(held)
                assert (rows.loc[tr,'day']==fit).all()
                bank=pd.concat([static,prototype_features(rows,fit,allowed_frames=allowed)],axis=1)
                bank['class_name']=rows.class_name
                cols=list(bank)
                for seed in [23,24,25]:
                    model=CatBoostClassifier(iterations=400,depth=4,learning_rate=.03,l2_leaf_reg=20,random_seed=seed,
                                            thread_count=2,loss_function='Logloss',verbose=False,allow_writing_files=False)
                    model.fit(bank.loc[tr,cols],rows.loc[tr,'error'],sample_weight=weights.loc[tr[tr].index],cat_features=['class_name'])
                    p=model.predict_proba(bank.loc[ev,cols])[:,1]
                    modelname=f'nested_visual_w{budget}_fit{fit[-2:]}_block{m["inner_block"]}_s{seed}'
                    model.save_model(str(A/f'models/{modelname}.cbm'))
                    output.append(rows.loc[ev,['id','split']].assign(fit_day=fit,budget=budget,seed=seed,p_error=p,role='meta_train'))
                ledger.append(dict(fit_day=fit,budget=budget,block=m['inner_block'],fit_frames=sorted(allowed),held_frames=sorted(held),
                                   original_train_rows=int((tr&rows.split.eq('oof')).sum()),augmented_train_rows=int((tr&rows.split.eq('augmentation')).sum()),
                                   prototype_frames=sorted(allowed),features=cols))
                print('INNER',fit,budget,m['inner_block'],int(tr.sum()),int(ev.sum()),flush=True)
            outer=pd.read_csv(A/f'features/{name}_actual_confirm_predictions.csv')
            outer=outer[outer.fit_day.eq(fit)].copy();outer['budget']=budget
            outer['role']=np.where(outer.split.eq('test'),'test','oof')
            output.append(outer[['id','split','fit_day','budget','seed','p_error','role']])
    out=pd.concat(output);out.to_csv(path,index=False)
    (A/'audit/auxiliary_inner_manifest.json').write_text(json.dumps(ledger,indent=2))
    return out


def main():
    rows=pd.read_csv(A/'inputs/records.csv'); aux=auxiliary(rows)
    bank=FeatureBank(); q,t,h=inputs(); bo,bt=baseline(q,t); v52,_=incumbent(q,t)
    all_pred=[]; results=[]; manifests=[]; imports=[]
    for variant in ['anchor','noaug_feature','aug_feature','both_features','delta_feature']:
        for depth in [2,3]:
            preds=[]
            for fit in DAYS:
                tr=bank.rows.split.eq('oof')&bank.rows.day.eq(fit);ev=~tr
                X,groups=bank.build(tr,fit);X=X.copy()
                auxp=aux[aux.fit_day.eq(fit)].groupby(['id','budget']).p_error.mean().unstack().loc[bank.rows.id]
                assert auxp.notna().all().all()
                X['aux_noaug']=auxp[0].to_numpy();X['aux_aug']=auxp[1].to_numpy();X['aux_delta']=X.aux_aug-X.aux_noaug
                extra={'anchor':[],'noaug_feature':['aux_noaug'],'aug_feature':['aux_aug'],
                       'both_features':['aux_noaug','aux_aug','aux_delta'],'delta_feature':['aux_delta']}[variant]
                cols=list(dict.fromkeys(groups['base23']+groups['gt']+extra+['class_name']))
                for seed in [23,24,25]:
                    name=f'stack_{variant}_d{depth}_s{seed}_fit{fit[-2:]}'
                    model=CatBoostClassifier(iterations=250,depth=depth,learning_rate=.03,l2_leaf_reg=10,
                         random_seed=seed,thread_count=2,loss_function='Logloss',verbose=False,allow_writing_files=False)
                    model.fit(X.loc[tr,cols],bank.rows.loc[tr,'error'],cat_features=['class_name'])
                    pred=model.predict_proba(X.loc[ev,cols])[:,1];model.save_model(str(A/f'models/{name}.cbm'))
                    preds.append(bank.rows.loc[ev,['id','split']].assign(seed=seed,fit_day=fit,p_error=pred))
                    manifests.append(dict(name=name,variant=variant,depth=depth,seed=seed,fit_day=fit,features=cols,train_frames=sorted(bank.rows.loc[tr,'frame_id'].unique())))
                    imports.extend(dict(model=name,feature=c,importance=float(v)) for c,v in zip(cols,model.feature_importances_))
                X.loc[ev,cols].assign(id=bank.rows.loc[ev,'id']).to_pickle(A/f'features/stack_{variant}_d{depth}_fit{fit[-2:]}_eval.pkl')
            pred=pd.concat(preds);tag=f'stack_{variant}_d{depth}';pred.to_csv(A/f'features/{tag}_predictions.csv',index=False)
            oo=pred[pred.split.eq('oof')].groupby('id').p_error.mean().loc[q.id].to_numpy()
            tt=pred[pred.split.eq('test')].groupby('id').p_error.mean().loc[t.id].to_numpy()
            for scope in ['person','all']:
                om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool)
                tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
                for alpha in [.1,.2]:
                    s,z=bo.copy(),bt.copy();s[om]=(bo[om]+alpha*oo[om])/(1+alpha);z[tm]=(bt[tm]+alpha*tt[tm])/(1+alpha)
                    name=f'{tag}_{scope}_a{int(alpha*100):02}'
                    q.assign(score=s).to_csv(A/f'results/oof/{name}.csv',index=False)
                    t[['id']].assign(score=z).to_csv(A/f'results/submissions/{name}.csv',index=False)
                    results.append(dict(name=name,variant=variant,depth=depth,scope=scope,alpha=alpha,**metrics(q,s),delta_v52=AP(q.error,s)-AP(q.error,v52),standalone_ap=AP(q.error,oo)))
            print(tag,'standalone',round(AP(q.error,oo),6),'bestblend',round(max(r['ap'] for r in results[-4:]),6),flush=True)
            pd.DataFrame(results).to_csv(A/'results/stack_experiments.csv',index=False)
            pd.DataFrame(imports).to_csv(A/'audit/stack_importances.csv',index=False)
            (A/'models/stack_manifest.json').write_text(json.dumps(manifests,indent=2))
    print(pd.DataFrame(results).sort_values('ap',ascending=False).to_string(index=False),flush=True)


if __name__=='__main__':main()
