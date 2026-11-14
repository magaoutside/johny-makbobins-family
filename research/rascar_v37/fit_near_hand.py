from common37 import *
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def main():
    rec=pd.read_csv(ROOT/'rascar_v36/inputs/pool_augmented_semantic_records.csv');x=np.load(ROOT/'rascar_v34/features/semantic_vectors.npy')
    nr=pd.read_csv(A/'inputs/near_hand_review.csv');lab=json.loads((A/'inputs/near_hand_review.json').read_text());nr=nr[nr.review_index.isin(lab['approved_negative_indices'])].copy()
    nx=np.load(A/'features/near_hand_vectors.npy')[nr.feature_index.to_numpy(int)]
    nr['semantic_target']=0.;nr['source_weight']=1.;nr['split']='reviewed_near_negative'
    rec=pd.concat([rec,nr],ignore_index=True);x=np.concatenate([x,nx]);assert rec.id.is_unique
    rec.to_csv(A/'inputs/near_augmented_records.csv',index=False)
    pool=pd.read_csv(A/'inputs/promoted_pool_presence.csv');bx=np.concatenate([np.load(ROOT/'rascar_v36/features/blind_pool_dino.npy'),np.load(A/'features/pool_context_vectors.npy')]);assert len(pool)==len(bx)
    preds={};summ=[];manifest=[];ev=[]
    for fit in DAYS+['all']:
        train=(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))&rec.semantic_target.notna()&~rec.split.isin(['test','fresh_pool'])
        assert not set(rec.loc[train,'frame_id'])&set(pool.frame_id)
        weights=rec.loc[train,'source_weight']/rec.loc[train].groupby('frame_id').id.transform('size');weights*=train.sum()/weights.sum()
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1500,random_state=3701));model.fit(x[train],rec.loc[train,'semantic_target'],logisticregression__sample_weight=weights)
        joblib.dump(model,A/f'models/near_hand_fit{fit[-2:]}.joblib');preds[fit]=pd.Series(model.predict_proba(x)[:,1],index=rec.id)
        manifest.append(dict(fit_day=fit,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),new_negatives=int((train&rec.split.eq('reviewed_near_negative')).sum()),pool_v36_v37_in_fit=False,test_in_fit=False))
        if fit!='all':
            use=pool.day.ne(fit)&pool.semantic_target.notna();d=pool.loc[use].copy();d['p_present']=model.predict_proba(bx[use])[:,1];d['fit_day']=fit;ev.append(d)
    for (source,day),d in pd.concat(ev).groupby(['review_source','day']):
        summ.append(dict(review_source=source,day=day,n=len(d),negative=int(d.semantic_target.eq(0).sum()),false_rejects=int((d.semantic_target.eq(1)&d.p_present.lt(.15)).sum()),true_rejects=int((d.semantic_target.eq(0)&d.p_present.lt(.15)).sum())))
    pd.DataFrame(summ).to_csv(A/'results/near_hand_pool.csv',index=False);pd.concat(ev).to_csv(A/'audit/near_hand_pool_predictions.csv',index=False);(A/'models/near_hand_manifest.json').write_text(json.dumps(manifest,indent=2));print(pd.DataFrame(summ).to_string(index=False))
    results=[]
    for mode in ['daily_consensus','full_fit']:
        probs={s:np.ones(len(rows)) for s,rows in ROWS.items()}
        for fit in DAYS:
            m=q.class_name.eq('nude_hands')&q.day.ne(fit);probs['oof'][m]=preds[fit].loc[q.loc[m,'id']]
        m=te.class_name.eq('nude_hands');ids=te.loc[m,'id'];probs['test'][m]=preds['all'].loc[ids] if mode=='full_fit' else np.maximum.reduce([preds[fit].loc[ids].to_numpy() for fit in DAYS])
        for s,rows in ROWS.items():rows.assign(p_present=probs[s]).to_csv(A/f'features/near_hand_{mode}_{s}.csv',index=False)
        for cutoff in [.05,.15]:
            for weight in [.25,.5]:
                scores={s:BASE[s]+weight*np.maximum(1-p-BASE[s],0)*(p<cutoff)*ROWS[s].class_name.eq('nude_hands').to_numpy() for s,p in probs.items()}
                results.append(save(f'near_hand_{mode}_p{cutoff}_w{weight}',scores,'12 visually checked negatives mined near training GT hands. V36/V37 pool reviews excluded from fit, opposite-day diagnostics.'))
    pd.DataFrame(results).to_csv(A/'results/near_hand_experiments.csv',index=False)

if __name__=='__main__':main()
