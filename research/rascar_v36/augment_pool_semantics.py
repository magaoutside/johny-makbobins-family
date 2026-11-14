"""Promote only the older V34 presence review into training in this experiment.
The source review and its original evaluation history stay unchanged.
The newly reviewed V36 pool remains excluded, as do all test images.
"""
from common36 import *
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

def main():
 rec=pd.read_csv(ROOT/'rascar_v34/inputs/semantic_records.csv');old=pd.read_csv(ROOT/'rascar_v34/inputs/fresh_pool_candidates.csv');lab=json.loads((ROOT/'rascar_v34/inputs/fresh_pool_presence_review.json').read_text());labels={}
 for target,key in [(1,'present_review_indices'),(0,'absent_review_indices')]:
  labels.update({k:target for k in old.loc[old.review_index.isin(lab[key]),'id']})
 take=rec.id.isin(labels);rec.loc[take,'semantic_target']=rec.loc[take,'id'].map(labels);rec.loc[take,'split']='historical_review_train';rec.loc[take,'source_weight']=1.
 rec.to_csv(A/'inputs/pool_augmented_semantic_records.csv',index=False)
 blind=pd.read_csv(A/'inputs/blind_pool_candidates.csv');bl=json.loads((A/'inputs/blind_pool_presence_review.json').read_text());blind['presence']=np.nan
 blind.loc[blind.review_index.isin(bl['present_review_indices']),'presence']=1;blind.loc[blind.review_index.isin(bl['absent_review_indices']),'presence']=0
 results=[];manifests=[];evalrows=[];summaries=[]
 for family,vectorpath in [('dino',ROOT/'rascar_v34/features/semantic_vectors.npy'),('siglip',A/'features/siglip_vectors.npy')]:
  x=np.load(vectorpath);bx=np.load(A/f'features/blind_pool_{family}.npy');preds={}
  for fit in DAYS:
   train=rec.day.eq(fit)&rec.semantic_target.notna()&~rec.split.isin(['test','fresh_pool']);assert not set(rec.loc[train,'frame_id'])&set(blind.frame_id)
   weights=rec.loc[train,'source_weight']/rec.loc[train].groupby('frame_id').id.transform('size');weights*=train.sum()/weights.sum()
   model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=3611));model.fit(x[train],rec.loc[train,'semantic_target'],logisticregression__sample_weight=weights.to_numpy())
   joblib.dump(model,A/f'models/{family}_pool_aug_fit{fit[-2:]}.joblib');preds[fit]=pd.Series(model.predict_proba(x)[:,1],index=rec.id)
   manifests.append(dict(family=family,fit_day=fit,C=.1,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),train_crops=int(train.sum()),new_reviewed_training_crops=int((train&take).sum()),label_scope='hand presence only',v36_blind_pool_in_fit=False,test_in_fit=False))
   use=blind.day.ne(fit);d=blind.loc[use,['id','review_index','day','presence']].copy();d['p_hand']=model.predict_proba(bx[use])[:,1];d['family']=family;d['fit_day']=fit;evalrows.append(d);known=d[d.presence.notna()]
   row=dict(family=family,held_day=known.day.iloc[0],n=len(known),nonhands=int(known.presence.eq(0).sum()),presence_detection_ap=float(AP(1-known.presence,1-known.p_hand)))
   for cutoff in [.05,.15]:
    row[f'true_hands_flagged_p{cutoff}']=int((known.presence.eq(1)&known.p_hand.lt(cutoff)).sum());row[f'nonhands_flagged_p{cutoff}']=int((known.presence.eq(0)&known.p_hand.lt(cutoff)).sum())
   summaries.append(row)
  probs={s:np.ones(len(rows)) for s,rows in ROWS.items()}
  for fit in DAYS:
   m=q.class_name.eq('nude_hands')&q.day.ne(fit);probs['oof'][m]=preds[fit].loc[q.loc[m,'id']]
  m=te.class_name.eq('nude_hands');probs['test'][m]=np.maximum.reduce([preds[fit].loc[te.loc[m,'id']].to_numpy() for fit in DAYS])
  for cutoff in [.05,.15]:
   scores={s:BASE[s]+.25*np.maximum(1-p-BASE[s],0)*(p<cutoff)*ROWS[s].class_name.eq('nude_hands').to_numpy() for s,p in probs.items()}
   results.append(save(f'{family}_reviewed_pool_p{cutoff}_w0.25',scores,'Older 29 clear pool-presence reviews added only to their training day; new 36-frame pool excluded; fixed C=.1, weight=.25.'))
 pd.DataFrame(results).to_csv(A/'results/pool_augmented_experiments.csv',index=False);pd.DataFrame(summaries).to_csv(A/'results/blind_pool_after_augmentation.csv',index=False);pd.concat(evalrows).to_csv(A/'audit/blind_pool_augmented_predictions.csv',index=False);(A/'models/pool_augmented_manifest.json').write_text(json.dumps(manifests,indent=2));print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=='__main__':main()
