"""Independent SigLIP 2 hand-presence features, same train-only labels as V34.
Frozen vision backbone, 1.15x query crop, no test labels or positional features.
The older 36-frame pool review is a diagnostic set, no longer a fresh holdout.
"""
from common36 import *
import torch,cv2,time,joblib
from PIL import Image
from transformers import SiglipVisionModel,AutoImageProcessor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
torch.set_num_threads(3);cv2.setNumThreads(1)

def extract(rec,filename='siglip_vectors.npy'):
 path=A/'features'/filename
 if path.exists():
  x=np.load(path);assert len(x)==len(rec);return x
 model_path=A/'model_cache/siglip2-base-patch16-224'
 model=SiglipVisionModel.from_pretrained(str(model_path),local_files_only=True).eval()
 processor=AutoImageProcessor.from_pretrained(str(model_path),local_files_only=True,use_fast=False)
 out=[];start=time.time()
 for off in range(0,len(rec),8):
  batch=[]
  for _,r in rec.iloc[off:off+8].iterrows():
   b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],4);cc=(b[:2]+b[2:])/2
   lo=np.maximum(0,np.floor(cc-.575*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+.575*wh)).astype(int)
   im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))
   batch.append(Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]))
  data=processor(images=batch,return_tensors='pt')
  with torch.inference_mode():z=model(**data).pooler_output
  out.append(torch.nn.functional.normalize(z,dim=1).numpy())
  if off%160==0:print('SIGLIP_FEATURES',off,len(rec),round(time.time()-start,1),flush=True)
 x=np.concatenate(out);np.save(path,x);return x

def main():
 rec=pd.read_csv(ROOT/'rascar_v34/inputs/semantic_records.csv');rec.to_csv(A/'inputs/siglip_records.csv',index=False)
 x=extract(rec);results=[];manifest=[];evalrows=[];fresh_frames=set(rec.loc[rec.split.eq('fresh_pool'),'frame_id'])
 for cval in [.03,.1]:
  preds={}
  for fit in DAYS:
   train=rec.day.eq(fit)&rec.semantic_target.notna()&~rec.split.isin(['test','fresh_pool'])
   assert not set(rec.loc[train,'frame_id'])&fresh_frames
   weights=rec.loc[train,'source_weight']/rec.loc[train].groupby('frame_id').id.transform('size');weights*=train.sum()/weights.sum()
   model=make_pipeline(StandardScaler(),LogisticRegression(C=cval,class_weight='balanced',max_iter=1000,random_state=3611));model.fit(x[train],rec.loc[train,'semantic_target'],logisticregression__sample_weight=weights.to_numpy())
   joblib.dump(model,A/f'models/siglip_hand_c{cval}_fit{fit[-2:]}.joblib');prob=model.predict_proba(x)[:,1];preds[fit]=pd.Series(prob,index=rec.id)
   manifest.append(dict(C=cval,fit_day=fit,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),train_crops=int(train.sum()),positives=int(rec.loc[train,'semantic_target'].sum()),test_in_fit=False,pool_review_in_fit=False,backbone='google/siglip2-base-patch16-224',crop_scale=1.15))
   use=rec.split.eq('test')|(rec.day.ne(fit)&rec.split.isin(['oof','fresh_pool']));e=rec.loc[use,['id','frame_id','day','class_name','split','semantic_target']].copy();e['C']=cval;e['fit_day']=fit;e['p_hand']=prob[use];evalrows.append(e)
  probs={s:np.ones(len(rows)) for s,rows in ROWS.items()}
  for fit in DAYS:
   m=q.class_name.eq('nude_hands')&q.day.ne(fit);probs['oof'][m]=preds[fit].loc[q.loc[m,'id']]
  m=te.class_name.eq('nude_hands');probs['test'][m]=np.maximum.reduce([preds[fit].loc[te.loc[m,'id']].to_numpy() for fit in DAYS])
  for cutoff in [.05,.15]:
   for weight in [.25,.5]:
    scores={s:BASE[s]+weight*np.maximum(1-p-BASE[s],0)*(p<cutoff)*ROWS[s].class_name.eq('nude_hands').to_numpy() for s,p in probs.items()}
    results.append(save(f'siglip_hand_c{cval}_p{cutoff}_w{weight}',scores,'Independent frozen SigLIP2, hand-presence logreg; same V34 labels/weights; test requires both daily models to reject hand presence. Geometry is not relabeled.'))
 pd.DataFrame(results).to_csv(A/'results/siglip_experiments.csv',index=False);(A/'models/siglip_manifest.json').write_text(json.dumps(manifest,indent=2));ev=pd.concat(evalrows);ev.to_csv(A/'audit/siglip_predictions.csv',index=False)
 fresh=pd.read_csv(ROOT/'rascar_v34/inputs/fresh_pool_candidates.csv');lab=json.loads((ROOT/'rascar_v34/inputs/fresh_pool_presence_review.json').read_text());fresh['presence']=np.nan
 fresh.loc[fresh.review_index.isin(lab['present_review_indices']),'presence']=1;fresh.loc[fresh.review_index.isin(lab['absent_review_indices']),'presence']=0
 ev=ev[ev.split.eq('fresh_pool')].merge(fresh[['id','review_index','presence','stratum']],on='id');ev.to_csv(A/'audit/pool_siglip_predictions.csv',index=False);summ=[]
 for (c,day),d in ev.groupby(['C','day']):
  d=d[d.presence.notna()];summ.append(dict(C=c,day=day,n=len(d),nonhands=int(d.presence.eq(0).sum()),presence_detection_ap=float(AP(1-d.presence,1-d.p_hand)),true_hands_flagged_p005=int((d.presence.eq(1)&d.p_hand.lt(.05)).sum()),true_hands_flagged_p015=int((d.presence.eq(1)&d.p_hand.lt(.15)).sum()),nonhands_flagged_p005=int((d.presence.eq(0)&d.p_hand.lt(.05)).sum()),nonhands_flagged_p015=int((d.presence.eq(0)&d.p_hand.lt(.15)).sum())))
 pd.DataFrame(summ).to_csv(A/'results/pool_siglip_presence.csv',index=False);print(pd.DataFrame(summ).to_string(index=False),flush=True)

if __name__=='__main__':main()
