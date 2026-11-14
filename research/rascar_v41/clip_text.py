"""CLIP text/image agreement on tight crops, with train-day presence calibration.
This never claims that presence proves a correct bbox.
"""
from common import *
import torch,cv2,time,joblib,gc
from transformers import CLIPModel,CLIPTokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(3);cv2.setNumThreads(1);device='mps';assert torch.backends.mps.is_available()
rec=pd.read_csv(A/'inputs/records.csv');extra=pd.read_csv(ROOT/'rascar_v37/inputs/promoted_pool_presence.csv')
hold=set(rec.loc[rec.split.eq('pool_holdout'),'frame_id']);extra=extra[~extra.frame_id.isin(hold)&extra.semantic_target.notna()].copy()
extra['id']='review_'+extra.id;extra['split']='old_review';extra['presence']=extra.semantic_target;extra['weight']=1.;rec=pd.concat([rec,extra],ignore_index=True)
names=['person','nude_hands','cup','tray','elbow','sleeve','phone','fabric','bottle','plate','table','chair','bag','dog']
phrases=['a person in a cafe','a bare human hand with fingers','a drinking cup or drinking glass','a rectangular food serving tray','a human elbow and forearm','a clothing sleeve','a mobile phone','a piece of fabric or clothing','a bottle','a round plate or saucer','a wooden tabletop','an empty chair','a handbag or backpack','a dog']
prompts=[f'a close-up photo of {s}.' for s in phrases]
p=A/'model_cache/clip';model=CLIPModel.from_pretrained(str(p),local_files_only=True).eval().to(device);tok=CLIPTokenizer.from_pretrained(str(p),local_files_only=True)
with torch.inference_mode():
 text=model.get_text_features(**{k:v.to(device) for k,v in tok(prompts,padding=True,return_tensors='pt').items()});text=torch.nn.functional.normalize(text,dim=1)
mean=torch.tensor([.48145466,.4578275,.40821073],device=device).view(1,3,1,1);std=torch.tensor([.26862954,.26130258,.27577711],device=device).view(1,3,1,1)
similarities=[];cache={};start=time.time()
for off in range(0,len(rec),16):
 batch=[]
 for _,r in rec.iloc[off:off+16].iterrows():
  if r.frame_id not in cache:cache[r.frame_id]=cv2.resize(cv2.imread(str(D/f'{r.frame_id}.jpg')),(1280,720))
  b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;wh=np.maximum(b[2:]-b[:2],4)*1.1;lo=np.maximum(0,np.floor(cc-wh/2)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+wh/2)).astype(int)
  im=cache[r.frame_id][lo[1]:hi[1],lo[0]:hi[0],::-1];batch.append(cv2.resize(im,(224,224),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
 x=(torch.tensor(np.stack(batch),device=device).float()/255-mean)/std
 with torch.inference_mode():z=torch.nn.functional.normalize(model.get_image_features(pixel_values=x),dim=1);similarities.append((z@text.T).cpu().numpy())
 if off%320==0:print('TEXT CLIP',off,len(rec),round(time.time()-start,1),flush=True)
sim=np.concatenate(similarities);np.save(A/'features/clip_text_similarities.npy',sim);rec.to_csv(A/'inputs/clip_text_records.csv',index=False)
cls=rec.class_name.map(CL.index).to_numpy();onehot=np.eye(4)[cls];features=np.c_[sim,onehot,(sim[:,:,None]*onehot[:,None,:]).reshape(len(rec),-1)]
outputs=[];manifest=[]
for fit in DAYS:
 train=rec.day.eq(fit)&rec.presence.notna()&~rec.split.isin(['test','pool_holdout']);ev=rec.split.isin(['test','pool_holdout'])|(rec.split.eq('query')&rec.day.ne(fit))
 assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
 w=rec.loc[train,'weight'].to_numpy()/rec.loc[train].groupby('frame_id').id.transform('size').to_numpy();w*=len(w)/w.sum()
 lr=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,random_state=4101));lr.fit(features[train],rec.loc[train,'presence'],logisticregression__sample_weight=w)
 prob=lr.predict_proba(features[ev])[:,1];d=rec.loc[ev,['id','frame_id','day','class_name','split']].copy();d['fit_day']=fit;d['presence']=prob;d['score']=1-prob;outputs.append(d)
 joblib.dump(lr,A/f'models/clip_text_fit{fit[-2:]}.joblib');manifest.append(dict(fit_day=fit,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),test_in_training=False,presence_only=True,old_review_in_training=int((train&rec.split.eq('old_review')).sum())))
pd.concat(outputs).to_csv(A/'features/clip_text_predictions.csv',index=False);(A/'models/clip_text_manifest.json').write_text(json.dumps(dict(models=manifest,prompts=dict(zip(names,prompts)),backbone='openai/clip-vit-base-patch32',crop_scale=1.1),indent=2))
