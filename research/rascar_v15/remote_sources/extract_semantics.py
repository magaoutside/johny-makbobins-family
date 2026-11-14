from common import *
import torch,joblib
from PIL import Image
from transformers import AutoImageProcessor,AutoModel
from kaggle_secrets import UserSecretsClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
os.environ['TOKENIZERS_PARALLELISM']='false';torch.set_num_threads(2)
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict()
R=[]
for kind,df in [('query_train',q),('query_test',te)]:
 for r in df.to_dict('records'):R.append({**r,'kind':kind,'semantic_class':'unknown','weight':1})
gt=h[h.class_name.isin(CL)].copy()
for k,r in gt.iterrows():
 b=r[B].to_numpy(float)
 R.append(dict(id=f'gt_{k}',frame_id=r.frame_id,day=r.day,class_name=r.class_name,semantic_class=r.class_name,kind='gt',weight=1,error=0,**dict(zip(B,b))))
 if r.class_name=='person':
  for amount in [.25,.45,.6]:
   bb=b.copy();bb[3]-=(b[3]-b[1])*amount
   ov=iou(bb,gt[(gt.frame_id==r.frame_id)&(gt.class_name=='person')][B]).max()
   R.append(dict(id=f'short_{k}_{amount}',frame_id=r.frame_id,day=r.day,class_name='person',semantic_class='person',kind='synthetic',weight=.25,error=int(ov<.6),**dict(zip(B,bb))))
# Empty views at the same scene positions supply chairs, clothing and table negatives.
# Reject any crop overlapping manually annotated foreground; no test labels enter fitting.
rng=np.random.default_rng(17)
for frame in fr[fr.frame_id.isin(h.frame_id)].itertuples():
 fg=gt[gt.frame_id.eq(frame.frame_id)][B].to_numpy();candidates=q[q.day.eq(frame.day)]
 accepted=0
 for ix in rng.permutation(len(candidates)):
  rr=candidates.iloc[ix];b=rr[B].to_numpy(float);area=np.prod(b[2:]-b[:2])
  inter=np.maximum(0,np.minimum(fg[:,2:],b[2:])-np.maximum(fg[:,:2],b[:2])).prod(1) if len(fg) else np.array([0])
  if inter.max()/area>.015:continue
  R.append(dict(id=f'bg_{frame.frame_id}_{accepted}',frame_id=frame.frame_id,day=frame.day,class_name='background',semantic_class='background',kind='background',weight=.5,error=1,**dict(zip(B,b))))
  accepted+=1
  if accepted==8:break
records=pd.DataFrame(R);records.to_csv(A/'semantic_records.csv',index=False)
manifest=json.loads((A/'v7/sam31_v7/dino_b_manifest.json').read_text());token=UserSecretsClient().get_secret('HF_TOKEN')
processor=AutoImageProcessor.from_pretrained(manifest['model'],revision=manifest['revision'],token=token)
model=AutoModel.from_pretrained(manifest['model'],revision=manifest['revision'],token=token,dtype=torch.float16).to('cuda:1').eval();del token
for p in model.parameters():p.requires_grad_(False)
vectors=[];skip=1+model.config.num_register_tokens
for start in range(0,len(records),24):
 batch=records.iloc[start:start+24];images=[]
 for r in batch.itertuples():
  im=Image.open(paths[r.frame_id]).convert('RGB');b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
  for context in [0,.35]:
   bb=b+np.array([-1,-1,1,1])*np.tile(wh,2)*context;bb[[0,2]]=np.clip(bb[[0,2]],0,1280);bb[[1,3]]=np.clip(bb[[1,3]],0,720)
   images.append(im.crop(tuple(bb)).resize((224,224),Image.Resampling.BICUBIC))
 inp=processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to('cuda:1',torch.float16)
 with torch.inference_mode():z=model(**inp).last_hidden_state.float();cls=torch.nn.functional.normalize(z[:,0],dim=-1);patch=torch.nn.functional.normalize(z[:,skip:].mean(1),dim=-1);features=torch.cat([cls,patch],dim=-1).cpu().numpy().reshape(len(batch),-1)/2
 vectors.append(features)
 if start%240==0:print('SEMANTIC_FEATURES',start,len(records),flush=True)
X=np.concatenate(vectors);np.save(A/'semantic_features.npy',X);del model;torch.cuda.empty_cache()
semantic_oof=np.zeros((len(q),5));semantic_test=[];classes=CL+['background']
quality_oof=np.zeros(len(q));quality_test=[]
for fit,held in zip(DAYS,DAYS[::-1]):
 train=records.day.eq(fit)&records.kind.isin(['gt','synthetic','background']);val=records.kind.eq('query_train')&records.day.eq(held);test=records.kind.eq('query_test')
 model=LogisticRegression(C=10,max_iter=2000,class_weight='balanced');model.fit(X[train],records.loc[train,'semantic_class'],sample_weight=records.loc[train,'weight'])
 pred=model.predict_proba(X[val]);predt=model.predict_proba(X[test]);order=[list(model.classes_).index(c) for c in classes]
 ids=records.loc[val,'id'];idx=q.index[q.id.isin(ids)];mapping=pd.DataFrame(pred[:,order],index=ids,columns=classes);semantic_oof[idx]=mapping.loc[q.loc[idx,'id']]
 semantic_test.append(predt[:,order]);joblib.dump(model,A/f'semantic_{fit}.joblib')
 # Separate quality head sees actual error labels plus controlled person truncations.
 train=records.day.eq(fit)&records.class_name.eq('person')&records.kind.isin(['gt','synthetic','query_train'])
 val=records.kind.eq('query_train')&records.day.eq(held)&records.class_name.eq('person');test=records.kind.eq('query_test')&records.class_name.eq('person')
 model=LogisticRegression(C=1,max_iter=2000,class_weight='balanced');model.fit(X[train],records.loc[train,'error'].astype(int),sample_weight=records.loc[train,'weight'])
 lookup=dict(zip(records.loc[val,'id'],model.predict_proba(X[val])[:,1]));ii=q.id.isin(lookup);quality_oof[ii]=q.loc[ii,'id'].map(lookup);quality_test.append(model.predict_proba(X[test])[:,1]);joblib.dump(model,A/f'quality_{fit}.joblib')
o=q[['id']].copy();t=te[['id']].copy()
for j,c in enumerate(classes):o['p_'+c]=semantic_oof[:,j];t['p_'+c]=np.mean(semantic_test,axis=0)[:,j]
o['quality_person']=quality_oof;t['quality_person']=0.;t.loc[te.class_name.eq('person'),'quality_person']=np.mean(quality_test,axis=0)
o.to_csv(A/'semantic_oof.csv',index=False);t.to_csv(A/'semantic_test.csv',index=False)
(A/'semantic_manifest.json').write_text(json.dumps(dict(backbone=manifest,classes=classes,contexts=[0,.35],crop_size=224,fit_days=DAYS,training_frames=84,test_labels_used=False,train_counts=records.groupby(['day','kind','semantic_class']).size().to_dict().__str__()),indent=2))
print('SEMANTICS_DONE',len(records),flush=True)
