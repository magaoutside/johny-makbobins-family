from pathlib import Path
import os,sys,json,hashlib
os.environ['OMP_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,pandas as pd,torch,joblib
from PIL import Image
from transformers import AutoImageProcessor,AutoModel
from kaggle_secrets import UserSecretsClient
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.metrics import average_precision_score
E=Path('/kaggle/working/rascar_v14');O=Path('/kaggle/working/oof_validation_v11');Y=E/'yolo26_clean';S=E/'stage3';S.mkdir(exist_ok=True);P=S/'patches';P.mkdir(exist_ok=True)
B=['x1','y1','x2','y2'];DAYS=['2026-10-25','2026-10-26'];torch.set_num_threads(2)
q=pd.read_csv(E/'stage1/oof/v8_baseline.csv');tr=pd.read_csv(O/'train.csv');q=q.merge(tr[['id']+B],on='id',validate='one_to_one');te=pd.read_csv(O/'test.csv');base_test=pd.read_csv('/kaggle/working/submission.csv').set_index('id').loc[te.id,'score'].to_numpy()
fr=pd.read_csv(O/'train_frames.csv');fr['day']=pd.to_datetime(fr.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d');tf=pd.read_csv(E/'stage2/test_frames.csv')
h=pd.read_csv(O/'human_boxes_train.csv');h.class_name=h.class_name.replace(json.loads((O/'scoring_policy.json').read_text())['aliases']);h=h[h.class_name.eq('person')].merge(fr[['frame_id','day']],on='frame_id',validate='many_to_one');h[B]=np.c_[(h.cx-h.w/2)*1280,(h.cy-h.h/2)*720,(h.cx+h.w/2)*1280,(h.cy+h.h/2)*720]
rows=[]
for r in q[q.class_name.eq('person')].to_dict('records'):rows.append(dict(**{k:r[k] for k in ['id','frame_id','day','error']+B},kind='real',weight=2.0))
def iou(b,bs):
 it=np.maximum(0,np.minimum(bs[:,2:],b[2:])-np.maximum(bs[:,:2],b[:2])).prod(1);return it/np.maximum((b[2:]-b[:2]).prod()+(bs[:,2:]-bs[:,:2]).prod(1)-it,1e-9)
# Deterministic synthetic errors, with separate original GT and real official labels.
changes=[(0,0,0,0),(0,0,0,-.2),(0,0,0,-.4),(0,0,0,-.55),(0,.2,0,0),(0,.4,0,0),(-.3,0,-.3,0),(.3,0,.3,0),(0,-.25,0,.25),(-.25,0,.25,0),(0,0,0,.5),(0,.15,0,-.15)]
for k,r in enumerate(h.itertuples()):
 b=np.array([getattr(r,c) for c in B]);wh=np.tile(b[2:]-b[:2],2);allb=h[h.frame_id.eq(r.frame_id)][B].to_numpy()
 for j,delta in enumerate(changes):
  bb=b+np.array(delta)*wh;bb[[0,2]]=np.clip(bb[[0,2]],0,1280);bb[[1,3]]=np.clip(bb[[1,3]],0,720);ov=float(iou(bb,allb).max());rows.append(dict(id=f'synthetic_{k}_{j}',frame_id=r.frame_id,day=r.day,error=int(ov<.6),kind='synthetic',weight=.25,quality_iou=ov,**dict(zip(B,bb))))
for r in te[te.class_name.eq('person')].to_dict('records'):rows.append(dict(**{k:r[k] for k in ['id','frame_id']+B},day='test',error=-1,kind='test',weight=0))
records=pd.DataFrame(rows);records.to_csv(S/'spatial_records.csv',index=False)
manifest=json.loads((E/'stage2/v7_replay/sam31_v7/dino_b_manifest.json').read_text());name=manifest['model'];revision=manifest['revision'];tok=UserSecretsClient().get_secret('HF_TOKEN')
processor=AutoImageProcessor.from_pretrained(name,revision=revision,token=tok);model=AutoModel.from_pretrained(name,revision=revision,token=tok,dtype=torch.float16).to('cuda:1').eval();del tok
for p in model.parameters():p.requires_grad_(False)
skip=1+model.config.num_register_tokens;paths={r.frame_id:Y/'dataset/images'/r.image_path for r in fr.itertuples()};paths.update({r.frame_id:Y/'test_images'/r.image_path for r in tf.itertuples()})
frames=records.frame_id.unique()
for n,fid in enumerate(frames):
 im=Image.open(paths[fid]).convert('RGB').resize((768,432),Image.Resampling.BICUBIC);inp=processor(images=[im],do_resize=False,do_center_crop=False,return_tensors='pt').to('cuda:1',torch.float16)
 with torch.inference_mode():z=model(**inp).last_hidden_state[:,skip:].float()
 z=torch.nn.functional.normalize(z,dim=-1).cpu().numpy()[0].reshape(27,48,-1);np.savez_compressed(P/f'{fid}.npz',features=z.astype('float16'))
 if n%25==0:print('SPATIAL_PATCHES',n,len(frames),flush=True)
del model;torch.cuda.empty_cache()
features=[]
for fid,g in records.groupby('frame_id',sort=False):
 f=np.load(P/f'{fid}.npz')['features'].astype('float32')
 def pool(bb):
  x1,y1,x2,y2=bb*np.array([48/1280,27/720,48/1280,27/720]);a=max(0,min(47,int(np.floor(x1))));b=max(0,min(26,int(np.floor(y1))));c=max(a+1,min(48,int(np.ceil(x2))));d=max(b+1,min(27,int(np.ceil(y2))));return f[b:d,a:c].mean((0,1))
 for idx,r in g.iterrows():
  x1,y1,x2,y2=r[B].to_numpy(float);w=x2-x1;hh=y2-y1;patch=[]
  for iy in range(3):
   for ix in range(3):patch.append(pool(np.array([x1+w*ix/3,y1+hh*iy/3,x1+w*(ix+1)/3,y1+hh*(iy+1)/3])))
  for bb in [[x1,y1-.25*hh,x2,y1],[x1,y2,x2,y2+.25*hh],[x1-.25*w,y1,x1,y2],[x2,y1,x2+.25*w,y2]]:patch.append(pool(np.array(bb)))
  features.append((idx,np.concatenate(patch+[np.array([x1/1280,y1/720,x2/1280,y2/720,w/1280,hh/720])])) )
X=np.stack([a[1] for a in sorted(features)]);np.save(S/'spatial_features.npy',X);assert np.isfinite(X).all()
results=[];pred={k:np.full(len(q),np.nan) for k in ['lr','catboost','ridge']};tp={k:[] for k in pred};real_mask=records.kind.eq('real');test_mask=records.kind.eq('test')
for fit,held in zip(DAYS,DAYS[::-1]):
 a=records.day.eq(fit).to_numpy();b=(records.day.eq(held)&real_mask).to_numpy();c=test_mask.to_numpy();assert set(records.loc[a,'frame_id']).isdisjoint(set(records.loc[b,'frame_id']))
 # Every learned transform is fit on the fitting day only.
 sc=StandardScaler();xa=sc.fit_transform(X[a]);pca=PCA(n_components=32,random_state=17,svd_solver='randomized');xa=pca.fit_transform(xa);xb=pca.transform(sc.transform(X[b]));xc=pca.transform(sc.transform(X[c]));yy=records.loc[a,'error'].to_numpy();ww=records.loc[a,'weight'].to_numpy()
 models={'lr':LogisticRegression(C=.03,max_iter=2000),'catboost':CatBoostClassifier(iterations=200,depth=2,learning_rate=.03,l2_leaf_reg=10,verbose=False,thread_count=2,random_seed=17),'ridge':Ridge(alpha=100)}
 for key,m in models.items():
  m.fit(xa,yy,sample_weight=ww);pb=m.predict_proba(xb)[:,1] if key!='ridge' else np.clip(m.predict(xb),0,1);pc=m.predict_proba(xc)[:,1] if key!='ridge' else np.clip(m.predict(xc),0,1)
  ii=q.id.map(dict(zip(records.loc[b,'id'],pb)));valid=ii.notna();pred[key][valid]=ii[valid];tp[key].append(pc);joblib.dump(dict(scaler=sc,pca=pca,model=m,fit_day=fit),S/f'quality_{key}_{fit}.joblib')
for key,pp in pred.items():
 assert np.isfinite(pp[q.class_name.eq('person')]).all()
 for w in [.1,.25,.5,1.0]:
  ss=q.score.to_numpy().copy();m=q.class_name.eq('person').to_numpy();ss[m]=(1-w)*ss[m]+w*pp[m];name=f'dino_spatial_{key}_w{w}';row={'experiment':name,'pooled_ap':float(average_precision_score(q.error,ss))}
  for day,g in q.groupby('day'):row['day_'+day[-2:]]=float(average_precision_score(g.error,ss[g.index]))
  results.append(row);q.assign(score=ss).to_csv(S/f'{name}_oof.csv',index=False);tt=base_test.copy();m=te.class_name.eq('person').to_numpy();lookup=dict(zip(records.loc[test_mask,'id'],np.mean(tp[key],axis=0)));tt[m]=(1-w)*tt[m]+w*te.loc[m,'id'].map(lookup).to_numpy();te[['id']].assign(score=tt).to_csv(S/f'submission_{name}.csv',index=False);print('DINO_SPATIAL_RESULT',json.dumps(row),flush=True)
pd.DataFrame(results).to_csv(S/'spatial_experiment_metrics.csv',index=False);(S/'spatial_manifest.json').write_text(json.dumps(dict(model=manifest['model'],revision=revision,backbone_frozen=True,features='3x3 inside ROI and four outside boundary bands; full-frame DINO patch map 768x432',synthetic_weight=.25,real_weight=2,fit_day_only=True,records=len(records),public_lb=None),indent=2));print('SPATIAL_COMPLETE',flush=True)
