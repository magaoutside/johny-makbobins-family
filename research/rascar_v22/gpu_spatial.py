"""Frozen public DINOv2 patch features; cross-day spatial quality heads."""
from pathlib import Path
import json,os,gc,time
import numpy as np,pandas as pd,torch,joblib
from PIL import Image
from transformers import AutoModel
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
A=Path('/kaggle/working/rascar_v22');O=A/'spatial';O.mkdir(exist_ok=True)
roots=json.loads((A/'roots.json').read_text());paths={p.stem:p for p in Path(roots['images']).rglob('*.jpg')}
d=pd.read_csv(A/'inputs/spatial_records.csv');B=['x1','y1','x2','y2'];torch.set_num_threads(3)
model=AutoModel.from_pretrained('facebook/dinov2-base').cuda().eval();features=[]
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
def crop(r):
    b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];cr=np.r_[b[:2]-.5*wh,b[2:]+.5*wh]
    im=Image.open(paths[r.frame_id]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)
    return np.asarray(im.crop(tuple(cr)).resize((224,224),Image.Resampling.BICUBIC)).copy()
records=list(d.itertuples())
with torch.inference_mode():
 for start in range(0,len(d),48):
    x=torch.from_numpy(np.stack([crop(r) for r in records[start:start+48]])).permute(0,3,1,2).to('cuda',dtype=torch.float32)/255
    with torch.autocast('cuda',dtype=torch.float16):z=model(pixel_values=(x-mean)/std).last_hidden_state.float()
    patch=z[:,1:].reshape(-1,16,16,768)
    parts=[z[:,0],patch[:,4:12,4:12].mean((1,2)),patch[:,:4,4:12].mean((1,2)),patch[:,12:,4:12].mean((1,2)),patch[:,4:12,:4].mean((1,2)),patch[:,4:12,12:].mean((1,2))]
    features.append(torch.stack([torch.nn.functional.normalize(v,dim=1) for v in parts],1).cpu().numpy().astype('float16'))
    if start%480==0:print('SPATIAL_FEATURES',start,len(d),flush=True)
f=np.concatenate(features);np.save(O/'features.npy',f);del model;gc.collect();torch.cuda.empty_cache()
outputs=[];metrics=[]
for fit,held in [('2026-10-25','2026-10-26'),('2026-10-26','2026-10-25')]:
 for cls in ['person','nude_hands','cup','tray']:
  tr=d.index[d.split.eq('train')&d.day.eq(fit)&d.class_name.eq(cls)].to_numpy()
  ev=d.index[d.class_name.eq(cls)&(d.split.eq('test')|(d.split.eq('held')&d.day.eq(held)))].to_numpy()
  y=d.loc[tr,'error'].to_numpy(int);weight=d.loc[tr,'weight'].to_numpy(float);groups=d.loc[tr,'frame_id'].to_numpy()
  for representation,parts in [('inside',[0,1]),('spatial',list(range(6)))]:
   x=f[:,parts,:].reshape(len(f),-1).astype('float32')/np.sqrt(len(parts))
   for algorithm in ['logreg','svm']:
    def make():
     return LogisticRegression(C=12,max_iter=1500,class_weight='balanced') if algorithm=='logreg' else SVC(C=6,gamma=2,class_weight='balanced')
    cv=np.zeros(len(tr))
    for i,j in GroupKFold(3).split(tr,y,groups):
     m=make();m.fit(x[tr[i]],y[i],sample_weight=weight[i]);cv[j]=m.decision_function(x[tr[j]])
    cal=LogisticRegression(C=2,max_iter=1000);cal.fit(cv[:,None],y,sample_weight=weight)
    m=make();m.fit(x[tr],y,sample_weight=weight);raw=m.decision_function(x[ev]);pred=cal.predict_proba(raw[:,None])[:,1]
    name=f'{fit}_{cls}_{representation}_{algorithm}'
    joblib.dump(dict(model=m,calibrator=cal,parts=parts,feature_model='facebook/dinov2-base',training_indices=tr),O/f'{name}.joblib',compress=3)
    oo=d.loc[ev,['id','frame_id','day','class_name','split','error']].copy();oo['fit']=fit;oo['representation']=representation;oo['algorithm']=algorithm;oo['probability']=pred;outputs.append(oo)
    mask=oo.split.eq('held');ap=average_precision_score(oo.loc[mask,'error'],pred[mask]);metrics.append(dict(fit=fit,held=held,class_name=cls,representation=representation,algorithm=algorithm,ap=ap,group_cv_ap=average_precision_score(y,cv),ntrain=len(tr)))
    pd.concat(outputs).to_csv(O/'predictions.csv',index=False);pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False)
    print('SPATIAL_HEAD',name,round(ap,5),flush=True)
print('SPATIAL_DONE',flush=True)
