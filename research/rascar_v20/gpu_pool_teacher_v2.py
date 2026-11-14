"""Independent geometric + semantic checks of machine candidates.
All fitting-day pool data stay within their day. Calibration uses the other
day's real query labels and is never counted as additional ground truth.
"""
from pathlib import Path
import os,sys,json,hashlib,gc
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,pandas as pd,torch,joblib
from PIL import Image
from ultralytics import YOLO,settings
from transformers import AutoImageProcessor,AutoModel
from kaggle_secrets import UserSecretsClient
A=Path('/kaggle/working/rascar_v20');D=Path('/kaggle/working/competition_data')
fit,device=sys.argv[1:3];device=int(device);held='2026-10-26' if fit.endswith('25') else '2026-10-25'
B=['x1','y1','x2','y2'];CL=['person','cup','nude_hands','tray'];torch.set_num_threads(2)
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
out=A/'pool_v2'/fit;out.mkdir(parents=True,exist_ok=True)
paths={p.stem:p for p in D.rglob('*.jpg')}
def image1280(fid):return Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)
def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-9)
pool=pd.read_csv(A/'inputs/pool_frame_ledger.csv');pool=pool[pool.day.eq(fit)&pool.eligible_time.eq(True)]
machine=pd.read_csv(A/'inputs/machine_predictions.csv');machine=machine[machine.frame_id.isin(pool.frame_id)&machine.class_name.isin(CL)].copy()
machine['id']=machine.detection_id;machine['day']=fit;machine['source']='machine_pool';machine['error']=np.nan
q=pd.read_csv(A/'inputs/baseline_oof.csv');cal=q[q.day.eq(held)].copy();cal['source']='held_day_calibration'
rows=pd.concat([machine,cal],ignore_index=True)
target=sorted(set(rows.frame_id));assert all(f in paths for f in target)
refs=pd.read_csv(A/'pool'/fit/'multiclass_references.csv')
e=[]
for r in rows.itertuples():
    b=np.array([getattr(r,c) for c in B]);rr=refs[refs.frame_id.eq(r.frame_id)&refs.class_name.eq(r.class_name)&refs.confidence.ge(.6)]
    ov=iou(b,rr[B])[0] if len(rr) else np.array([]);j=int(ov.argmax()) if len(ov) else 0
    e.append(dict(teacher_iou=float(ov[j]) if len(ov) else 0.,teacher_conf=float(rr.iloc[j].confidence) if len(ov) else 0.,
                  teacher_boxes=len(rr),min_side=float(min(b[2:]-b[:2])),hand_iou=0.,hand_conf=0.,hand_stability=0.))
rows=pd.concat([rows.reset_index(drop=True),pd.DataFrame(e)],axis=1)
hand=YOLO(A/f'weights/hand_{fit}.pt')
for n,(ix,r) in enumerate(rows[rows.class_name.eq('nude_hands')&(rows.confidence.ge(.75)|rows.source.eq('held_day_calibration'))].iterrows()):
    im=image1280(r.frame_id);b=r[B].to_numpy(float);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2;pred=[]
    for context in [4,6]:
        side=max(float(wh.max()*context),192);x1=max(0,int(center[0]-side/2));y1=max(0,int(center[1]-side/2));x2=min(1280,int(center[0]+side/2));y2=min(720,int(center[1]+side/2))
        crop=np.array(im.crop((x1,y1,x2,y2)))
        rr=hand.predict(crop[:,:,::-1],imgsz=800,device=device,conf=.15,iou=.5,verbose=False)[0]
        bb=rr.boxes.xyxy.cpu().numpy()+[x1,y1,x1,y1];cc=rr.boxes.conf.cpu().numpy();ov=iou(b,bb)[0] if len(bb) else np.array([])
        if len(ov):j=int(ov.argmax());pred.append((bb[j],float(cc[j]),float(ov[j])))
    if len(pred)==2:
        rows.loc[ix,['hand_iou','hand_conf','hand_stability']]=[min(p[2] for p in pred),min(p[1] for p in pred),float(iou(pred[0][0],pred[1][0])[0,0])]
    if n%100==0:print('POOL_HAND',fit,n,flush=True)
del hand;gc.collect();torch.cuda.empty_cache()
rows['semantic_conf']=0.
candidate=rows[((rows.confidence.ge(.75)&rows.min_side.ge(16)))|rows.source.eq('held_day_calibration')]
model_name='facebook/dinov3-vitb16-pretrain-lvd1689m';revision='5931719e67bbdb9737e363e781fb0c67687896bc'
token=UserSecretsClient().get_secret('HF_TOKEN')
processor=AutoImageProcessor.from_pretrained(model_name,revision=revision,token=token)
model=AutoModel.from_pretrained(model_name,revision=revision,token=token,dtype=torch.float16).to(f'cuda:{device}').eval();del token
skip=1+model.config.num_register_tokens;semantic=joblib.load(A/f'models/semantic_both_{fit}.joblib')
for n in range(0,len(candidate),24):
    batch=candidate.iloc[n:n+24];ims=[]
    for r in batch.itertuples():
        im=image1280(r.frame_id);b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
        for context in [0,.35]:
            bb=b+np.array([-1,-1,1,1])*np.tile(wh,2)*context;bb[[0,2]]=np.clip(bb[[0,2]],0,1280);bb[[1,3]]=np.clip(bb[[1,3]],0,720)
            ims.append(im.crop(tuple(bb)).resize((224,224),Image.Resampling.BICUBIC))
    inp=processor(images=ims,do_resize=False,do_center_crop=False,return_tensors='pt').to(f'cuda:{device}',torch.float16)
    with torch.inference_mode():
        z=model(**inp).last_hidden_state.float();cls=torch.nn.functional.normalize(z[:,0],dim=-1);mean=torch.nn.functional.normalize(z[:,skip:].mean(1),dim=-1)
        features=(torch.cat([cls,mean],dim=-1).cpu().numpy().reshape(len(batch),-1)/np.sqrt(2)).astype(np.float32)
    p=semantic.predict_proba(features);lookup={c:j for j,c in enumerate(semantic.classes_)}
    rows.loc[batch.index,'semantic_conf']=[p[j,lookup[c]] for j,c in enumerate(batch.class_name)]
    if n%240==0:print('POOL_DINO',fit,n,len(candidate),flush=True)
del model;gc.collect();torch.cuda.empty_cache()
# Fixed stringent rule; lowering it requires a separately recorded experiment.
rows['accepted']=(rows.confidence.ge(.85)&rows.teacher_conf.ge(.7)&rows.teacher_iou.ge(.8)&rows.semantic_conf.ge(.85)&rows.min_side.ge(16))
hands=rows.class_name.eq('nude_hands')
rows.loc[hands,'accepted'] = rows.loc[hands,'confidence'].ge(.8)&rows.loc[hands,'min_side'].ge(16)&rows.loc[hands,'semantic_conf'].ge(.85)&rows.loc[hands,'hand_iou'].ge(.8)&rows.loc[hands,'hand_conf'].ge(.65)&rows.loc[hands,'hand_stability'].ge(.85)
# Ambiguous same-class overlapping source boxes are not pseudo ground truth.
rows['overlapping_source']=False
for (source,fid,c),g in rows.groupby(['source','frame_id','class_name']):
    ov=iou(g[B],g[B]);np.fill_diagonal(ov,0);m=ov.max(1)>.4;rows.loc[g.index[m],'overlapping_source']=True
rows.loc[rows.overlapping_source,'accepted']=False
rows.to_csv(out/'pseudo_label_ledger.csv',index=False)
acc=rows[rows.accepted&rows.source.eq('machine_pool')].copy();acc.to_csv(out/'accepted_pseudo_boxes.csv',index=False)
calib=rows[rows.source.eq('held_day_calibration')];summary=[]
for c,g in calib.groupby('class_name'):
    aa=g[g.accepted];summary.append(dict(class_name=c,accepted=len(aa),incorrect=int(aa.error.sum()),precision=float(1-aa.error.mean()) if len(aa) else None,total=len(g)))
(out/'summary.json').write_text(json.dumps(dict(fit=fit,held=held,pool_frames=len(target)-cal.frame_id.nunique(),proposals=len(machine),accepted_boxes=len(acc),accepted_frames=acc.frame_id.nunique(),counts=acc.class_name.value_counts().to_dict(),calibration=summary),indent=2))
# Deterministic inspection sample, stratified by class and confidence.
sample=[]
for c,g in acc.groupby('class_name'):
    g=g.sort_values(['teacher_iou','id']);ix=np.unique(np.linspace(0,len(g)-1,min(24,len(g))).astype(int));sample.append(g.iloc[ix])
review=pd.concat(sample,ignore_index=True) if sample else acc;review.to_csv(out/'visual_review_sample.csv',index=False)
from PIL import ImageDraw
for start in range(0,len(review),24):
    canvas=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(canvas)
    for j,r in enumerate(review.iloc[start:start+24].itertuples()):
        im=image1280(r.frame_id);b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];bb=b+np.array([-1,-1,1,1])*np.tile(np.maximum(wh*.4,30),2)
        ImageDraw.Draw(im).rectangle(tuple(b),outline='red',width=2);im=im.crop(tuple(bb));im.thumbnail((294,170));x=j%4*300;y=j//4*200
        canvas.paste(im,(x+(300-im.width)//2,y+28));draw.text((x+4,y+4),f'{start+j} {r.class_name} IoU {r.teacher_iou:.2f}',fill='black')
    canvas.save(out/f'review_{start//24}.jpg',quality=90)
print('POOL_DONE',fit,len(acc),acc.class_name.value_counts().to_dict(),flush=True)
