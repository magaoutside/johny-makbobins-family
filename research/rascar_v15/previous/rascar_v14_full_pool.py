from pathlib import Path
exec(Path('/kaggle/working/rascar_local_data.py').read_text())

from pathlib import Path
import os,sys,json,hashlib,time
os.environ['DISABLE_KAGGLE_CACHE']='true';os.environ['OMP_NUM_THREADS']='2'
import numpy as np,pandas as pd,kagglehub,torch,cv2
from PIL import Image,ImageDraw
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from scipy.special import expit
E=Path('/kaggle/working/rascar_v14');O=Path('/kaggle/working/oof_validation_v11');Y=E/'yolo26_clean';S=E/'stage2';P=E/'stage6/pool';P.mkdir(parents=True,exist_ok=True)
B=['x1','y1','x2','y2'];days=['2026-10-25','2026-10-26'];torch.set_num_threads(2)
q=pd.read_csv(O/'train.csv');fr=pd.read_csv(O/'train_frames.csv');fr['day']=pd.to_datetime(fr.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d');q=q.merge(fr[['frame_id','day']],on='frame_id');q=q[q.class_name.isin(['person','cup','nude_hands','tray'])].reset_index(drop=True)
def overlap(a,b):
 a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4);inter=np.prod(np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])),axis=2);return inter/np.maximum(np.prod(a[:,2:]-a[:,:2],axis=1)[:,None]+np.prod(b[:,2:]-b[:,:2],axis=1)[None]-inter,1e-9)
# Select a time-diverse machine sample before looking at scores.
mf=pd.read_csv(S/'machine_frames.csv');mp=pd.read_csv(S/'machine_predictions.csv');mf['timestamp']=pd.to_datetime(mf.timestamp,utc=True,format='mixed');mf['day']=mf.timestamp.dt.strftime('%Y-%m-%d');mf['hour']=mf.timestamp.dt.floor('h')
assert set(mf.day).issubset(set(days))
train_times=pd.to_datetime(fr.timestamp,utc=True,format='mixed').astype('int64').to_numpy();mt=mf.timestamp.astype('int64').to_numpy();nearest=np.min(abs(mt[:,None]-train_times[None,:]),axis=1)/1e9
mf=mf[nearest>600].copy();sample=pd.concat([g.sample(min(8,len(g)),random_state=17) for _,g in mf.groupby('hour')],ignore_index=True);sample.to_csv(P/'sample_frames.csv',index=False)
print('POOL_SCHEMA',mp.columns.tolist(),mp.head(1).to_dict('records'),'sample',len(sample),flush=True)
imgdir=P/'images';imgdir.mkdir(exist_ok=True)
full=Path(kagglehub.competition_download('rascar-hack-case-1',output_dir=str(E/'full_competition')))
print('FULL_COMPETITION_LOCATION',str(full),flush=True)
lookup={p.name:p for p in full.rglob('*.jpg')};print('FULL_IMAGE_FILES',len(lookup),flush=True)

download_errors=[]
def fetch(row):
 try:
  path=lookup.get(Path(row.image_path).name)
  if path is None:raise FileNotFoundError(row.image_path)
  return str(path)
 except Exception as e:download_errors.append({'frame_id':row.frame_id,'image_path':row.image_path,'error_type':type(e).__name__});return None
with ThreadPoolExecutor(max_workers=6) as pool:paths=list(pool.map(fetch,sample.itertuples(index=False)))
sample['local_path']=paths;(P/'unavailable_images.json').write_text(json.dumps(download_errors,indent=2));sample=sample[sample.local_path.notna()].copy();sample.to_csv(P/'sample_frames.csv',index=False);print('AVAILABLE_POOL_FRAMES',len(sample),'unavailable',len(download_errors),flush=True)
ledger=[];allpred=[];review=[];seen=set();number=0
for fit in days:
 model=YOLO(str(Y/f'fit_{fit}/train/weights/last.pt'));part=sample[sample.day.eq(fit)]
 if part.empty:continue
 for result in model.predict(part.local_path.tolist(),imgsz=1280,device=1,conf=.05,iou=.7,max_det=300,stream=True,verbose=False):
  fid=Path(result.path).stem;im=Image.open(result.path).convert('RGB');native=np.asarray(im);gray=cv2.cvtColor(native,cv2.COLOR_RGB2GRAY);sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());mean=float(gray.mean());sha=hashlib.sha256(Path(result.path).read_bytes()).hexdigest();quality=sharp>=20 and 25<=mean<=235 and sha not in seen;seen.add(sha)
  ledger.append(dict(frame_id=fid,day=fit,width=im.width,height=im.height,sharpness=sharp,brightness=mean,sha256=sha,quality_ok=quality))
  scale=np.array([1280/im.width,720/im.height]*2);rb=result.boxes.xyxy.cpu().numpy()*scale;conf=result.boxes.conf.cpu().numpy();classes=[model.names[int(c)] for c in result.boxes.cls.cpu().numpy()]
  for j,b in enumerate(rb):allpred.append(dict(frame_id=fid,fit_day=fit,class_name=classes[j],confidence=float(conf[j]),**dict(zip(B,b))))
  # Only cup/tray crops are proposed for a specialist detector. Person/hands are not pseudo-labeled.
  if not quality:continue
  source=mp[(mp.frame_id==fid)&mp.class_name.isin(['cup','tray'])&mp.confidence.ge(.7)]
  for _,r in source.iterrows():
   b=r[B].to_numpy(float);m=np.array([c==r.class_name for c in classes],dtype=bool)&(conf>=.65)
   if not m.any() or min(b[2:]-b[:2])<24:continue
   candidate_boxes=rb[m];ovs=overlap(b,candidate_boxes)[0];j=int(np.argmax(ovs))
   if ovs[j]<.75:continue
   # Query and independently trained teacher must closely agree on all edges.
   t=candidate_boxes[j];bb=(b+t)/2;wh=bb[2:]-bb[:2];center=(bb[2:]+bb[:2])/2;roi=np.r_[center-np.maximum(wh*1.8,80)/2,center+np.maximum(wh*1.8,80)/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
   # Reject regions containing additional cup/tray proposals: review must verify completeness.
   others=mp[(mp.frame_id==fid)&mp.class_name.isin(['cup','tray'])]
   other_boxes=others[B].to_numpy();inter=np.prod(np.maximum(0,np.minimum(other_boxes[:,2:],roi[2:])-np.maximum(other_boxes[:,:2],roi[:2])),axis=1)
   if ((inter>0)&(overlap(bb,other_boxes)[0]<.5)).any():continue
   teacher_other=np.array([c in ['cup','tray'] for c in classes],dtype=bool)&(conf>=.1);obs=rb[teacher_other]
   if len(obs):
    inter2=np.prod(np.maximum(0,np.minimum(obs[:,2:],roi[2:])-np.maximum(obs[:,:2],roi[:2])),axis=1)
    if ((inter2>0)&(overlap(bb,obs)[0]<.5)).any():continue
   number+=1;review.append(dict(review_id=number,frame_id=fid,day=fit,class_name=r.class_name,source_confidence=float(r.confidence),teacher_confidence=float(conf[m][j]),agreement_iou=float(ovs[j]),roi=roi.tolist(),bbox=bb.tolist(),image_path=result.path,label_status='unreviewed_silver_candidate'))
 del model
pd.DataFrame(ledger).to_csv(P/'image_quality_ledger.csv',index=False);pd.DataFrame(allpred).to_csv(P/'teacher_predictions.csv',index=False)
# At most two candidates from one frame and 30 per day; deterministic diversity selection.
chosen=[]
for day in days:
 count={}
 for r in sorted([r for r in review if r['day']==day],key=lambda r:hashlib.sha256(str(r['review_id']).encode()).hexdigest()):
  if count.get(r['frame_id'],0)>=2:continue
  chosen.append(r);count[r['frame_id']]=count.get(r['frame_id'],0)+1
  if sum(c['day']==day for c in chosen)>=30:break
(P/'review_candidates.json').write_text(json.dumps(chosen,indent=2));(P/'all_eligible_candidates.json').write_text(json.dumps(review,indent=2))
for page in range((len(chosen)+11)//12):
 canvas=Image.new('RGB',(1200,960),'white');draw=ImageDraw.Draw(canvas)
 for k,r in enumerate(chosen[page*12:(page+1)*12]):
  im=Image.open(r['image_path']).convert('RGB');sc=np.array([im.width/1280,im.height/720]*2);roi=np.array(r['roi']);native_roi=(roi*sc).astype(int);crop=im.crop(tuple(native_roi));factor=min(290/crop.width,270/crop.height);crop=crop.resize((max(1,int(crop.width*factor)),max(1,int(crop.height*factor))),Image.Resampling.NEAREST);x=(k%4)*300;y=(k//4)*320;canvas.paste(crop,(x,y+35));bx=(np.array(r['bbox'])-np.tile(roi[:2],2))*np.array([crop.width/(roi[2]-roi[0]),crop.height/(roi[3]-roi[1])]*2);draw.rectangle(tuple((bx+np.array([x,y+35,x,y+35])).astype(int)),outline='red',width=2);draw.text((x+4,y+4),f"#{r['review_id']} {r['class_name']} {r['day'][-2:]}",fill='black')
 canvas.save(P/f'review_page_{page:02d}.jpg',quality=95)
print('POOL_REVIEW_READY',len(review),'eligible',len(chosen),'selected',flush=True)
print('POOL_LABELS_NOT_USED_UNTIL_VISUAL_REVIEW',flush=True)
