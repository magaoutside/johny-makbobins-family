"""Fresh YOLO26n: fixed 20 epochs, clean manual data, local-query + full frames.
Validation belongs to the opposite UTC day; last.pt, never best held-day epoch.
"""
from common import *
import argparse,time,torch,cv2,yaml,gc
from ultralytics import YOLO,settings
import yolo_runtime
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(3);cv2.setNumThreads(1)
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune','tensorboard'] if k in settings})
p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
q,te,h=inputs();target=h[h.class_name.isin(CL)];rec=pd.read_csv(A/'inputs/records.csv');ledger=[]
data=A/'yolo_data'
for d in ['images','labels']:(data/d).mkdir(parents=True,exist_ok=True)
for fid,g in h.groupby('frame_id'):
 im=cv2.resize(cv2.imread(str(D/f'{fid}.jpg')),(1280,720));gt=target[target.frame_id.eq(fid)];crops=[np.array([0,0,1280,720])]
 for _,r in gt.iterrows():
  b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;side=max(192,float((b[2:]-b[:2]).max()*2.4));crops.append(np.clip(np.r_[cc-side/2,cc+side/2],0,[1280,720,1280,720]).astype(int))
 for k,crop in enumerate(crops):
  lo=crop[:2];size=crop[2:]-lo;lines=[]
  for _,r in gt.iterrows():
   b=r[B].to_numpy(float);bb=np.r_[np.maximum(b[:2],lo),np.minimum(b[2:],crop[2:])];wh=bb[2:]-bb[:2]
   if min(wh)<4 or wh.prod()/np.maximum((b[2:]-b[:2]).prod(),1)<.5:continue
   center=((bb[:2]+bb[2:])/2-lo)/size;wh=wh/size
   lines.append(str(CL.index(r.class_name))+' '+' '.join(f'{v:.7f}' for v in [*center,*wh]))
  name=f'{fid}_{k}';path=data/'images'/f'{name}.jpg'
  if not path.exists():cv2.imwrite(str(path),im[crop[1]:crop[3],crop[0]:crop[2]])
  (data/'labels'/f'{name}.txt').write_text('\n'.join(lines)+'\n')
  ledger.append(dict(frame_id=fid,day=day(fid),path=str(path),boxes=len(lines)))
ledger=pd.DataFrame(ledger);ledger.to_csv(A/'inputs/yolo_training_ledger.csv',index=False)
for fit in DAYS:
 train=ledger[ledger.day.eq(fit)];valid=ledger[ledger.day.ne(fit)]
 assert not set(train.frame_id)&set(valid.frame_id)
 for split,g in [('train',train),('val',valid)]: (data/f'{split}_{fit[-2:]}.txt').write_text('\n'.join(g.path)+'\n')
 (data/f'fit{fit[-2:]}.yaml').write_text(yaml.safe_dump(dict(path=str(data),train=f'train_{fit[-2:]}.txt',val=f'val_{fit[-2:]}.txt',names=dict(enumerate(CL)))))
print('DATA',len(ledger),ledger.groupby('day').size().to_dict(),flush=True)
if args.prepare_only:raise SystemExit
assert torch.backends.mps.is_available()
source=A/'model_cache/yolo26n.pt';manifests=[]
for fit in DAYS:
 name=f'fit{fit[-2:]}';checkpoint=A/f'yolo_runs/{name}/weights/last.pt';start=time.time()
 if not checkpoint.exists():
  model=YOLO(str(source))
  model.train(data=str(data/f'{name}.yaml'),epochs=20,imgsz=640,batch=8,device='mps',workers=0,optimizer='AdamW',lr0=.0003,lrf=.1,warmup_epochs=1,freeze=10,mosaic=0,mixup=0,copy_paste=0,fliplr=.0,degrees=0,translate=.05,scale=.2,seed=4101,deterministic=True,amp=False,cache=False,patience=100,val=False,plots=False,save=True,project=str(A/'yolo_runs'),name=name,exist_ok=True,verbose=False)
  del model;gc.collect();torch.mps.empty_cache()
 model=YOLO(str(checkpoint));refs=[]
 queries=pd.concat([q[q.day.ne(fit)].assign(split='query'),te.assign(split='test')],ignore_index=True)
 for n,(fid,g) in enumerate(queries.groupby('frame_id')):
  im=cv2.resize(cv2.imread(str(D/f'{fid}.jpg')),(1280,720));crops=[np.array([0,0,1280,720])]
  for _,r in g.iterrows():
   b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;size=np.maximum((b[2:]-b[:2])*2.4,128);crops.append(np.clip(np.r_[cc-size/2,cc+size/2],0,[1280,720,1280,720]).astype(int))
  for off in range(0,len(crops),8):
   cc=crops[off:off+8];ims=[im[b[1]:b[3],b[0]:b[2]].copy() for b in cc]
   predictions=model.predict(ims,imgsz=640,device='mps',conf=.08,iou=.7,verbose=False)
   for crop,pred in zip(cc,predictions):
    for box,confidence,cid in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy(),pred.boxes.cls.cpu().numpy()):
     bb=box+np.tile(crop[:2],2);refs.append(dict(frame_id=fid,class_name=CL[int(cid)],confidence=float(confidence),**dict(zip(B,bb))))
  if n%20==0:print('YOLO INFER',fit,n,queries.frame_id.nunique(),flush=True)
 # Deduplicate cross-scale references before one-to-one query matching.
 raw=pd.DataFrame(refs);kept=[]
 for (fid,c),g in raw.groupby(['frame_id','class_name']):
  picked=[]
  for j,r in g.sort_values('confidence',ascending=False).iterrows():
   if picked and iou(r[B],g.loc[picked,B]).max()>.7:continue
   picked.append(j)
  kept.extend(picked)
 ref=raw.loc[kept];ref.to_csv(A/f'features/yolo_{name}_references.csv',index=False);out=[]
 for (fid,c),g in queries.groupby(['frame_id','class_name']):
  r=ref[ref.frame_id.eq(fid)&ref.class_name.eq(c)];ov=iou(g[B],r[B]);assigned=np.zeros(len(g));best=np.zeros(len(g));confidence=np.zeros(len(g))
  if ov.size:
   rr,ss=linear_sum_assignment(-((ov>=.6)*(max(ov.shape)+1)+ov));assigned[rr]=ov[rr,ss];best=ov.max(1);confidence=r.confidence.to_numpy()[ov.argmax(1)]
  for k,(_,row) in enumerate(g.iterrows()):
   out.append(dict(id=row.id,frame_id=fid,day=row.day,class_name=c,split=row.split,fit_day=fit,best_iou=best[k],assigned_iou=assigned[k],reference_confidence=confidence[k],score=float(np.clip((.85-assigned[k])/.5,0,1))))
 pd.DataFrame(out).to_csv(A/f'features/yolo_{name}_predictions.csv',index=False)
 manifests.append(dict(fit_day=fit,epochs=20,checkpoint='last.pt',freeze=10,fit_frames=sorted(ledger.loc[ledger.day.eq(fit),'frame_id'].unique()),source='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),seconds=time.time()-start,pool_used=False))
 (A/'models/yolo_manifest.json').write_text(json.dumps(manifests,indent=2));del model;gc.collect();torch.mps.empty_cache()
print('YOLO DONE',flush=True)
