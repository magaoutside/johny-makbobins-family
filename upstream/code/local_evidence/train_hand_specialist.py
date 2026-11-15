from common import *
import sys,torch,yaml,hashlib
from PIL import Image
from ultralytics import YOLO,settings
fit=sys.argv[1];device=int(sys.argv[2]);torch.set_num_threads(2)
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict()
root=A/'hand_specialist'/fit;root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True);(root/'labels').mkdir(exist_ok=True)
frames=fr[fr.frame_id.isin(h.frame_id)&(True if fit=='all' else fr.day.eq(fit))]
handgt=h[h.class_name.eq('nude_hands')];ledger=[];rng=np.random.default_rng(17)
for f in frames.itertuples():
 im=Image.open(paths[f.frame_id]).convert('RGB');fg=handgt[handgt.frame_id.eq(f.frame_id)][B].to_numpy();centers=[]
 for b in fg:
  for scale,minimum in [(3,128),(5,192),(7,256)]:centers.append(((b[2:]+b[:2])/2,np.maximum((b[2:]-b[:2])*scale,minimum),'gt'))
 for b in q[(q.frame_id==f.frame_id)&(q.class_name=='nude_hands')][B].to_numpy():centers.append(((b[2:]+b[:2])/2,np.maximum((b[2:]-b[:2])*4,160),'query'))
 # Empty scenes and forearm/table neighborhoods supply full-annotation negatives.
 for _ in range(3):centers.append((np.array([rng.uniform(80,1200),rng.uniform(210,450)]),np.array([192,192]),'background'))
 for j,(center,wh,kind) in enumerate(centers):
  roi=np.r_[center-wh/2,center+wh/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720);roi=roi.astype(int);w,hg=roi[2:]-roi[:2]
  stem=f'{f.frame_id}_{j}';im.crop(tuple(roi)).save(root/'images'/f'{stem}.jpg',quality=95);labels=[]
  for b in fg:
   bb=np.r_[np.maximum(b[:2],roi[:2]),np.minimum(b[2:],roi[2:])]-np.tile(roi[:2],2)
   if min(bb[2:]-bb[:2])<2:continue
   xy=(bb[:2]+bb[2:])/2/np.array([w,hg]);size=(bb[2:]-bb[:2])/np.array([w,hg]);labels.append('0 '+' '.join(map(str,np.r_[xy,size])))
  (root/'labels'/f'{stem}.txt').write_text('\n'.join(labels));ledger.append(dict(frame_id=f.frame_id,day=f.day,kind=kind,path=str(root/'images'/f'{stem}.jpg'),hands=len(labels),roi=roi.tolist()))
pd.DataFrame(ledger).to_csv(root/'training_ledger.csv',index=False);train=root/'train.txt';train.write_text('\n'.join(r['path'] for r in ledger)+'\n')
config=root/'data.yaml';config.write_text(yaml.safe_dump(dict(path=str(root),train=str(train),val=str(train),names={0:'nude_hands'})))
model=YOLO(str(A/'weights/yolo26s.pt'))
if not (root/'train/weights/last.pt').exists():
 model.train(data=str(config),epochs=35,imgsz=640,batch=16,device=device,workers=2,seed=17,deterministic=True,optimizer='AdamW',lr0=.0003,lrf=.1,weight_decay=.0005,warmup_epochs=2,patience=0,val=False,amp=True,cache=False,mosaic=.15,close_mosaic=5,mixup=0,scale=.15,translate=.05,fliplr=.5,flipud=0,save=True,plots=False,project=str(root),name='train',exist_ok=False,verbose=False)
model=YOLO(str(root/'train/weights/last.pt'));target=te if fit=='all' else q[q.day.ne(fit)];out=[]
for n,r in enumerate(target[target.class_name.eq('nude_hands')].itertuples()):
 im=Image.open(paths[r.frame_id]).convert('RGB');b=np.array([getattr(r,c) for c in B]);center=(b[2:]+b[:2])/2
 for scale,minimum in [(3,128),(6,224)]:
  wh=np.maximum((b[2:]-b[:2])*scale,minimum);roi=np.r_[center-wh/2,center+wh/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720);roi=roi.astype(int)
  result=model.predict(im.crop(tuple(roi)),imgsz=640,device=device,conf=.02,iou=.5,max_det=100,verbose=False)[0]
  for j,(bb,cf) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy())):out.append(dict(query_id=r.id,frame_id=r.frame_id,fit_day=fit,held_day='test' if fit=='all' else r.day,scale=scale,class_name='nude_hands',confidence=float(cf),reference_id=f'hand_{r.id}_{scale}_{j}',**dict(zip(B,bb+np.tile(roi[:2],2)))))
pd.DataFrame(out).to_csv(root/'references.csv',index=False)
(root/'manifest.json').write_text(json.dumps(dict(fit_day=fit,epochs=35,frames=len(frames),crops=len(ledger),seed=17,checkpoint='last.pt',test_labels_used=False,heldout_for_early_stopping=False),indent=2))
print('HAND_SPECIALIST_DONE',fit,len(ledger),len(out),flush=True)
