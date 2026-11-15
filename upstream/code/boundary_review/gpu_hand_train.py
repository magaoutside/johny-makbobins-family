"""Window/hard-negative fine-tuning with a continuation control and zoom control."""
from pathlib import Path
import os,sys,json,math,gc
import numpy as np,pandas as pd,torch,yaml
from PIL import Image,ImageEnhance,ImageFilter
from ultralytics import YOLO,settings
A=Path(__file__).resolve().parent;fit=sys.argv[1];held='2026-10-26' if fit.endswith('25') else '2026-10-25'
roots=json.loads((A/'roots.json').read_text());paths={p.stem:p for p in Path(roots['images']).rglob('*.jpg')}
B=['x1','y1','x2','y2'];torch.set_num_threads(2)
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
out=A/'gpu_results/hand_training'/fit;out.mkdir(parents=True,exist_ok=True)
h=pd.read_csv(A/'inputs/manual_boxes.csv');h=h[h.class_name.eq('nude_hands')&h.day.eq(fit)]
pool=pd.read_csv(A/f'inputs/approved_pseudo_boxes_{fit}.csv');pool=pool[pool.class_name.eq('nude_hands')]
machine=pd.read_csv(A/'inputs/machine_predictions.csv');machine=machine[machine.class_name.eq('nude_hands')]
review=pd.read_csv(A/'inputs/reviewed_pool_hands.csv');review=review[review.day.eq(fit)]
rc=pd.read_csv(A/'inputs/reviewed_pool_crops.csv');rc=rc[rc.day.eq(fit)]
q=pd.read_csv(A/'inputs/baseline_oof.csv');te=pd.read_csv(A/'inputs/test_queries.csv')
def coverage(crop,boxes):
 boxes=np.asarray(boxes,float).reshape(-1,4);inter=np.maximum(0,np.minimum(crop[2:],boxes[:,2:])-np.maximum(crop[:2],boxes[:,:2])).prod(1)
 return inter/np.maximum((boxes[:,2:]-boxes[:,:2]).prod(1),1e-9)
ledger=[];rng=np.random.default_rng(2202)
def save_crop(fid,crop,boxes,source,j,transform='none'):
 im=Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC).crop(tuple(crop));w,hh=im.size
 if transform=='bright':im=ImageEnhance.Brightness(im).enhance(1.5)
 elif transform=='dark':im=ImageEnhance.Brightness(im).enhance(.65)
 elif transform=='blur':im=im.filter(ImageFilter.GaussianBlur(.65))
 elif transform=='contrast':im=ImageEnhance.Contrast(im).enhance(1.5)
 name=f'{source}_{fid}_{j}_{transform}';ip=out/'dataset/images'/f'{name}.jpg';lp=out/'dataset/labels'/f'{name}.txt';ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
 im.save(ip,quality=95);lines=[]
 for bb in boxes:
  cc=((bb[:2]+bb[2:])/2-crop[:2])/[w,hh];wh=(bb[2:]-bb[:2])/[w,hh];lines.append('0 '+' '.join(f'{v:.8f}' for v in [*cc,*wh]))
 lp.write_text('\n'.join(lines)+'\n');ledger.append(dict(frame_id=fid,day=fit,source=source,transform=transform,path=str(ip),boxes=len(boxes)))
for source,gt in [('manual',h),('pseudo',pool)]:
 for fid,g in gt.groupby('frame_id'):
  boxes=g[B].to_numpy();unknown=machine[machine.frame_id.eq(fid)&~machine.detection_id.isin(pool.id)][B].to_numpy() if source=='pseudo' else np.empty((0,4));crops=[]
  for bb in boxes:
   cc=(bb[:2]+bb[2:])/2
   for size in [192,288]:
    for delta in [np.zeros(2),rng.uniform(-.12,.12,2)*size]:crops.append(np.r_[cc+delta-size/2,cc+delta+size/2])
  if source=='manual':
   for _ in range(6):
    cc=rng.uniform([128,128],[1152,592]);crops.append(np.r_[cc-128,cc+128])
  for j,cr in enumerate(crops):
   cr=np.clip(cr,0,[1280,720,1280,720]).astype(int);cv=coverage(cr,boxes)
   if ((cv>.01)&(cv<.99)).any():continue
   if len(unknown) and (coverage(cr,unknown)>.01).any():continue
   bb=boxes[cv>=.99]
   if source=='pseudo' and not len(bb):continue
   save_crop(fid,cr,bb,source,j)
   # Additional brightness/blur examples target verified small left-seat hands.
   if source=='manual' and len(bb) and ((bb[:,0]<350)&(bb[:,1]<400)).any():
    for trans in ['bright','dark','blur','contrast']:save_crop(fid,cr,bb,'manual_window_aug',j,trans)
for r in rc.itertuples():
 cr=np.array([r.roi_x1,r.roi_y1,r.roi_x2,r.roi_y2]);bb=review[review.frame_id.eq(r.frame_id)][B].to_numpy();assert (coverage(cr,bb)>=.99).all()
 for trans in ['none','bright','dark','blur']:
  save_crop(r.frame_id,cr,bb,r.source,r.review_index,trans)
ledger=pd.DataFrame(ledger);ledger.to_csv(out/'training_ledger.csv',index=False)
original=list(Path(roots['checkpoints']).glob(f'**/yolo_hands/{fit}/expanded/weights/last.pt'));assert len(original)==1
checkpoints={};manifests=[]
for variant in ['targeted']:
 chosen=ledger[ledger.source.isin(['manual','pseudo'])] if variant=='control' else ledger
 # Preserve at least half manual examples; added pool crops stay a small part.
 lines=chosen.path.tolist();manual=chosen[chosen.source.isin(['manual','manual_window_aug'])].path.tolist()
 if len(manual)*2<len(lines):lines+=manual*int((len(lines)-len(manual))/max(len(manual),1))
 listing=out/f'{variant}.txt';listing.write_text('\n'.join(lines)+'\n')
 config=out/f'{variant}.yaml';config.write_text(yaml.safe_dump(dict(path=str(out/'dataset'),train=str(listing),val=str(listing),names={0:'nude_hands'})))
 steps_per_epoch=math.ceil(len(lines)/16);epochs=max(4,min(20,math.ceil(240/steps_per_epoch)))
 model=YOLO(original[0])
 if not (out/variant/'weights/last.pt').exists():model.train(data=str(config),epochs=epochs,imgsz=640,batch=16,device=os.environ.get('RASCAR_DEVICE','cuda:0'),workers=2,seed=2202,deterministic=True,optimizer='AdamW',lr0=.00005,lrf=.2,weight_decay=.0005,warmup_epochs=1,patience=0,val=False,amp=True,cache=False,mosaic=0,mixup=0,scale=.2,translate=.05,degrees=10,fliplr=.5,save=True,plots=False,project=str(out),name=variant,exist_ok=False,verbose=False)
 checkpoints[variant]=out/variant/'weights/last.pt';manifests.append(dict(variant=variant,fit=fit,held=held,crops=len(chosen),training_entries=len(lines),epochs=epochs,approx_optimizer_updates=epochs*steps_per_epoch,by_source=chosen.source.value_counts().to_dict()))
 del model;gc.collect();torch.cuda.empty_cache()
 print('HAND_TRAINED',fit,variant,flush=True)
queries=pd.concat([q[q.day.eq(held)].assign(split='held'),te.assign(split='test')],ignore_index=True);queries=queries[queries.class_name.eq('nude_hands')]
for variant,checkpoint in checkpoints.items():
 model=YOLO(checkpoint);refs=[]
 for n,r in enumerate(queries.itertuples()):
  im=Image.open(paths[r.frame_id]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC);b=np.array([getattr(r,c) for c in B]);cc=(b[:2]+b[2:])/2
  for zoom in [96,160,256]:
   size=max(zoom,float((b[2:]-b[:2]).max()*2));cr=np.clip(np.r_[cc-size/2,cc+size/2],0,[1280,720,1280,720]).astype(int)
   image=np.array(im.crop(tuple(cr)))[:,:,::-1].copy();p=model.predict(image,imgsz=640,conf=.08,iou=.5,device=os.environ.get('RASCAR_DEVICE','cuda:0'),verbose=False)[0]
   bb=p.boxes.xyxy.cpu().numpy()+np.tile(cr[:2],2);conf=p.boxes.conf.cpu().numpy()
   for j,(box,c) in enumerate(zip(bb,conf)):
    if min(box[:2]-cr[:2])<2 or min(cr[2:]-box[2:])<2:continue
    refs.append(dict(query_id=r.id,frame_id=r.frame_id,class_name='nude_hands',split=r.split,fit=fit,zoom=zoom,confidence=float(c),reference_id=f'{variant}_{fit}_{r.id}_{zoom}_{j}',**dict(zip(B,box))))
  if n%40==0:print('HAND_INFER',fit,variant,n,len(queries),flush=True)
 pd.DataFrame(refs).to_csv(out/f'{variant}_zoom_references.csv',index=False)
 del model;gc.collect();torch.cuda.empty_cache()
(out/'manifest.json').write_text(json.dumps(manifests,indent=2));print('HAND_JOB_DONE',fit,flush=True)
