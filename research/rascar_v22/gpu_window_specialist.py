"""Stronger targeted adaptation, balanced by the original manual crops."""
from pathlib import Path
import sys,json,math,gc
import pandas as pd,numpy as np,torch,yaml
from PIL import Image
from ultralytics import YOLO
A=Path('/kaggle/working/rascar_v22');fit=sys.argv[1];held='2026-10-26' if fit.endswith('25') else '2026-10-25'
roots=json.loads((A/'roots.json').read_text());paths={p.stem:p for p in Path(roots['images']).rglob('*.jpg')};B=['x1','y1','x2','y2'];torch.set_num_threads(2)
out=A/'hand_training'/fit;ledger=pd.read_csv(out/'training_ledger.csv')
chosen=ledger.copy();chosen['repetitions']=chosen.source.map({'manual':2,'pseudo':1,'manual_window_aug':4,'reviewed_pool_focused':6,'reviewed_pool_negative':6}).fillna(1).astype(int)
chosen.to_csv(out/'window_sampling.csv',index=False);lines=[]
for r in chosen.itertuples():lines += [r.path]*r.repetitions
listing=out/'window_specialist.txt';listing.write_text('\n'.join(lines)+'\n')
config=out/'window_specialist.yaml';config.write_text(yaml.safe_dump(dict(path=str(out/'dataset'),train=str(listing),val=str(listing),names={0:'nude_hands'})))
original=list(Path(roots['checkpoints']).glob(f'**/yolo_hands/{fit}/expanded/weights/last.pt'))[0]
epochs=max(4,math.ceil(500/math.ceil(len(lines)/16)))
m=YOLO(original);m.train(data=str(config),epochs=epochs,imgsz=640,batch=16,device=0,workers=2,seed=2213,deterministic=True,optimizer='AdamW',lr0=.0002,lrf=.1,weight_decay=.001,warmup_epochs=1,patience=0,val=False,amp=True,cache=False,mosaic=0,mixup=0,scale=.15,translate=.05,degrees=7,fliplr=.5,save=True,plots=False,project=str(out),name='window_specialist',exist_ok=False,verbose=False)
del m;gc.collect();torch.cuda.empty_cache();m=YOLO(out/'window_specialist/weights/last.pt')
q=pd.read_csv(A/'inputs/baseline_oof.csv');te=pd.read_csv(A/'inputs/test_queries.csv');queries=pd.concat([q[q.day.eq(held)].assign(split='held'),te.assign(split='test')]);refs=[]
for r in queries[queries.class_name.eq('nude_hands')].itertuples():
 im=Image.open(paths[r.frame_id]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC);b=np.array([getattr(r,c) for c in B]);cc=(b[:2]+b[2:])/2
 for zoom in [96,160,256]:
  size=max(zoom,float((b[2:]-b[:2]).max()*2));cr=np.clip(np.r_[cc-size/2,cc+size/2],0,[1280,720,1280,720]).astype(int)
  pred=m.predict(np.array(im.crop(tuple(cr)))[:,:,::-1].copy(),imgsz=640,conf=.08,iou=.5,device=0,verbose=False)[0]
  for j,(bb,co) in enumerate(zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy())):
   bb+=np.tile(cr[:2],2)
   if min(bb[:2]-cr[:2])<2 or min(cr[2:]-bb[2:])<2:continue
   refs.append(dict(query_id=r.id,frame_id=r.frame_id,class_name='nude_hands',split=r.split,fit=fit,zoom=zoom,confidence=float(co),reference_id=f'window_{fit}_{r.id}_{zoom}_{j}',**dict(zip(B,bb))))
pd.DataFrame(refs).to_csv(out/'window_specialist_zoom_references.csv',index=False)
h=pd.read_csv(A/'inputs/manual_boxes.csv');cr=np.array([64,152,400,424]);refs=[]
for fid in h[h.day.eq(held)].frame_id.unique():
 im=np.array(Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC).crop(tuple(cr)))[:,:,::-1].copy()
 for scale in [640,960]:
  pred=m.predict(im,imgsz=scale,conf=.05,iou=.5,device=0,verbose=False)[0]
  for bb,co in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy()):
   bb+=np.tile(cr[:2],2);refs.append(dict(frame_id=fid,class_name='nude_hands',fit=fit,day=held,variant='window_specialist',scale=scale,confidence=float(co),**dict(zip(B,bb))))
pd.DataFrame(refs).to_csv(out/'window_roi_benchmark.csv',index=False)
(out/'window_manifest.json').write_text(json.dumps(dict(fit=fit,held=held,epochs=epochs,training_entries=len(lines),lr0=.0002,seed=2213,scope='same-day manual and reviewed pool only'),indent=2))
print('WINDOW_DONE',fit,flush=True)
