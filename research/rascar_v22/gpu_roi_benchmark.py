"""Independent left-seat detector diagnostic on every held-day annotated frame."""
from pathlib import Path
import json,time,sys,gc
import pandas as pd,numpy as np,torch
from PIL import Image
from ultralytics import YOLO
A=Path('/kaggle/working/rascar_v22');DAYS=['2026-10-25','2026-10-26'];torch.set_num_threads(2)
for attempt in range(240):
 logs=[(A/f'hand_{day}.log').read_text() for day in DAYS]
 if all('HAND_JOB_DONE' in t for t in logs):break
 if any('Traceback (most recent call last)' in t for t in logs):raise RuntimeError('Hand job failed')
 time.sleep(10)
else:raise TimeoutError()
roots=json.loads((A/'roots.json').read_text());paths={p.stem:p for p in Path(roots['images']).rglob('*.jpg')}
h=pd.read_csv(A/'inputs/manual_boxes.csv');B=['x1','y1','x2','y2'];cr=np.array([64,152,400,424]);out=[]
for fit,held in zip(DAYS,DAYS[::-1]):
 for variant in ['frozen','control','targeted']:
  if variant=='frozen':p=list(Path(roots['checkpoints']).glob(f'**/yolo_hands/{fit}/expanded/weights/last.pt'))[0]
  else:p=A/'hand_training'/fit/variant/'weights/last.pt'
  model=YOLO(p)
  for fid in h[h.day.eq(held)].frame_id.unique():
   im=np.asarray(Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC).crop(tuple(cr)))[:,:,::-1].copy()
   for scale in [640,960]:
    pred=model.predict(im,imgsz=scale,conf=.05,iou=.5,device=0,verbose=False)[0]
    for bb,co in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy()):
     bb+=np.tile(cr[:2],2)
     out.append(dict(frame_id=fid,class_name='nude_hands',fit=fit,day=held,variant=variant,scale=scale,confidence=float(co),**dict(zip(B,bb))))
  print('ROI_BENCH',fit,variant,flush=True);del model;gc.collect();torch.cuda.empty_cache()
pd.DataFrame(out).to_csv(A/'roi_benchmark.csv',index=False)
print('ROI_BENCH_DONE',flush=True)
