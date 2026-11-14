"""Independent multi-scale sliced references for cups and trays, frozen fold YOLO."""
from pathlib import Path
import sys,json,gc,os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import numpy as np,pandas as pd,torch
torch.cuda.init()
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
A=Path('/kaggle/working/rascar_v20');D=Path('/kaggle/working/competition_data');fit,device=sys.argv[1:3];device=int(device)
held='2026-10-26' if fit.endswith('25') else '2026-10-25';B=['x1','y1','x2','y2'];torch.set_num_threads(2)
q=pd.read_csv(A/'inputs/baseline_oof.csv');te=pd.read_csv(A/'inputs/test_queries.csv');frames=sorted(set(q[q.day.eq(held)&q.class_name.isin(['cup','tray'])].frame_id)|set(te[te.class_name.isin(['cup','tray'])].frame_id))
paths={p.stem:p for p in D.rglob('*.jpg')};out=A/'cup_tray'/fit;out.mkdir(parents=True,exist_ok=True)
model=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(A/f'weights/multiclass_{fit}.pt'),confidence_threshold=.08,device=f'cuda:{device}',image_size=960)
refs=[]
for n,fid in enumerate(frames):
 im=Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)
 for tile in [384,640]:
  result=get_sliced_prediction(np.asarray(im),model,slice_height=tile,slice_width=tile,overlap_height_ratio=.3,overlap_width_ratio=.3,perform_standard_pred=False,postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,verbose=0)
  for j,p in enumerate(result.object_prediction_list):
   if p.category.name not in ['cup','tray']:continue
   refs.append(dict(frame_id=fid,class_name=p.category.name,confidence=float(p.score.value),tile=tile,reference_id=f'{fit}_{fid}_{tile}_{j}',**dict(zip(B,p.bbox.to_xyxy()))))
 if n%20==0:print('CUP_TRAY_SAHI',fit,n,len(frames),flush=True)
pd.DataFrame(refs).to_csv(out/'references.csv',index=False)
(out/'manifest.json').write_text(json.dumps(dict(fit=fit,held=held,checkpoint=f'multiclass_{fit}.pt',frozen=True,tiles=[384,640]),indent=2))
print('CUP_TRAY_DONE',fit,flush=True)
del model;gc.collect();torch.cuda.empty_cache()
# Same-checkpoint unsliced controls isolate the effect of slicing.
from ultralytics import YOLO
hand_frames=sorted(set(q[q.day.eq(held)].frame_id)|set(te.frame_id))
for variant,checkpoint in [('frozen',A/f'weights/hand_{fit}.pt'),('clean',A/'yolo_hands'/fit/'clean/weights/last.pt'),('expanded',A/'yolo_hands'/fit/'expanded/weights/last.pt')]:
 model=YOLO(checkpoint);raw=[]
 for fid,r in zip(hand_frames,model.predict([str(paths[f]) for f in hand_frames],imgsz=1280,batch=4,device=device,conf=.08,iou=.5,verbose=False,stream=True)):
  for j,(bb,co) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy())):
   bb=bb*np.array([1280/r.orig_shape[1],720/r.orig_shape[0]]*2);raw.append(dict(frame_id=fid,class_name='nude_hands',confidence=float(co),tile=1280,reference_id=f'{variant}_{fid}_{j}',**dict(zip(B,bb))))
 pd.DataFrame(raw).to_csv(A/'yolo_hands'/fit/f'{variant}_unsliced_references.csv',index=False)
 del model;gc.collect();torch.cuda.empty_cache()
model=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(A/f'weights/hand_{fit}.pt'),confidence_threshold=.08,device=f'cuda:{device}',image_size=800)
refs=[]
for n,fid in enumerate(hand_frames):
 im=Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)
 for tile in [256,384]:
  result=get_sliced_prediction(np.asarray(im),model,slice_height=tile,slice_width=tile,overlap_height_ratio=.25,overlap_width_ratio=.25,perform_standard_pred=False,postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,verbose=0)
  for j,p in enumerate(result.object_prediction_list):refs.append(dict(frame_id=fid,class_name='nude_hands',confidence=float(p.score.value),tile=tile,reference_id=f'frozen_{fit}_{fid}_{tile}_{j}',**dict(zip(B,p.bbox.to_xyxy()))))
 if n%20==0:print('FROZEN_SAHI_CONTROL',fit,n,len(hand_frames),flush=True)
pd.DataFrame(refs).to_csv(A/'yolo_hands'/fit/'frozen_sahi_references.csv',index=False)
print('ALL_CONTROLS_DONE',fit,flush=True)
