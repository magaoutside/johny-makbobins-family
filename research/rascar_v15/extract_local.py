from common import *
from PIL import Image
import torch
from ultralytics import YOLO
torch.set_num_threads(2)
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict();out=[]
for fit,held,rows in [(DAYS[0],DAYS[1],q[q.day.eq(DAYS[1])]),(DAYS[1],DAYS[0],q[q.day.eq(DAYS[0])]),('all','test',te)]:
 model=YOLO(str(A/f'weights/{fit}.pt'))
 for n,r in enumerate(rows[rows.class_name.isin(['nude_hands','cup'])].itertuples()):
  im=Image.open(paths[r.frame_id]).convert('RGB');b=np.array([getattr(r,c) for c in B]);center=(b[2:]+b[:2])/2
  for scale,minimum in [(3.,128),(6.,224)]:
   wh=np.maximum((b[2:]-b[:2])*scale,minimum);roi=np.r_[center-wh/2,center+wh/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720);roi=roi.astype(int)
   result=model.predict(im.crop(tuple(roi)),imgsz=960,device=0,conf=.025,iou=.5,max_det=100,verbose=False)[0]
   for j,(bb,conf,c) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.boxes.cls.cpu().numpy())):
    out.append(dict(query_id=r.id,frame_id=r.frame_id,fit_day=fit,held_day=held,scale=scale,class_name=model.names[int(c)],confidence=float(conf),reference_id=f'local_{r.id}_{scale}_{j}',**dict(zip(B,bb+np.tile(roi[:2],2)))))
  if n%25==0:print('LOCAL_CROPS',fit,n,flush=True)
 pd.DataFrame(out).to_csv(A/'local_references.csv',index=False)
 del model;torch.cuda.empty_cache()
print('LOCAL_DONE',len(out),flush=True)
