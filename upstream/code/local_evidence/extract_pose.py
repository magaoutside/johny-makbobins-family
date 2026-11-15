from common import *
import torch
from ultralytics import YOLO
torch.set_num_threads(2)
model=YOLO(str(Path(os.environ['RASCAR_WEIGHTS'])/'pose/yolo26m-pose.pt'));paths=pd.read_csv(A/'image_paths.csv');out=[]
for n,r in enumerate(paths.itertuples()):
 result=model.predict(r.path,imgsz=1280,device=os.environ.get('RASCAR_DEVICE','cuda:0'),conf=.1,iou=.5,max_det=40,verbose=False)[0]
 for j,(bb,conf,kp) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),result.boxes.conf.cpu().numpy(),result.keypoints.data.cpu().numpy())):
  rec=dict(frame_id=r.frame_id,reference_id=f'pose_{r.frame_id}_{j}',confidence=float(conf),class_name='person',**dict(zip(B,bb)))
  for k in range(len(kp)):
   for c,value in zip(['x','y','conf'],kp[k]):rec[f'kp{k}_{c}']=float(value)
  out.append(rec)
 if n%25==0:print('POSE_PROGRESS',n,len(paths),flush=True)
pd.DataFrame(out).to_csv(A/'pose_references.csv',index=False);print('POSE_DONE',len(out),flush=True)
