"""Frozen pose at rotated full views and overlapping tiles; no Rascar fitting."""
from common import *
from PIL import Image
import torch
from ultralytics import YOLO
torch.set_num_threads(2)
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict()
model=YOLO('yolo26m-pose.pt');out=[]
def inverse_xy(xy,turn,w,h):
    x,y=xy[...,0].copy(),xy[...,1].copy()
    if turn==1:return np.stack([w-y,x],axis=-1)
    if turn==3:return np.stack([y,h-x],axis=-1)
    return xy
for i,(fid,path) in enumerate(paths.items()):
    im=Image.open(path).convert('RGB')
    specs=[(0,0,1280,720,t) for t in [0,1,3]]
    specs += [(x,y,x+800,y+500,0) for x in [0,480] for y in [0,220]]
    for vi,(x1,y1,x2,y2,turn) in enumerate(specs):
        crop=im.crop((x1,y1,x2,y2));w,hh=crop.size
        arr=np.rot90(np.asarray(crop),turn).copy()
        # PIL preserves RGB; passing ndarray would imply BGR in Ultralytics.
        r=model.predict(Image.fromarray(arr),imgsz=960,conf=.08,iou=.5,device=0,verbose=False)[0]
        for j,(box,conf,kp) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.keypoints.data.cpu().numpy())):
            corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]);xy=inverse_xy(corners,turn,w,hh)+[x1,y1]
            bb=np.r_[xy.min(0),xy.max(0)];kxy=inverse_xy(kp[:,:2],turn,w,hh)+[x1,y1]
            row=dict(frame_id=fid,view=vi,turn=turn,confidence=float(conf),reference_id=f'pose16_{fid}_{vi}_{j}',**dict(zip(B,bb)))
            for k in range(17):row.update({f'kp{k}_x':float(kxy[k,0]),f'kp{k}_y':float(kxy[k,1]),f'kp{k}_conf':float(kp[k,2])})
            out.append(row)
    if i%10==0:print('POSE16_PROGRESS',i,len(paths),flush=True)
pd.DataFrame(out).to_csv(A/'multiview_pose.csv',index=False)
print('POSE16_DONE',len(out),flush=True)
