"""New pretrained hand evidence. No RASCAR labels are used in inference."""
from pathlib import Path
import sys,json,time,hashlib,urllib.request,zipfile,gc
import numpy as np,pandas as pd,cv2,torch
A=Path('/kaggle/working/rascar_v25');B=['x1','y1','x2','y2'];kind=sys.argv[1]
cv2.setNumThreads(2);torch.set_num_threads(2)
paths=json.loads((A/'image_paths.json').read_text());queries=pd.read_csv(A/'queries.csv');persons=pd.read_csv(A/'persons.csv')
frames=sorted(queries.frame_id.unique());out=A/kind;out.mkdir(exist_ok=True)
missing=[x for x in frames if x not in paths];assert not missing,missing[:10]
records=[];poses=[];start=time.time()
def image(fid):return cv2.resize(cv2.imread(paths[fid]),(1280,720))
def save():
 pd.DataFrame(records).to_csv(out/'references.csv',index=False)
 if poses:pd.DataFrame(poses).to_csv(out/'keypoints.csv',index=False)
 (out/'progress.json').write_text(json.dumps({'seconds':time.time()-start,'detections':len(records),'poses':len(poses)}))
def rot_points(x,turn,w,h):
 return np.array([w,h])-x if turn==2 else x
if kind=='rtmw':
 from rtmlib import Wholebody
 # performance = 384x288 RTMW-l distilled from x, not old 21-point RTMPose-hand.
 model=Wholebody(mode='performance',to_openpose=False,backend='onnxruntime',device='cuda')
 print('RTMW_MODEL',model.pose_model.__dict__.keys(),flush=True)
 for fi,fid in enumerate(frames):
  im=image(fid)
  for turn in [0,2]:
   pic=np.rot90(im,turn).copy();rb=model.det_model(pic)
   # Add independent pretrained person references (same-frame, no manual boxes).
   extra=persons[persons.frame_id.eq(fid)][B].to_numpy()
   if turn:extra=np.c_[1280-extra[:,2],720-extra[:,3],1280-extra[:,0],720-extra[:,1]]
   candidates=list(np.asarray(rb).reshape(-1,4))
   for b in extra:
    if all(np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])).prod()/max((b[2:]-b[:2]).prod()+(c[2:]-c[:2]).prod()-np.maximum(0,np.minimum(b[2:],c[2:])-np.maximum(b[:2],c[:2])).prod(),1)<.6 for c in candidates):candidates.append(b)
   if not candidates:continue
   kp,sc=model.pose_model(pic,bboxes=np.array(candidates))
   for pi,(xy,co) in enumerate(zip(kp,sc)):
    xy=rot_points(xy,turn,1280,720)
    for side,sl in [('left',slice(91,112)),('right',slice(112,133))]:
     pts=xy[sl];conf=co[sl];good=conf>=.2
     poses.append(dict(frame_id=fid,turn=turn,person=pi,side=side,xy=json.dumps(pts.tolist()),confidence=json.dumps(conf.tolist()),body_xy=json.dumps(xy[:17].tolist()),body_conf=json.dumps(co[:17].tolist())))
     if good.sum()<8:continue
     bb=np.r_[pts[good].min(0),pts[good].max(0)];wh=bb[2:]-bb[:2]
     if min(wh)<2:continue
     records.append(dict(frame_id=fid,query_id='',variant='rtmw_wholebody',turn=turn,side=side,confidence=float(np.median(conf)),points_above02=int(good.sum()),**dict(zip(B,bb))))
  if fi%10==0:save();print('RTMW_PROGRESS',fi,len(frames),len(records),flush=True)
elif kind=='wilor_detector':
 from ultralytics import YOLO
 url='https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt';path=out/'detector.pt'
 if not path.exists():urllib.request.urlretrieve(url,path)
 model=YOLO(path);print('WILOR_DETECTOR',model.names,flush=True)
 def run(pic,fid,qid,variant,origin=np.zeros(2),turn=0,size=640):
  hh,ww=pic.shape[:2];view=np.rot90(pic,turn).copy()
  pred=model.predict(view,conf=.10,iou=.5,imgsz=size,device=0,verbose=False)[0]
  for b,co in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy()):
   bb=np.array(b)
   if turn:bb=np.r_[ww-b[2],hh-b[3],ww-b[0],hh-b[1]]
   if variant!='full' and (min(bb[:2])<2 or min(np.array([ww,hh])-bb[2:])<2):continue
   bb+=np.tile(origin,2)
   records.append(dict(frame_id=fid,query_id=qid,variant=variant,turn=turn,confidence=float(co),points_above02=0,**dict(zip(B,bb))))
 for fi,fid in enumerate(frames):
  im=image(fid);run(im,fid,'','full',size=1280)
  for r in queries[queries.frame_id.eq(fid)&queries.class_name.eq('nude_hands')].itertuples():
   b=np.array([getattr(r,c) for c in B]);cc=(b[:2]+b[2:])/2
   for width in [128,224]:
    width=max(width,float((b[2:]-b[:2]).max()*2));cr=np.clip(np.r_[cc-width/2,cc+width/2],0,[1280,720,1280,720]).astype(int)
    for turn in [0,2]:run(im[cr[1]:cr[3],cr[0]:cr[2]],fid,r.id,f'local{width}',cr[:2],turn)
  if fi%10==0:save();print('WILOR_PROGRESS',fi,len(frames),len(records),flush=True)
 (out/'model_manifest.json').write_text(json.dumps({'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':'Detector only; no MANO mesh model'},indent=2))
else:raise ValueError(kind)
save();(out/'DONE.json').write_text(json.dumps({'frames':len(frames),'references':len(records),'seconds':time.time()-start,'kind':kind,'cuda':torch.cuda.get_device_name(0)},indent=2));print('BENCHMARK_DONE',kind,len(records),flush=True)
