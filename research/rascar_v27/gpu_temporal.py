"""Frozen WiLoR detector + bidirectional optical flow on neighbouring frames.

Inference only. No labels, audit decisions or scores are read by this script.
Current-frame TTA is retained as a separate control for temporal evidence.
"""
from pathlib import Path
import numpy as np,pandas as pd,cv2,json,time,torch,urllib.request
from ultralytics import YOLO
A=Path('/kaggle/working/rascar_v27');B=['x1','y1','x2','y2']
paths=json.loads((A/'image_paths.json').read_text());q=pd.read_csv(A/'queries.csv')
q=q[q.class_name.eq('nude_hands')].copy();cv2.setNumThreads(2);torch.set_num_threads(2)
checkpoint=A/'wilor_detector.pt'
if not checkpoint.exists():urllib.request.urlretrieve('https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt',checkpoint)
model=YOLO(checkpoint)
frames=sorted(paths);stamps={f:pd.to_datetime(f[-15:],format='%Y%m%d_%H%M%S').timestamp() for f in frames if f.startswith('GENERAL_')}
out=[];neighbours=[];started=time.time()
def crop(fid,roi):
    im=cv2.resize(cv2.imread(paths[fid]),(1280,720))
    return im[roi[1]:roi[3],roi[0]:roi[2]].copy()
def unrotate(bb,k,w,h):
    pts=np.array([[bb[0],bb[1]],[bb[2],bb[1]],[bb[2],bb[3]],[bb[0],bb[3]]])
    if k==1:pts=np.c_[w-pts[:,1],pts[:,0]]
    elif k==2:pts=np.c_[w-pts[:,0],h-pts[:,1]]
    elif k==3:pts=np.c_[pts[:,1],h-pts[:,0]]
    return np.r_[pts.min(0),pts.max(0)]
def transfer(source,target,b):
    a=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY);z=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    w,h=b[2:]-b[:2]
    xx,yy=np.meshgrid(np.linspace(b[0]+.1*w,b[2]-.1*w,7),np.linspace(b[1]+.1*h,b[3]-.1*h,7))
    p=np.c_[xx.ravel(),yy.ravel()].astype('float32').reshape(-1,1,2)
    fw,ok,_=cv2.calcOpticalFlowPyrLK(a,z,p,None,winSize=(15,15),maxLevel=3,criteria=(3,25,.01))
    if fw is None:return None
    back,ok2,_=cv2.calcOpticalFlowPyrLK(z,a,fw,None,winSize=(15,15),maxLevel=3,criteria=(3,25,.01))
    if back is None:return None
    fb=np.linalg.norm(p-back,axis=2).ravel();valid=ok.ravel().astype(bool)&ok2.ravel().astype(bool)&(fb<1.5)
    valid&=(fw[:,:,0].ravel()>=2)&(fw[:,:,0].ravel()<z.shape[1]-2)&(fw[:,:,1].ravel()>=2)&(fw[:,:,1].ravel()<z.shape[0]-2)
    if valid.sum()<18:return None
    matrix,inliers=cv2.estimateAffinePartial2D(p[valid],fw[valid],method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=500)
    if matrix is None:return None
    scale=np.sqrt(np.linalg.det(matrix[:,:2]));frac=float(inliers.mean())
    if not .7<scale<1.4 or frac<.65:return None
    corners=np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]])
    warped=corners@matrix[:,:2].T+matrix[:,2];bb=np.r_[warped.min(0),warped.max(0)]
    # Photometric agreement is evaluated over the full hand patch under warp.
    gridx,gridy=np.meshgrid(np.linspace(b[0],b[2],24),np.linspace(b[1],b[3],24))
    old=np.c_[gridx.ravel(),gridy.ravel()];new=old@matrix[:,:2].T+matrix[:,2]
    aa=cv2.remap(a,old[:,0].astype('float32').reshape(24,24),old[:,1].astype('float32').reshape(24,24),cv2.INTER_LINEAR).astype(float).ravel()
    zz=cv2.remap(z,new[:,0].astype('float32').reshape(24,24),new[:,1].astype('float32').reshape(24,24),cv2.INTER_LINEAR).astype(float).ravel()
    if min(aa.std(),zz.std())<5:return None
    ncc=float(np.corrcoef(aa,zz)[0,1]);err=float(np.median(abs(aa-zz)))
    return bb,dict(fb_error=float(np.median(fb[valid])),tracked_fraction=float(valid.mean()),inlier_fraction=frac,ncc=ncc,pixel_error=err,scale=scale)

for index,r in enumerate(q.itertuples()):
    cache=A/'cache'/f'{r.id}.json';cache.parent.mkdir(exist_ok=True)
    if cache.exists():out.extend(json.loads(cache.read_text()));continue
    box=np.array([getattr(r,c) for c in B]);cc=(box[:2]+box[2:])/2;width=max(160,float((box[2:]-box[:2]).max()*2.5))
    roi=np.r_[cc-width/2,cc+width/2].clip([0,0,0,0],[1280,720,1280,720]).astype(int)
    target=crop(r.frame_id,roi);hh,ww=target.shape[:2];now=stamps[r.frame_id]
    nearby=[f for f,t in stamps.items() if f[-15:-7]==r.frame_id[-15:-7] and 0<abs(t-now)<=65]
    nearby=sorted(nearby,key=lambda f:abs(stamps[f]-now))[:4]
    neighbours.append(dict(id=r.id,frame_id=r.frame_id,neighbours=';'.join(nearby),count=len(nearby)))
    records=[]
    for fid in [r.frame_id]+nearby:
        source=crop(fid,roi);views=[np.rot90(source,k).copy() for k in range(4)]
        predictions=model.predict(views,conf=.15,iou=.45,imgsz=640,device=0,verbose=False)
        for k,pred in enumerate(predictions):
            for b,confidence in zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy()):
                bb=unrotate(b,k,ww,hh)
                if min(bb[:2])<3 or min(np.array([ww,hh])-bb[2:])<3:continue
                if fid==r.frame_id:
                    transferred=bb;flow=dict(fb_error=0.,tracked_fraction=1.,inlier_fraction=1.,ncc=1.,pixel_error=0.,scale=1.)
                else:
                    result=transfer(source,target,bb)
                    if result is None:continue
                    transferred,flow=result
                globalbox=transferred+np.tile(roi[:2],2);sourcebox=bb+np.tile(roi[:2],2)
                records.append(dict(id=r.id,frame_id=r.frame_id,source_frame=fid,seconds=stamps[fid]-now,turn=k,confidence=float(confidence),
                    **flow,**dict(zip(B,globalbox)),**dict(zip(['source_'+c for c in B],sourcebox))))
    cache.write_text(json.dumps(records));out.extend(records)
    if index%10==0:
        pd.DataFrame(out).to_csv(A/'temporal_references.csv',index=False)
        print('TEMPORAL',index,len(q),'rows',len(out),'seconds',round(time.time()-started),flush=True)
pd.DataFrame(out).to_csv(A/'temporal_references.csv',index=False);pd.DataFrame(neighbours).to_csv(A/'neighbours.csv',index=False)
(A/'DONE.json').write_text(json.dumps(dict(queries=len(q),references=len(out),seconds=time.time()-started,gpu=torch.cuda.get_device_name(0),labels_used=False,source='WiLoR detector + OpenCV bidirectional LK affine transfer')))
print('TEMPORAL_DONE',len(out),flush=True)
