"""Kaggle worker: fixed-epoch hand refinement and rotation/scale TTA.
Training uses only complete manual annotations of the selected fit day.
"""
from pathlib import Path
import json,sys,hashlib,time,os
import numpy as np,pandas as pd,torch,yaml
from PIL import Image
from ultralytics import YOLO,settings

A=Path(__file__).resolve().parent;B=['x1','y1','x2','y2']
fit=sys.argv[1];device=int(sys.argv[2]);torch.set_num_threads(2)
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
q=pd.read_csv(A/'train_queries.csv');te=pd.read_csv(A/'test_queries.csv');gt=pd.read_csv(A/'manual_boxes.csv');frames=pd.read_csv(A/'frames.csv')
paths=pd.read_csv(A/'image_paths.csv').set_index('frame_id').path.to_dict()
trainframes=frames[frames.split.eq('train')&(True if fit=='all' else frames.day.eq(fit))]
hands=gt[gt.class_name.eq('nude_hands')]
root=A/'gpu_results/hand_refinement'/fit;root.mkdir(parents=True,exist_ok=True)
for folder in ['images','labels']:(root/folder).mkdir(exist_ok=True)
rng=np.random.default_rng(171);ledger=[]
for f in trainframes.itertuples():
    im=Image.open(paths[f.frame_id]).convert('RGB');fg=hands[hands.frame_id.eq(f.frame_id)][B].to_numpy();specs=[]
    for b in fg:
        center=(b[:2]+b[2:])/2;wh=b[2:]-b[:2]
        for scale,minimum in [(3,128),(5,192),(7,256)]:
            size=np.maximum(wh*scale,minimum)
            for shift in [np.zeros(2),rng.uniform(-.15,.15,2)*size]:specs.append((center+shift,size,'manual'))
    for b in q[q.frame_id.eq(f.frame_id)&q.class_name.eq('nude_hands')][B].to_numpy():specs.append(((b[:2]+b[2:])/2,np.maximum((b[2:]-b[:2])*5,192),'query'))
    for _ in range(4):specs.append((np.array([rng.uniform(90,1190),rng.uniform(130,520)]),np.array([192,192]),'background'))
    for j,(center,size,kind) in enumerate(specs):
        roi=np.r_[center-size/2,center+size/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720);roi=roi.astype(int)
        inter=np.maximum(0,np.minimum(fg[:,2:],roi[2:])-np.maximum(fg[:,:2],roi[:2])).prod(1) if len(fg) else np.array([])
        coverage=inter/np.maximum(np.prod(fg[:,2:]-fg[:,:2],axis=1),1) if len(fg) else np.array([])
        # Do not teach truncated palm fragments as complete hands or as background.
        if ((coverage>.01)&(coverage<.99)).any():continue
        w,hh=roi[2:]-roi[:2];name=f'{f.frame_id}_{j}';im.crop(tuple(roi)).save(root/'images'/f'{name}.jpg',quality=95);lines=[]
        for b in fg[coverage>=.99]:
            box=b-np.tile(roi[:2],2);c=(box[:2]+box[2:])/2/[w,hh];s=(box[2:]-box[:2])/[w,hh];lines.append('0 '+' '.join(map(str,np.r_[c,s])))
        (root/'labels'/f'{name}.txt').write_text('\n'.join(lines));ledger.append(dict(frame_id=f.frame_id,day=f.day,kind=kind,path=str(root/'images'/f'{name}.jpg'),hands=len(lines),roi=roi.tolist()))
pd.DataFrame(ledger).to_csv(root/'training_ledger.csv',index=False)
train=root/'train.txt';train.write_text('\n'.join(r['path'] for r in ledger)+'\n')
config=root/'data.yaml';config.write_text(yaml.safe_dump(dict(path=str(root),train=str(train),val=str(train),names={0:'nude_hands'})))
start=A/'weights'/f'hand{fit[-2:] if fit!="all" else "all"}_last.pt'
last=root/'train/weights/last.pt'
if not last.exists():
    model=YOLO(str(start));model.train(data=str(config),epochs=20,imgsz=640,batch=16,device=device,workers=2,seed=171,deterministic=True,optimizer='AdamW',lr0=.0001,lrf=.1,weight_decay=.0005,warmup_epochs=1,patience=0,val=False,amp=True,cache=False,mosaic=0,mixup=0,scale=.1,translate=.05,degrees=15,fliplr=.5,flipud=0,box=12.,cls=.5,save=True,plots=False,project=str(root),name='train',exist_ok=False,verbose=False)
    del model;torch.cuda.empty_cache()
target=te[te.class_name.eq('nude_hands')].copy() if fit=='all' else q[q.day.ne(fit)&q.class_name.eq('nude_hands')].copy()
target['kind']='query_test' if fit=='all' else 'query_oof'
if fit!='all':
    hh=hands[hands.day.ne(fit)].copy();hh['id']=['gt_'+str(i) for i in hh.index];hh['kind']='manual_held';target=pd.concat([target,hh],ignore_index=True)
def inverse(xy,turn,w,hh):
    x,y=xy[...,0].copy(),xy[...,1].copy()
    if turn==1:return np.stack([w-y,x],-1)
    if turn==2:return np.stack([w-x,hh-y],-1)
    if turn==3:return np.stack([y,hh-x],-1)
    return xy
for checkpoint,path in [('v15',start),('refined',last)]:
    dest=root/f'tta_{checkpoint}.csv'
    if dest.exists():continue
    model=YOLO(str(path));out=[]
    for n,r in enumerate(target.itertuples()):
        im=Image.open(paths[r.frame_id]).convert('RGB');b=np.array([getattr(r,c) for c in B]);center=(b[:2]+b[2:])/2
        for scale,minimum in [(3,160),(6,256)]:
            size=np.maximum((b[2:]-b[:2])*scale,minimum);roi=np.r_[center-size/2,center+size/2];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720);roi=roi.astype(int)
            crop=im.crop(tuple(roi));w,hh=crop.size
            for turn in [0,1,2,3]:
                arr=np.rot90(np.asarray(crop),turn).copy();pred=model.predict(Image.fromarray(arr),imgsz=800,device=device,conf=.05,iou=.5,max_det=30,verbose=False)[0]
                for j,(box,cf) in enumerate(zip(pred.boxes.xyxy.cpu().numpy(),pred.boxes.conf.cpu().numpy())):
                    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]);xy=inverse(corners,turn,w,hh)+roi[:2];bb=np.r_[xy.min(0),xy.max(0)]
                    out.append(dict(query_id=r.id,frame_id=r.frame_id,kind=r.kind,fit_day=fit,held_day='test' if fit=='all' else r.day,checkpoint=checkpoint,scale=scale,turn=turn,confidence=float(cf),roi_x1=roi[0],roi_y1=roi[1],roi_x2=roi[2],roi_y2=roi[3],**dict(zip(B,bb))))
        if n%15==0:print('TTA17',fit,checkpoint,n,len(target),len(out),flush=True)
    pd.DataFrame(out).to_csv(dest,index=False);del model;torch.cuda.empty_cache()
(root/'manifest.json').write_text(json.dumps(dict(fit_day=fit,epochs=20,train_frames=len(trainframes),training_crops=len(ledger),heldout_for_selection=False,initial_checkpoint_sha256=hashlib.sha256(start.read_bytes()).hexdigest(),refined_checkpoint_sha256=hashlib.sha256(last.read_bytes()).hexdigest(),tta_turns=[0,1,2,3],tta_scales=[3,6],test_labels_used=False),indent=2))
print('HAND17_DONE',fit,flush=True)
