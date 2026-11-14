"""Fine-tune existing YOLO26 hand detector; compare clean/pool; SAHI inference.
Pseudo crops are rejected if they intersect any unverified hand proposal.
"""
from pathlib import Path
import os,sys,json,gc,random
import numpy as np,pandas as pd,torch,yaml
from PIL import Image
from ultralytics import YOLO,settings
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
A=Path('/kaggle/working/rascar_v20');D=Path('/kaggle/working/competition_data')
fit,device=sys.argv[1:3];device=int(device);held='2026-10-26' if fit.endswith('25') else '2026-10-25'
days=['2026-10-25','2026-10-26'] if fit=='all' else [fit]
B=['x1','y1','x2','y2'];torch.set_num_threads(2)
settings.update({k:False for k in ['wandb','mlflow','comet','clearml','dvc','neptune'] if k in settings})
out=A/'yolo_hands'/fit;out.mkdir(parents=True,exist_ok=True);paths={p.stem:p for p in D.rglob('*.jpg')}
h=pd.read_csv(A/'inputs/manual_boxes.csv');h=h[h.class_name.eq('nude_hands')&h.day.isin(days)]
pool=pd.concat([pd.read_csv(A/'pool'/d/'approved_pseudo_boxes.csv') for d in days],ignore_index=True);pool=pool[pool.class_name.eq('nude_hands')]
all_machine=pd.read_csv(A/'inputs/machine_predictions.csv');all_machine=all_machine[all_machine.class_name.eq('nude_hands')]
q=pd.read_csv(A/'inputs/baseline_oof.csv');te=pd.read_csv(A/'inputs/test_queries.csv')
def coverage(crop,boxes):
    boxes=np.asarray(boxes,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(crop[2:],boxes[:,2:])-np.maximum(crop[:2],boxes[:,:2])).prod(1)
    return inter/np.maximum((boxes[:,2:]-boxes[:,:2]).prod(1),1e-9)
ledger=[];rng=np.random.default_rng(201)
for source,gt in [('manual',h),('pseudo',pool)]:
    for fid,g in gt.groupby('frame_id'):
        im=Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC);boxes=g[B].to_numpy(float)
        unknown=all_machine[all_machine.frame_id.eq(fid)&~all_machine.detection_id.isin(pool.id)][B].to_numpy(float) if source=='pseudo' else np.empty((0,4))
        crops=[]
        for b in boxes:
            wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2
            for scale in [4,6]:
                side=max(192,float(wh.max()*scale));crops.append(np.r_[center-side/2,center+side/2]);shift=rng.uniform(-.15,.15,2)*side;crops.append(np.r_[center+shift-side/2,center+shift+side/2])
        if source=='manual':
            for _ in range(4):
                side=256;cc=rng.uniform([128,128],[1152,592]);crops.append(np.r_[cc-side/2,cc+side/2])
        for ix,crop in enumerate(crops):
            crop=np.clip(crop,0,[1280,720,1280,720]).astype(int);cv=coverage(crop,boxes)
            if ((cv>.01)&(cv<.99)).any():continue
            if len(unknown) and (coverage(crop,unknown)>.01).any():continue
            included=boxes[cv>=.99];w,hh=crop[2:]-crop[:2]
            if min(w,hh)<100 or (source=='pseudo' and not len(included)):continue
            name=f'{source}_{fid}_{ix}';imagepath=out/'dataset/images'/f'{name}.jpg';labelpath=out/'dataset/labels'/f'{name}.txt';imagepath.parent.mkdir(parents=True,exist_ok=True);labelpath.parent.mkdir(parents=True,exist_ok=True)
            im.crop(tuple(crop)).save(imagepath,quality=95);lines=[]
            for bb in included:
                cc=((bb[:2]+bb[2:])/2-crop[:2])/[w,hh];wh=(bb[2:]-bb[:2])/[w,hh];lines.append('0 '+' '.join(f'{v:.8f}' for v in [*cc,*wh]))
            labelpath.write_text('\n'.join(lines)+'\n');ledger.append(dict(frame_id=fid,source=source,path=str(imagepath),boxes=len(included)))
ledger=pd.DataFrame(ledger);ledger.to_csv(out/'training_ledger.csv',index=False)
for variant in ['clean','expanded']:
    selected=ledger[ledger.source.eq('manual')] if variant=='clean' else ledger
    # Repeat manual crops to preserve at least half the expanded training set.
    lines=selected.path.tolist()
    if variant=='expanded':
        mm=ledger[ledger.source.eq('manual')].path.tolist();extra=len(selected)-len(mm)
        if len(mm) and extra>len(mm):lines+=mm*(int(extra/len(mm))-1)
    listpath=out/f'{variant}_train.txt';listpath.write_text('\n'.join(lines)+'\n')
    config=out/f'{variant}.yaml';config.write_text(yaml.safe_dump(dict(path=str(out/'dataset'),train=str(listpath),val=str(listpath),names={0:'nude_hands'})))
    model=YOLO(A/f'weights/hand_{fit}.pt')
    model.train(data=str(config),epochs=12,imgsz=640,batch=16,device=device,workers=2,seed=201,deterministic=True,optimizer='AdamW',lr0=.00005,lrf=.2,weight_decay=.0005,warmup_epochs=1,patience=0,val=False,amp=True,cache=False,mosaic=0,mixup=0,scale=.1,translate=.05,degrees=10,fliplr=.5,save=True,plots=False,project=str(out),name=variant,exist_ok=False,verbose=False)
    del model;gc.collect();torch.cuda.empty_cache()
    checkpoint=out/variant/'weights/last.pt'
    detection_model=AutoDetectionModel.from_pretrained(model_type='ultralytics',model_path=str(checkpoint),confidence_threshold=.08,device=f'cuda:{device}',image_size=800)
    frames=sorted(set(te.frame_id)|set(q[q.day.eq(held)].frame_id)) if fit!='all' else sorted(set(te.frame_id))
    refs=[]
    for n,fid in enumerate(frames):
        im=Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)
        for tile in [256,384]:
            result=get_sliced_prediction(np.asarray(im),detection_model,slice_height=tile,slice_width=tile,overlap_height_ratio=.25,overlap_width_ratio=.25,perform_standard_pred=False,postprocess_type='NMS',postprocess_match_metric='IOU',postprocess_match_threshold=.5,verbose=0)
            for j,p in enumerate(result.object_prediction_list):
                bb=p.bbox.to_xyxy();refs.append(dict(frame_id=fid,class_name='nude_hands',confidence=float(p.score.value),tile=tile,reference_id=f'{variant}_{fit}_{fid}_{tile}_{j}',**dict(zip(B,bb))))
        if n%20==0:print('SAHI_HAND',fit,variant,n,len(frames),flush=True)
    pd.DataFrame(refs).to_csv(out/f'{variant}_sahi_references.csv',index=False)
    del detection_model;gc.collect();torch.cuda.empty_cache()
(out/'manifest.json').write_text(json.dumps(dict(fit=fit,epochs=12,manual_crops=int(ledger.source.eq('manual').sum()),pseudo_crops=int(ledger.source.eq('pseudo').sum()),pseudo_boxes=len(pool),seed=201,held_day_in_training=False,partial_or_unknown_hand_crops_rejected=True,sahi_tiles=[256,384]),indent=2))
print('YOLO_HAND_DONE',fit,flush=True)
