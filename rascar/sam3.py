"""Frozen SAM 3.1 full-frame detections, mask bounds and temporal confirmation."""
from pathlib import Path
import hashlib
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from .foundation_scoring import BBOX, SCORED_CLASSES, box_iou, match_from_iou


def table(raw, name):
    files = list(Path(raw).rglob(name))
    if len(files) != 1:
        raise ValueError(f'Expected one {name}, found {len(files)}')
    return pd.read_csv(files[0])


def extract(raw, weights, out, device='cuda:0'):
    import torch
    from ultralytics.models.sam import SAM3SemanticPredictor
    raw, weights, out = map(Path, (raw, weights, out))
    out.mkdir(parents=True, exist_ok=True)
    cache = out/'sam3_frames'; cache.mkdir(exist_ok=True)
    images = {p.stem: p for p in raw.rglob('*.jpg')}
    prompts = dict(person='person', cup='cup', nude_hands='bare human hand', tray='tray', table='table')
    checkpoint = weights/'sam3/sam3.1_multiplex.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError('Run download_weights.py --models sam3 first')
    settings = dict(model=str(checkpoint), task='segment', mode='predict', conf=.15,
                    iou=.7, imgsz=1008, quantize=16, device=device, save=False, verbose=False, compile=False)
    signature = hashlib.sha256((json.dumps({k:v for k,v in settings.items() if k!='model'}, sort_keys=True)
                               +json.dumps(prompts,sort_keys=True)+hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()).encode()).hexdigest()
    predictor = SAM3SemanticPredictor(overrides=settings)
    predictor.setup_model(verbose=False)
    predictor.model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    contexts = {}; required = set()
    for split in ['train', 'test']:
        ctx = table(raw, split+'_context.csv')
        q = table(raw, split+'.csv')
        q.class_name = q.class_name.replace({'nude_hand':'nude_hands'})
        centers = q[q.class_name.isin(SCORED_CLASSES)].frame_id
        neighbors = set(ctx[ctx.frame_id.isin(centers)][['previous','next']].stack().astype(str))
        contexts[split] = (ctx, neighbors, set(centers))
        required.update(ctx.frame_id.astype(str)); required.update(neighbors)
    all_rows = []
    for n, fid in enumerate(sorted(required), 1):
        key = hashlib.sha256(fid.encode()).hexdigest()[:16]
        path = cache/(key+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['signature'] != signature or saved['frame_id'] != fid:
                raise ValueError(f'Incompatible SAM cache {path}')
            records = saved['records']
        else:
            records = []
            predictor.reset_image(); predictor.set_image(str(images[fid]))
            with torch.inference_mode():
                for cls, prompt in prompts.items():
                    r = predictor(text=[prompt])[0]
                    if r.masks is None:
                        if len(r.boxes): raise ValueError('SAM returned boxes without masks')
                        continue
                    boxes = r.boxes.xyxy.cpu().numpy(); scores = r.boxes.conf.cpu().numpy()
                    masks = r.masks.data.cpu().numpy().astype(bool)
                    if masks.shape[1:] != tuple(r.orig_shape): raise ValueError('SAM mask size mismatch')
                    for j, (b, confidence, mask) in enumerate(zip(boxes, scores, masks)):
                        y, x = np.nonzero(mask)
                        mb = [float(x.min()),float(y.min()),float(x.max()+1),float(y.max()+1)] if len(x) else [None]*4
                        records.append(dict(frame_id=fid,class_name=cls,reference_id=f'{key}_{cls}_{j}',
                            confidence=float(confidence),mask_area=int(mask.sum()),
                            **dict(zip(BBOX,map(float,b))),**dict(zip(['mask_'+k for k in BBOX],mb))))
            temp = path.with_suffix('.partial'); temp.write_text(json.dumps(dict(frame_id=fid,signature=signature,records=records)));temp.replace(path)
        all_rows.extend(records)
        if n==1 or n%10==0 or n==len(required): print(f'SAM3: {n}/{len(required)}',flush=True)
    cols=['frame_id','class_name','reference_id','confidence','mask_area']+BBOX+['mask_'+k for k in BBOX]
    detections = pd.DataFrame(all_rows,columns=cols)
    detections.to_csv(out/'sam3_raw.csv',index=False)
    for split,(ctx,neighbors,centers) in contexts.items():
        center = detections[detections.frame_id.isin(ctx.frame_id)&detections.confidence.ge(.25)].copy()
        for fid in sorted(centers-set(center.frame_id)):
            center=pd.concat([center,pd.DataFrame([dict(frame_id=fid,class_name='_empty',reference_id='sentinel',confidence=0.,x1=0,y1=0,x2=1,y2=1)])],ignore_index=True)
        selected = temporal(center,detections[detections.frame_id.isin(neighbors)],ctx,images)
        valid=np.isfinite(selected[['mask_'+k for k in BBOX]].to_numpy(float)).all(1)
        change=selected.class_name.isin(['cup','nude_hands'])&valid
        selected.loc[change,BBOX]=selected.loc[change,['mask_'+k for k in BBOX]].to_numpy()
        selected.to_csv(out/f'sam3_{split}.csv',index=False)


def temporal(center, neighbors, context, images):
    import cv2
    @lru_cache(maxsize=700)
    def gray(fid):
        im=cv2.imread(str(images[fid]),cv2.IMREAD_GRAYSCALE)
        if im is None:raise ValueError(f'Unreadable image {fid}')
        return im
    def align(source,target,box):
        src,dst=gray(source),gray(target);h,w=src.shape;b=np.asarray(box,float)
        x1,y1,x2,y2=np.round(b).astype(int);x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
        if x2-x1<8 or y2-y1<8:return b,-1.
        patch=src[y1:y2,x1:x2]
        if patch.std()<3:return b,-1.
        m=min(24,max(6,int(min(x2-x1,y2-y1)*.2)))
        sx,sy=max(0,x1-m),max(0,y1-m)
        search=dst[sy:min(h,y2+m),sx:min(w,x2+m)]
        _,best,_,loc=cv2.minMaxLoc(cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED))
        dx,dy=loc[0]+sx-x1,loc[1]+sy-y1
        return b+np.array([dx,dy,dx,dy]),float(best)
    output=[];eligible=neighbors[neighbors.confidence.ge(.25)]
    for fid,g in center.groupby('frame_id',sort=True):
        current=g.copy().reset_index(drop=True);r=context[context.frame_id.eq(fid)].iloc[0]
        if pd.notna(r.previous) and pd.notna(r['next']):
            for cls in SCORED_CLASSES:
                p=eligible[eligible.frame_id.eq(r.previous)&eligible.class_name.eq(cls)].reset_index(drop=True)
                n=eligible[eligible.frame_id.eq(r['next'])&eligible.class_name.eq(cls)].reset_index(drop=True)
                ii,jj=match_from_iou(box_iou(p[BBOX],n[BBOX]),.6)
                for i,j in zip(ii,jj):
                    pb,pc=align(str(r.previous),fid,p.loc[i,BBOX].to_numpy(float))
                    nb,nc=align(str(r['next']),fid,n.loc[j,BBOX].to_numpy(float))
                    if min(pc,nc)<.6 or box_iou([pb],[nb])[0,0]<.6:continue
                    b=(pb+nb)/2;same=current[current.class_name.eq(cls)]
                    if len(same) and box_iou([b],same[BBOX]).max()>=.3:continue
                    row=dict(frame_id=fid,class_name=cls,reference_id=f'temporal_{fid}_{cls}_{i}_{j}',
                             confidence=min(float(p.loc[i,'confidence']),float(n.loc[j,'confidence']),pc,nc),**dict(zip(BBOX,b)))
                    current=pd.concat([current,pd.DataFrame([row])],ignore_index=True)
        output.append(current)
    result=pd.concat(output,ignore_index=True)
    return result[result.class_name.ne('_empty')].copy()
