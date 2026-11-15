"""DINOv3 crop embeddings and frozen SAM/WeVis/retrieval baseline."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from .core import queries, labels, BBOX
from .foundation_scoring import box_iou, score_refs_v4, retrieval_scores7, blend_scores7, fuse_person8


def records(data):
    q=queries(data).copy()
    q['error']=labels(data,q)
    h=pd.read_csv(Path(data)/'manual_boxes.csv')
    h=h[h.class_name.isin(['person','cup','nude_hands','tray'])].copy()
    q['kind']=np.where(q.split.eq('oof'),'input','test');q['record_id']=q.id
    h['kind']='gt';h['error']=0;h['id']=['gt_'+str(i) for i in h.index];h['record_id']=h.id
    return pd.concat([q[q.kind.eq('input')],h,q[q.kind.eq('test')]],ignore_index=True)


def extract(data,raw,weights,out,device='cuda:0'):
    import torch
    from PIL import Image
    from transformers import AutoModel,AutoImageProcessor
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    r=records(data);files={p.stem:p for p in Path(raw).rglob('*.jpg')}
    location=Path(weights)/'dinov3'
    processor=AutoImageProcessor.from_pretrained(str(location),local_files_only=True)
    model=AutoModel.from_pretrained(str(location),local_files_only=True,dtype=torch.float16).to(device).eval()
    model.requires_grad_(False);torch.set_num_threads(2);skip=1+model.config.num_register_tokens
    parts=[]
    for start in range(0,len(r),32):
        images=[]
        for row in r.iloc[start:start+32].itertuples():
            im=Image.open(files[row.frame_id]).convert('RGB')
            w,h=row.x2-row.x1,row.y2-row.y1
            b=[max(0,int(np.floor(row.x1-.1*w))),max(0,int(np.floor(row.y1-.1*h))),
               min(im.width,int(np.ceil(row.x2+.1*w))),min(im.height,int(np.ceil(row.y2+.1*h)))]
            images.append(im.crop(b).resize((224,224),Image.Resampling.BICUBIC))
        inputs=processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(device,torch.float16)
        with torch.inference_mode():z=model(**inputs).last_hidden_state.float()
        cls=torch.nn.functional.normalize(z[:,0],dim=-1)
        patch=torch.nn.functional.normalize(z[:,skip:].mean(1),dim=-1)
        parts.append((torch.cat([cls,patch],-1)/np.sqrt(2)).cpu().numpy().astype('float32'))
        print(f'Retrieval crops: {min(start+32,len(r))}/{len(r)}',flush=True)
    features=np.concatenate(parts)
    if not np.isfinite(features).all():raise ValueError('Invalid retrieval embeddings')
    np.save(out/'retrieval_features.npy',features);r.to_csv(out/'retrieval_records.csv',index=False)
    (out/'retrieval_complete.json').write_text(json.dumps(dict(ids=r.record_id.tolist(),shape=list(features.shape))))


def assemble(data,cache,detections,out):
    cache,detections,out=map(Path,(cache,detections,out));out.mkdir(parents=True,exist_ok=True)
    r=pd.read_csv(cache/'retrieval_records.csv');f=np.load(cache/'retrieval_features.npy')
    marker=json.loads((cache/'retrieval_complete.json').read_text())
    if marker['ids']!=r.record_id.tolist() or marker['shape']!=list(f.shape) or not np.isfinite(f).all():
        raise ValueError('Incomplete or misaligned retrieval features')
    wevis=pd.read_csv(detections/'wevis_full.csv');zoom=pd.read_csv(detections/'zoom_boxes.csv')
    q=queries(data);q['error']=labels(data,q)
    fused=[];scored=[]
    for split,name in [('oof','train'),('test','test')]:
        rows=q[q.split.eq(split)].drop(columns='split').reset_index(drop=True)
        sam=pd.read_csv(detections/f'sam3_{name}.csv')
        position=r.reset_index().set_index('id').loc[rows.id,'index'].to_numpy()
        knn=retrieval_scores7(rows,f[position],r,f,cross_day=split=='oof')
        base=blend_scores7(score_refs_v4(sam,rows),score_refs_v4(wevis,rows[rows.class_name.eq('person')]),knn)
        refs,_=fuse_person8(sam,wevis,.25,.3)
        new=score_refs_v4(refs,rows).set_index('id').loc[base.id]
        person=base.class_name.eq('person')
        base.loc[person,'score']=(base.loc[person,'score']+.65*(new.loc[base.loc[person,'id'],'score'].to_numpy()-base.loc[person,'sam_score'].to_numpy())).clip(0,1)
        for i,row in rows[~person].iterrows():
            boxes=zoom[zoom.id.eq(row.id)]
            overlap=float(box_iou(row[BBOX].to_numpy(float),boxes[BBOX]).max()) if len(boxes) else 0.
            base.loc[i,'score']=.85*base.loc[i,'score']+.15*expit((.6-overlap)/.08)
        base.loc[base.reason.eq('duplicate_one_to_one'),'score']=1.
        base['score']=base.score.clip(0,1)
        base.to_csv(out/f'foundation_{split}.csv',index=False)
        refs.to_csv(out/f'fused_{name}_references.csv',index=False)
        fused.append(refs);scored.append(base)
    pd.concat(fused,ignore_index=True).to_csv(out/'fused_references.csv',index=False)
    pd.concat(scored,ignore_index=True)[['id','sam_score','wevis_score']].to_csv(out/'reference_scores.csv',index=False)
