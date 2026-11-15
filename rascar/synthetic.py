"""Manual-box augmentation and frozen DINOv3 features for upstream quality heads."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from .core import queries, labels, BBOX as B, CLASSES as CL
from .geometry import iou


def build_records(data, raw):
    rows=queries(data)
    q=rows[rows.split.eq('oof')].copy()
    q['error']=labels(data,rows).loc[q.index].astype(int)
    te=rows[rows.split.eq('test')].copy()
    h=pd.read_csv(Path(data)/'manual_boxes.csv')
    found=list(Path(raw).rglob('train_frames.csv'))
    if len(found)!=1:raise ValueError('Expected one original train_frames.csv')
    fr=pd.read_csv(found[0]);fr['day']=pd.to_datetime(fr.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
    records=[]
    for kind,df in [('query_train',q),('query_test',te)]:
        for r in df.to_dict('records'):
            records.append({**r,'kind':kind,'semantic_class':'unknown','weight':2.0,'origin_id':r['id']})
    gt=h[h.class_name.isin(CL)].copy()
    def add_box(b,r,kind,weight=.3,semantic='unknown',serial=0):
        b=np.asarray(b,float);b[[0,2]]=np.clip(b[[0,2]],0,1280);b[[1,3]]=np.clip(b[[1,3]],0,720)
        if min(b[2:]-b[:2])<4:return
        pool=gt[(gt.frame_id==r.frame_id)&(gt.class_name==r.class_name)]
        best=float(iou(b,pool[B]).max()) if len(pool) else 0.
        records.append(dict(id=f'{kind}_{r.Index}_{serial}',origin_id=f'gt_{r.Index}',frame_id=r.frame_id,day=r.day,
            class_name=r.class_name,semantic_class=semantic,kind=kind,weight=weight,error=int(best<.6),
            target_iou=best,**dict(zip(B,b))))

    for r in gt.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2
        add_box(b,r,'gt',1.,r.class_name)
        for j,scale in enumerate([.65,.85,1.15,1.4,1.8]):
            add_box(np.r_[center-wh*scale/2,center+wh*scale/2],r,'scaled',.25,serial=j)
        for j,(dx,dy) in enumerate([(-.4,0),(.4,0),(0,-.4),(0,.4),(-.2,.2),(.2,-.2)]):
            add_box(b+np.tile(wh*[dx,dy],2),r,'shifted',.25,serial=j)
        if r.class_name=='person':
            for j,frac in enumerate([.3,.5,.65]):
                bb=b.copy();bb[3]-=wh[1]*frac;add_box(bb,r,'short_person',.4,r.class_name,j)
                bb=b.copy();bb[1]+=wh[1]*frac;add_box(bb,r,'headless_person',.4,r.class_name,j)
        if r.class_name=='nude_hands':
            # One-sided extension and translation create forearm/extra-background examples.
            for j,k in enumerate(range(4)):
                bb=b.copy();bb[k]+=(-1 if k<2 else 1)*wh[k%2]*1.2
                add_box(bb,r,'hand_extension',.4,serial=j)
            for j,(dx,dy) in enumerate([(-.8,0),(.8,0),(0,-.8),(0,.8)]):
                add_box(b+np.tile(wh*[dx,dy],2),r,'hand_displaced',.4,serial=j)
        peers=gt[(gt.frame_id==r.frame_id)&(gt.class_name==r.class_name)&(gt.index>r.Index)]
        for j,p in enumerate(peers.itertuples()):
            bb=np.array([getattr(p,c) for c in B]);distance=np.linalg.norm((bb[:2]+bb[2:])/2-center)
            if distance<max(wh.max(),(bb[2:]-bb[:2]).max())*1.4:
                add_box(np.r_[np.minimum(b[:2],bb[:2]),np.maximum(b[2:],bb[2:])],r,'merged',.4,serial=j)

    rng=np.random.default_rng(1701)
    for f in fr[fr.frame_id.isin(h.frame_id)].itertuples():
        fg=gt[gt.frame_id.eq(f.frame_id)][B].to_numpy();candidates=q[q.day.eq(f.day)]
        accepted=0
        for ix in rng.permutation(len(candidates)):
            rr=candidates.iloc[ix];b=rr[B].to_numpy(float);area=np.prod(b[2:]-b[:2])
            inter=np.maximum(0,np.minimum(fg[:,2:],b[2:])-np.maximum(fg[:,:2],b[:2])).prod(1) if len(fg) else np.array([0])
            if inter.max()/area>.015:continue
            records.append(dict(id=f'background_{f.frame_id}_{accepted}',origin_id=f.frame_id,frame_id=f.frame_id,day=f.day,
                class_name=rr.class_name,semantic_class='background',kind='background',weight=.4,error=1,target_iou=0.,**dict(zip(B,b))))
            accepted+=1
            if accepted>=8:break

    records=pd.DataFrame(records);assert records.id.is_unique
    return records

def extract(data, raw, weights, out, device):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor,AutoModel
    from .detectors import paths as image_paths
    torch.set_num_threads(2)
    A=Path(out);A.mkdir(parents=True,exist_ok=True)
    records=build_records(data,raw);records.to_csv(A/'dino_records.csv',index=False)
    paths=image_paths(raw)
    model_name='facebook/dinov3-vitb16-pretrain-lvd1689m'
    revision='5931719e67bbdb9737e363e781fb0c67687896bc'
    checkpoint=str(Path(weights)/'dinov3')
    processor=AutoImageProcessor.from_pretrained(checkpoint,local_files_only=True)
    model=AutoModel.from_pretrained(checkpoint,local_files_only=True,dtype=torch.float16).to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    skip=1+model.config.num_register_tokens;vectors=[]
    images_cache={fid:Image.open(path).convert('RGB') for fid,path in paths.items()}
    for start in range(0,len(records),24):
        batch=records.iloc[start:start+24];images=[]
        for r in batch.itertuples():
            im=images_cache[r.frame_id];b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
            for context in [0,.35]:
                bb=b+np.array([-1,-1,1,1])*np.tile(wh,2)*context;bb[[0,2]]=np.clip(bb[[0,2]],0,1280);bb[[1,3]]=np.clip(bb[[1,3]],0,720)
                images.append(im.crop(tuple(bb)).resize((224,224),Image.Resampling.BICUBIC))
        inp=processor(images=images,do_resize=False,do_center_crop=False,return_tensors='pt').to(device,torch.float16)
        with torch.inference_mode():
            z=model(**inp).last_hidden_state.float();cls=torch.nn.functional.normalize(z[:,0],dim=-1)
            patches=z[:,skip:];mean=torch.nn.functional.normalize(patches.mean(1),dim=-1)
            # Exact and contextual CLS/mean patch remain separable in the saved array.
            v=torch.cat([cls,mean],dim=-1).cpu().numpy().reshape(len(batch),2,-1)/np.sqrt(2)
        vectors.append(v.astype(np.float16))
        if start%240==0:print('DINO16_FEATURES',start,len(records),flush=True)
    X=np.concatenate(vectors);np.save(A/'dino_features.npy',X)
    (A/'dino_manifest.json').write_text(json.dumps(dict(model=model_name,revision=revision,frozen=True,
        shape=list(X.shape),contexts=[0,.35],synthetic_labels='best same-class GT IoU >= .6; individual quality, duplicates handled separately',
        training_frames=84,test_labels_used=False,counts=records.kind.value_counts().to_dict()),indent=2))
    print('DINO16_DONE',X.shape,flush=True)
