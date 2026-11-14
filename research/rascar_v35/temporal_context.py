"""Independent context inference at +/-30 seconds, registered to each query.

Frozen V31 heads. No training, test annotations, per-ID rules, or LB fitting.
OOF uses only the head fitted on the other day. Each test correction must
agree across both heads. Empty/failed context is abstention, not an error.
"""
from pathlib import Path
from functools import lru_cache
import os, sys, json, time, hashlib
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from transformers import AutoModel
from sklearn.metrics import average_precision_score as AP

ROOT=Path(__file__).resolve().parent.parent
A=ROOT/'rascar_v35'; V31=ROOT/'rascar_v31'
for d in ['inputs','features','audit','logs','results/oof','results/submissions']:
    (A/d).mkdir(parents=True,exist_ok=True)
B=['x1','y1','x2','y2']; CB=['crop_'+b for b in B]
CL=['person','nude_hands','cup','tray']; DAYS=['2026-10-25','2026-10-26']
q=pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv')
te=pd.read_csv(ROOT/'rascar_v22/inputs/test_queries.csv')
ROWS={'oof':q,'test':te}
PATHS={'oof':V31/'results/oof/linear_mask_cut0.5_all_two_sided_w0.25.csv',
       'test':V31/'submission_v31_dense_oof098784.csv'}
BASE={s:pd.read_csv(PATHS[s]).set_index('id').loc[d.id,'score'].to_numpy() for s,d in ROWS.items()}
torch.set_num_threads(3); cv2.setNumThreads(1)
os.environ['TOKENIZERS_PARALLELISM']='false'

def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4); b=np.asarray(b,float).reshape(-1,4)
    inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None]-inter,1e-9)

def metrics(y,s):
    out={'pooled_ap':float(AP(y.error,s))}
    for col,prefix in [('day','day_'),('class_name','ap_')]:
        for val in y[col].unique():
            m=y[col].eq(val).to_numpy()
            out[prefix+(val[-2:] if col=='day' else val)]=float(AP(y.loc[m,'error'],s[m]))
    return out

@lru_cache(maxsize=48)
def image(fid):
    p=ROOT/'rascar_data_full'/f'{fid}.jpg'
    im=cv2.imread(str(p)); assert im is not None,str(p)
    return cv2.resize(im,(1280,720),interpolation=cv2.INTER_LINEAR)

def prepare():
    rec=pd.read_csv(V31/'inputs/dense_records.csv')
    rec=rec[rec.split.isin(['oof','test'])].copy()
    contexts=pd.concat([pd.read_csv(ROOT/f'rascar_data_full/{s}_context.csv') for s in ['train','test']]).set_index('frame_id')
    assert contexts.index.is_unique
    out=[]
    for _,r in rec.iterrows():
        for side in ['previous','next']:
            fid=contexts.loc[r.frame_id,side]
            if pd.isna(fid):continue
            assert fid[-15:-7]==r.frame_id[-15:-7]
            sec=abs((pd.to_datetime(fid[-15:],format='%Y%m%d_%H%M%S')-pd.to_datetime(r.frame_id[-15:],format='%Y%m%d_%H%M%S')).total_seconds())
            assert sec==30
            x=r.to_dict(); x.update(query_frame=r.frame_id,frame_id=fid,side=side,query_feature_index=int(r.feature_index),feature_index=len(out))
            out.append(x)
    rec=pd.DataFrame(out)
    path=A/'inputs/context_records.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path),rec,check_dtype=False,atol=1e-10)
    rec.to_csv(path,index=False)
    return rec

def extract(rec):
    path=A/'features/context_grids.npy'
    if (A/'features/extraction_complete.json').exists():
        x=np.load(path,mmap_mode='r');assert len(x)==len(rec);return x
    model=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    dst=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(rec),384,32,32))
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    start=time.time()
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            cb=r[CB].to_numpy(int);im=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1]
            batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
        with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
        dst[off:off+len(batch)]=z.numpy().astype(np.float16)
        if off%120==0:print('CONTEXT_EXTRACT',off,len(rec),round(time.time()-start,1),flush=True)
    dst.flush()
    (A/'features/extraction_complete.json').write_text(json.dumps({'count':len(rec),'seconds':time.time()-start,'backbone':'facebook/dinov2-small','context_seconds':30,'test_annotations_used':False},indent=2))
    return np.load(path,mmap_mode='r')

def track(r):
    a=cv2.cvtColor(image(r.query_frame),cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(image(r.frame_id),cv2.COLOR_BGR2GRAY)
    box=r[B].to_numpy(float); x1,y1,x2,y2=np.rint(box).astype(int)
    mask=np.zeros_like(a);mask[max(y1,0):min(y2,720),max(x1,0):min(x2,1280)]=255
    p=cv2.goodFeaturesToTrack(a,180,.015,3,mask=mask,blockSize=5)
    out={'id':r.id,'side':r.side,'registered':False,'points':0,'retained':0.,'inlier_fraction':0.,'fb90':np.nan,'scale':np.nan,'rotation':np.nan,'shift':np.nan}
    for k in range(6):out[f'm{k}']=np.nan
    if p is None or len(p)<6:return out
    out['points']=len(p)
    opt=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    nxt,st,_=cv2.calcOpticalFlowPyrLK(a,b,p,None,**opt)
    back,stb,_=cv2.calcOpticalFlowPyrLK(b,a,nxt,None,**opt)
    fb=np.linalg.norm(back[:,0]-p[:,0],axis=1)
    good=st[:,0].astype(bool)&stb[:,0].astype(bool)&np.isfinite(fb)&(fb<=2.)
    out['retained']=float(good.mean())
    if good.sum()<6:return out
    mat,inliers=cv2.estimateAffinePartial2D(p[good,0],nxt[good,0],method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=2000,confidence=.99)
    if mat is None:return out
    inliers=inliers.ravel().astype(bool)
    out['inlier_fraction']=float(inliers.mean());out['fb90']=float(np.quantile(fb[good][inliers],.9))
    scale=np.hypot(mat[0,0],mat[1,0]);rot=np.degrees(np.arctan2(mat[1,0],mat[0,0]))
    center=(box[:2]+box[2:])/2;shift=np.linalg.norm(mat[:,:2]@center+mat[:,2]-center)/max(np.linalg.norm(box[2:]-box[:2]),1)
    out.update(scale=float(scale),rotation=float(rot),shift=float(shift))
    out['registered']=bool(out['retained']>=.5 and out['inlier_fraction']>=.7 and .85<=scale<=1.18 and abs(rot)<=8 and shift<=.4)
    out.update({f'm{k}':float(v) for k,v in enumerate(mat.ravel())})
    return out

def pooled(grid,r,box):
    cb=r[CB].to_numpy(float);lo=(box[:2]-cb[:2])/(cb[2:]-cb[:2]);hi=(box[2:]-cb[:2])/(cb[2:]-cb[:2])
    yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32
    mask=(xx>=lo[0])&(xx<hi[0])&(yy>=lo[1])&(yy<hi[1])
    if not mask.any():return np.zeros(384)
    x=np.asarray(grid[:,mask],np.float32).mean(1);return x/max(np.linalg.norm(x),1e-9)

def warp_box(box,matrix):
    x1,y1,x2,y2=box
    corners=np.array([[x1,y1,1],[x2,y1,1],[x1,y2,1],[x2,y2,1]])@matrix.T
    return np.r_[corners.min(0),corners.max(0)]

def components(prob,r,cls):
    n,lab,stats,_=cv2.connectedComponentsWithStats((prob>=.5).astype(np.uint8),8)
    cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2];out=[]
    for j in range(1,n):
        x,y,w,h,area=stats[j]
        if area<4:continue
        box=np.array([x,y,x+w,y+h])/32*np.r_[sz,sz]+np.r_[cb[:2],cb[:2]]
        out.append((box,float(prob[lab==j].mean())))
    return out

def infer(rec,feat):
    track_path=A/'features/registration.csv'
    if not track_path.exists():
        tracking=[];start=time.time()
        for i,r in rec.iterrows():
            tracking.append(track(r))
            if i%150==0:print('TRACK',i,len(rec),round(time.time()-start,1),flush=True)
        pd.DataFrame(tracking).to_csv(track_path,index=False)
    tracking=pd.read_csv(track_path);assert len(tracking)==len(rec)
    current=np.load(V31/'features/dense_grids.npy',mmap_mode='r');evidence=[]
    for fit in DAYS:
        # Same normalization and exact saved V31 linear head; no model refitting.
        head=nn.Conv2d(384,4,1)
        state=torch.load(V31/f'models/linear_mask_fit{fit[-2:]}.pt',weights_only=True)
        head.load_state_dict({k.replace('net.',''):v for k,v in state.items()});head.eval()
        selected=rec[rec.split.eq('test')|rec.day.ne(fit)].index.to_numpy()
        assert not rec.loc[selected][rec.loc[selected].split.eq('oof')].day.eq(fit).any()
        for off in range(0,len(selected),16):
            ids=selected[off:off+16]
            x=torch.from_numpy(np.array(feat[ids],np.float32))
            with torch.inference_mode():probs=head(torch.nn.functional.normalize(x,dim=1)*np.sqrt(384)).sigmoid().numpy()
            for j,idx in enumerate(ids):
                r=rec.loc[idx];tr=tracking.loc[idx]
                entry=dict(id=r.id,split=r.split,fit_day=fit,side=r.side,query_frame=r.query_frame,context_frame=r.frame_id,registered=bool(tr.registered),valid=False,appearance=np.nan,overlap=np.nan,confidence=np.nan,**{f'ref_{b}':np.nan for b in B})
                if tr.registered:
                    inv=cv2.invertAffineTransform(tr[[f'm{k}' for k in range(6)]].to_numpy(float).reshape(2,3))
                    refs=components(probs[j,CL.index(r.class_name)],r,r.class_name)
                    if refs:
                        boxes=np.array([warp_box(bb,inv) for bb,c in refs]);ov=iou(r[B],boxes).ravel();k=int(ov.argmax())
                        cb,conf=refs[k];bb=boxes[k]
                        similarity=float(pooled(current[int(r.query_feature_index)],r,r[B].to_numpy(float))@pooled(feat[idx],r,cb))
                        # Crop edge truncation is not evidence of a complete object.
                        crop=r[CB].to_numpy(float);gap=np.r_[cb[:2]-crop[:2],crop[2:]-cb[2:]]
                        edge=bool((gap<=1).any())
                        entry.update(valid=bool(ov[k]>=.15 and similarity>=.8 and not edge),appearance=similarity,overlap=float(ov[k]),confidence=conf,at_crop_edge=edge,**dict(zip(['ref_'+b for b in B],bb)))
                evidence.append(entry)
        print('CONTEXT_INFER',fit,len(selected),flush=True)
    evidence=pd.DataFrame(evidence);evidence.to_csv(A/'features/context_evidence.csv',index=False)
    return evidence

def main():
    rec=prepare();feat=extract(rec);infer(rec,feat)

if __name__=='__main__':main()
