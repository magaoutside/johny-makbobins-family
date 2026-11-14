"""New RGB augmentation views, with spatial targets transformed synchronously.

Only manually annotated training frames are augmented. Evaluation feature grids
remain the original V31 caches, and neither evaluation frames nor labels train a
fold's occupancy head.
"""
from common55 import *
import torch,cv2,argparse
from functools import lru_cache
from transformers import AutoModel

@lru_cache(maxsize=50)
def image(frame):
    path=D/f'{frame}.jpg';im=cv2.imread(str(path));assert im is not None
    return cv2.resize(im,(1280,720))[:,:,::-1].copy()

def occupancy(r,h):
    crop=r[CB].to_numpy(float);wh=(crop[2:]-crop[:2])/32
    xx=crop[0]+np.arange(32)*wh[0];yy=crop[1]+np.arange(32)*wh[1]
    out=np.zeros((4,32,32),np.float32)
    for k,cls in enumerate(CL):
        for b in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(cls)][B].to_numpy(float):
            wx=np.maximum(0,np.minimum(xx+wh[0],b[2])-np.maximum(xx,b[0]))/wh[0]
            wy=np.maximum(0,np.minimum(yy+wh[1],b[3])-np.maximum(yy,b[1]))/wh[1]
            out[k]=np.maximum(out[k],wy[:,None]*wx[None,:])
    return out[:,:,::-1].copy() if r.flip else out

def prepare():
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');q,t,h=inputs();rng=np.random.default_rng(5501);rows=[]
    for parent,r in rec[rec.split.eq('manual')].iterrows():
        for family in ['flip','photo','crop_photo']:
            z=r.to_dict();z.update(id=f'{r.id}_{family}',parent_index=int(parent),family=family,flip=family=='flip',seed=int(rng.integers(0,2**31)))
            if family=='crop_photo':
                box=r[B].to_numpy(float);wh=np.maximum(box[2:]-box[:2],16)
                center=(box[:2]+box[2:])/2+rng.uniform(-.3,.3,2)*wh;size=wh*rng.uniform(1.7,3.5)
                crop=np.r_[np.maximum(0,np.floor(center-size/2)),np.minimum([1280,720],np.ceil(center+size/2))]
                z.update(dict(zip(CB,crop)));z['flip']=bool(rng.integers(2))
            rows.append(z)
    out=pd.DataFrame(rows);out.to_csv(A/'inputs/augmented_crops.csv',index=False)
    masks=np.stack([occupancy(r,h) for _,r in out.iterrows()]);np.save(A/'features/augmented_targets.npy',masks.astype(np.float16))
    assert set(out.frame_id).isdisjoint(t.frame_id)
    old=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
    flip=out.family.eq('flip');assert np.array_equal(masks[flip].astype(np.float16),np.asarray(old[out.loc[flip,'parent_index']])[:,:,:,::-1])
    protocol=dict(seed=5501,crops=len(out),original_manual_crops=int(rec.split.eq('manual').sum()),manual_frames=int(out.frame_id.nunique()),
         families=out.groupby('family').size().to_dict(),input_size=448,grid=32,
         supervision='Soft union of manual GT rectangles, not segmentation masks.',
         augmentation='Actual RGB horizontal reflection; gamma/contrast/brightness/saturation; mild blur; new crop centers and scales. Spatial GT targets recomputed and reflected with pixels.',
         source_record_sha256=hashlib.sha256((ROOT/'rascar_v31/inputs/dense_records.csv').read_bytes()).hexdigest(),
         target_flip_exact_replay=True,validation='Original opposite-day queries only; no augmentation in validation or test.',test_used_in_fit=False)
    (A/'inputs/protocol.json').write_text(json.dumps(protocol,indent=2));print('PREPARE',json.dumps(protocol),flush=True)
    return out

def transform(r):
    cb=r[CB].to_numpy(int);x=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2]].copy()
    if r.family!='flip':
        rng=np.random.default_rng(int(r.seed));x=x.astype(np.float32)/255
        gamma=rng.uniform(.65,1.5);contrast=rng.uniform(.7,1.3);brightness=rng.uniform(-.10,.10);saturation=rng.uniform(0,1.4)
        x=np.power(x,gamma);gray=x@np.array([.299,.587,.114],np.float32)
        x=gray[:,:,None]+saturation*(x-gray[:,:,None]);x=(x-.5)*contrast+.5+brightness
        x=np.clip(x,0,1)
        if rng.random()<.25:x=cv2.GaussianBlur(x,(3,3),rng.uniform(.2,.8))
        x=np.round(x*255).astype(np.uint8)
    if r.flip:x=x[:,::-1].copy()
    return cv2.resize(x,(448,448),interpolation=cv2.INTER_CUBIC)

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    rec=prepare();model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    shape=(len(rec),384,32,32);out=np.lib.format.open_memmap(A/'features/augmented_grids.npy',mode='w+',dtype=np.float16,shape=shape)
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);start=time.time()
    # Verify that the current images/preprocessing still reproduce the original cache.
    oldrec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');old=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r')
    check=[]
    for i in [0,200,600]:
        r=oldrec.iloc[i];cb=r[CB].to_numpy(int);x=cv2.resize(image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2]],(448,448),interpolation=cv2.INTER_CUBIC)
        x=(torch.from_numpy(x.transpose(2,0,1).copy())[None].float()/255-mean)/std
        with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(384,32,32).numpy()
        delta=float(abs(z-np.asarray(old[i],np.float32)).max());check.append(dict(index=i,max_abs_delta=delta))
        assert delta<.03,check[-1]
    (A/'audit/original_feature_replay.json').write_text(json.dumps(check,indent=2))
    for off in range(0,len(rec),4):
        batch=np.stack([transform(r).transpose(2,0,1).copy() for _,r in rec.iloc[off:off+4].iterrows()])
        x=(torch.from_numpy(batch).float()/255-mean)/std
        with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
        out[off:off+len(batch)]=z.numpy().astype(np.float16)
        if off%100==0:print('EXTRACT',off,len(rec),round(time.time()-start,1),flush=True)
    out.flush();(A/'features/extraction_complete.json').write_text(json.dumps(dict(shape=shape,seconds=time.time()-start,record_sha256=hashlib.sha256((A/'inputs/augmented_crops.csv').read_bytes()).hexdigest()),indent=2))
    print('DONE',shape,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
