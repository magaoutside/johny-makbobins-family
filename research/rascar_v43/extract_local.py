"""Query-defined local context, no GT in preprocessing. Independent detector FPN."""
from common43 import *
import torch,cv2,time
import torch.nn.functional as F
def main():
    torch.set_num_threads(4);cv2.setNumThreads(2);assert torch.backends.mps.is_available();device='mps'
    m=detector(device);r=pd.read_csv(A/'inputs/records.csv');n=len(r);crops=[]
    for _,v in r.iterrows():
        b=v[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;span=np.maximum(wh*2.5,96)
        lo=np.maximum(0,np.floor(cc-span/2));hi=np.minimum([1280,720],np.ceil(cc+span/2));crops.append(np.r_[lo,hi])
    cb=np.array(crops);np.save(A/'features/local_context.npy',cb)
    cache=np.lib.format.open_memmap(A/'features/local_grids.npy',mode='w+',dtype=np.float16,shape=(n,256,28,28))
    # One image load per frame, batch local windows. Resize fixed 448, P2 stride4,
    # then reduce to 28x28. Smaller objects retain more cells than full-frame ROI.
    mean=torch.tensor(m.transform.image_mean,device=device)[None,:,None,None];std=torch.tensor(m.transform.image_std,device=device)[None,:,None,None]
    start=time.time();processed=0
    with torch.inference_mode():
      for fid,g in r.groupby('frame_id',sort=True):
        im=cv2.resize(cv2.imread(str(D/f'{fid}.jpg')),(1280,720))[:,:,::-1].copy()
        for off in range(0,len(g),8):
            ix=g.index.to_numpy()[off:off+8];arrays=[]
            for i in ix:
                x1,y1,x2,y2=cb[i].astype(int);arrays.append(cv2.resize(im[y1:y2,x1:x2],(448,448),interpolation=cv2.INTER_LINEAR).transpose(2,0,1))
            z=torch.tensor(np.stack(arrays),dtype=torch.float32,device=device)/255
            features=m.backbone((z-mean)/std)['0'];features=F.adaptive_avg_pool2d(features,(28,28));cache[ix]=features.cpu().numpy().astype(np.float16)
            processed+=len(ix)
        if processed//300!=(processed-len(g))//300:print('LOCAL',processed,n,round(time.time()-start,1),flush=True)
    cache.flush();(A/'features/local_manifest.json').write_text(json.dumps(dict(rows=n,record_sha256=hashlib.sha256((A/'inputs/records.csv').read_bytes()).hexdigest(),seconds=time.time()-start,grid=[256,28,28],preprocessing='RGB context2.5x min96, resize448x448, pretrained mean/std, P2 avgpool28',no_gt_crop_coordinates=True),indent=2));print('DONE',time.time()-start,flush=True)
if __name__=='__main__':main()
