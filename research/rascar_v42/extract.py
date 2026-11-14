"""Frozen first three ConvNeXt stages; cache last-stage inputs for actual tuning."""
from network import *
import time
assert torch.backends.mps.is_available();device='mps';rec=pd.read_csv(A/'inputs/records.csv');meta=np.load(A/'inputs/metadata.npy');imgs=np.load(A/'features/crops_uint8.npy',mmap_mode='r')
model=backbone().eval().to(device);early=model.features[:6];last=model.features[6:];start=time.time()
out=np.lib.format.open_memmap(A/'features/early_maps.npy',mode='w+',dtype=np.float16,shape=(len(rec),2,384,14,14));frozen=np.empty((len(rec),768*7+18),np.float32)
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
with torch.inference_mode():
 for off in range(0,len(rec),12):
  batch=np.asarray(imgs[off:off+12]);n=len(batch);x=torch.tensor(batch.reshape(n*2,3,224,224),device=device).float()/255;x=(x-mean)/std
  z=early(x);out[off:off+n]=z.reshape(n,2,384,14,14).cpu().numpy().astype(np.float16)
  end=last(z).reshape(n,2,768,7,7);v=vectors(end,torch.tensor(meta[off:off+n],device=device),'dual',model.classifier[0]);frozen[off:off+n]=v.cpu().numpy()
  if off%240==0:print('EXTRACT',off,len(rec),round(time.time()-start,1),flush=True)
out.flush();np.save(A/'features/frozen_vectors.npy',frozen)
(A/'features/extraction.json').write_text(json.dumps(dict(seconds=time.time()-start,record_sha256=hashlib.sha256((A/'inputs/records.csv').read_bytes()).hexdigest(),shape=list(out.shape),backbone='ConvNeXt-Tiny ImageNet1K V1',trainable_later='downsample + final 3 ConvNeXt blocks + LayerNorm + direct binary head'),indent=2));print('EXTRACT DONE',time.time()-start,flush=True)

