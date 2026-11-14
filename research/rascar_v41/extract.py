"""Cache inputs to the last transformer block: earlier blocks remain frozen."""
from vision import *
import argparse,time,gc
p=argparse.ArgumentParser();p.add_argument('key',choices=['clip','vit_b','vit_l']);args=p.parse_args();key=args.key
rec=pd.read_csv(A/'inputs/records.csv');device='mps' if torch.backends.mps.is_available() else 'cpu'
assert device=='mps','Run with local GPU access to finish before deadline.'
model=load(key).to(device);start=time.time();imgcache={};block,norm=parts(model,key)
grid=7 if key=='clip' else 16;dim=model.config.hidden_size;mask,extra=aux(rec,grid)
shape=(len(rec),grid*grid+1,dim);out=np.lib.format.open_memmap(A/f'features/{key}_penultimate.npy',mode='w+',dtype=np.float16,shape=shape)
vec=np.empty((len(rec),dim*3+13),np.float32)
mean=[.48145466,.4578275,.40821073] if key=='clip' else [.485,.456,.406]
std=[.26862954,.26130258,.27577711] if key=='clip' else [.229,.224,.225]
mean=torch.tensor(mean).view(1,3,1,1).to(device);std=torch.tensor(std).view(1,3,1,1).to(device)
for off in range(0,len(rec),8):
 images=[]
 for _,r in rec.iloc[off:off+8].iterrows():
  if r.frame_id not in imgcache:
   im=cv2.imread(str(D/f'{r.frame_id}.jpg'));assert im is not None
   imgcache[r.frame_id]=cv2.resize(im,(1280,720))
  b=r[CB].to_numpy(int);im=imgcache[r.frame_id][b[1]:b[3],b[0]:b[2],::-1]
  images.append(cv2.resize(im,(224,224),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
 x=(torch.from_numpy(np.stack(images)).to(device).float()/255-mean)/std
 with torch.inference_mode():
  y=model(pixel_values=x,output_hidden_states=True);z=y.hidden_states[-2]
  last=norm(y.last_hidden_state) if key=='clip' else y.last_hidden_state
  if off==0:
   re=block(z,attention_mask=None,causal_attention_mask=None)[0] if key=='clip' else block(z)[0]
   err=float((norm(re)-last).abs().max());assert err<1e-4,err
  f=pooled(last,mask[off:off+len(x)].to(device),extra[off:off+len(x)].to(device))
 out[off:off+len(x)]=z.cpu().numpy().astype(np.float16);vec[off:off+len(x)]=f.cpu().numpy()
 if off%160==0:print('EXTRACT',key,off,len(rec),round(time.time()-start,1),flush=True)
out.flush();np.save(A/f'features/{key}_frozen.npy',vec)
(A/f'features/{key}_extraction.json').write_text(json.dumps(dict(key=key,shape=shape,device=device,seconds=time.time()-start,last_block_replay_max_error=err,record_sha256=hashlib.sha256((A/'inputs/records.csv').read_bytes()).hexdigest()),indent=2))
print('DONE',key,time.time()-start,flush=True)
