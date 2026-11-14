"""Instance-conditioned reference boxes from frozen CLIP/DINO features.
Unlike V31 occupancy union, each target is ONE GT instance selected by the query.
Machine presence-only labels do not supervise this geometry branch.
"""
from vision import *
import time,argparse,gc
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('key',choices=['clip','vit_b','vit_l']);p.add_argument('--steps',type=int,default=350);args=p.parse_args();key=args.key
rec=pd.read_csv(A/'inputs/records.csv');q,te,h=inputs();feat=np.load(A/f'features/{key}_penultimate.npy',mmap_mode='r');grid=int(np.sqrt(feat.shape[1]-1));dim=feat.shape[2]
device='mps';assert torch.backends.mps.is_available();mask,extra=aux(rec,grid)
model=load(key);block,norm=parts(model,key);block=block.to(device).eval();norm=norm.to(device).eval()
cache=A/f'features/{key}_final_patches.npy'
if not cache.exists():
 out=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float16,shape=(len(rec),dim,grid,grid))
 with torch.inference_mode():
  for off in range(0,len(rec),24):
   x=torch.from_numpy(np.array(feat[off:off+24],dtype=np.float32)).to(device)
   z=block(x,attention_mask=None,causal_attention_mask=None)[0] if key=='clip' else block(x)[0]
   z=norm(z)[:,1:].transpose(1,2).reshape(-1,dim,grid,grid);out[off:off+len(x)]=z.cpu().numpy().astype(np.float16)
 out.flush()
del block,norm,model;gc.collect();torch.mps.empty_cache();features=np.load(cache,mmap_mode='r')
targets=np.zeros((len(rec),1,grid,grid),np.float32);valid=np.zeros(len(rec),bool)
for ix,r in rec[rec.split.isin(['synthetic','query'])].iterrows():
 gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy();valid[ix]=True
 if not len(gt):continue
 overlap=iou(r[B],gt)[0];j=overlap.argmax()
 if overlap[j]<.05:continue
 cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];b=(gt[j]-np.tile(cb[:2],2))/np.tile(size,2);lo=np.arange(grid)/grid;hi=(np.arange(grid)+1)/grid
 wx=np.maximum(0,np.minimum(b[2],hi)-np.maximum(b[0],lo))*grid;wy=np.maximum(0,np.minimum(b[3],hi)-np.maximum(b[1],lo))*grid
 targets[ix,0]=wy[:,None]*wx[None,:]
class Geometry(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(dim+7,96,1),nn.GroupNorm(8,96),nn.GELU(),nn.Conv2d(96,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU(),nn.Conv2d(48,1,1))
 def forward(self,x,m,e):
  n=len(x);xy=torch.stack(torch.meshgrid(torch.linspace(-1,1,grid,device=x.device),torch.linspace(-1,1,grid,device=x.device),indexing='ij')).expand(n,-1,-1,-1)
  cls=e[:,-4:,None,None].expand(-1,-1,grid,grid);z=F.normalize(x,dim=1)*np.sqrt(dim)
  return self.net(torch.cat([z,m.reshape(n,1,grid,grid),xy,cls],1))
outputs=[];manifest=[]
for fit in DAYS:
 ix=np.flatnonzero(rec.day.eq(fit)&valid);ev=rec.split.eq('test')|(rec.split.eq('query')&rec.day.ne(fit));ex=np.flatnonzero(ev)
 assert not set(rec.loc[ix,'frame_id'])&set(rec.loc[ev,'frame_id'])
 torch.manual_seed(4101);rng=np.random.default_rng(4101);net=Geometry().to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.02)
 weights=1/rec.loc[ix].groupby('frame_id').id.transform('size').to_numpy();weights/=weights.sum();start=time.time()
 for step in range(args.steps):
  ii=rng.choice(ix,32,replace=True,p=weights);x=torch.tensor(np.array(features[ii],dtype=np.float32),device=device);y=torch.tensor(targets[ii],device=device)
  logits=net(x,mask[ii].to(device),extra[ii].to(device));prob=logits.sigmoid()
  bce=F.binary_cross_entropy_with_logits(logits,y);dice=1-(2*(prob*y).sum((1,2,3))+1)/(prob.sum((1,2,3))+y.sum((1,2,3))+1)
  loss=bce+.3*dice.mean();opt.zero_grad();loss.backward();opt.step()
  if step%100==0:print('GEOMETRY',key,fit,step,float(loss),round(time.time()-start,1),flush=True)
 net.eval()
 with torch.inference_mode():
  for off in range(0,len(ex),24):
   ii=ex[off:off+24];x=torch.tensor(np.array(features[ii],dtype=np.float32),device=device);prob=net(x,mask[ii].to(device),extra[ii].to(device)).sigmoid().cpu().numpy()[:,0]
   for ri,pm in zip(ii,prob):
    r=rec.iloc[ri];cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];nb,labels,stats,cent=cv2.connectedComponentsWithStats((pm>=.5).astype(np.uint8),8)
    boxes=[];confs=[]
    for j in range(1,nb):
     xx,yy,w,hh,area=stats[j]
     if area<2:continue
     boxes.append(np.array([xx,yy,xx+w,yy+hh])/grid*np.tile(size,2)+np.tile(cb[:2],2));confs.append(float(pm[labels==j].mean()))
    overlap=iou(r[B],boxes)[0] if boxes else np.zeros(0);best=int(overlap.argmax()) if len(overlap) else None
    ov=float(overlap[best]) if best is not None else 0.;box=boxes[best] if best is not None else [np.nan]*4
    outputs.append(dict(id=r.id,frame_id=r.frame_id,class_name=r.class_name,day=r.day,split=r.split,fit_day=fit,key=key,score=float(np.clip((.85-ov)/.5,0,1)),reference_iou=ov,reference_confidence=confs[best] if best is not None else 0.,**dict(zip(['ref_'+b for b in B],box))))
 torch.save({k:v.cpu() for k,v in net.state_dict().items()},A/f'models/{key}_geometry_fit{fit[-2:]}.pt')
 manifest.append(dict(key=key,fit_day=fit,steps=args.steps,fit_frames=sorted(rec.loc[ix,'frame_id'].unique()),pool_used=False,target='single GT with max query IoU, empty if max < .05',seconds=time.time()-start))
 del net,opt;gc.collect();torch.mps.empty_cache()
pd.DataFrame(outputs).to_csv(A/f'features/{key}_geometry_predictions.csv',index=False);(A/f'models/{key}_geometry_manifest.json').write_text(json.dumps(manifest,indent=2))
