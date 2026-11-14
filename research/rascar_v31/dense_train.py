"""Two cross-day dense heads on frozen DINO grids.

linear_mask predicts box occupancy (NOT true pixel segmentation).
conv_detector is a compact, single-level FCOS-inspired detector, not a full
reproduction of FCOS. Only manual boxes from the fitting day supervise it.
"""
from pathlib import Path
import sys,json,time,os
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v31'
os.environ.setdefault('MPLCONFIGDIR',str(A/'model_cache/mpl'))
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import h,B,CL,DAYS,iou
import numpy as np,pandas as pd,torch,cv2
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3);torch.set_num_interop_threads(1);cv2.setNumThreads(1)

class Head(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind
  if kind=='linear_mask':self.net=nn.Conv2d(384,4,1)
  else:
   self.net=nn.Sequential(nn.Conv2d(384,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU())
   self.cls=nn.Conv2d(64,4,1);self.reg=nn.Conv2d(64,4,1);self.ctr=nn.Conv2d(64,1,1)
   nn.init.constant_(self.cls.bias,-3);nn.init.constant_(self.reg.bias,-1.5)
 def forward(self,x):
  x=F.normalize(x,dim=1)*np.sqrt(384);z=self.net(x)
  if self.kind=='linear_mask':return z
  return self.cls(z),F.softplus(self.reg(z)),self.ctr(z)

def det_target(r):
 cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);size=cb[2:]-cb[:2]
 yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32
 c=np.zeros((4,32,32),np.float32);b=np.zeros((4,32,32),np.float32);cent=np.zeros((1,32,32),np.float32)
 area=np.ones((32,32))*1e8;pos=np.zeros((32,32),bool)
 for _,g in h[h.frame_id.eq(r.frame_id)&h.class_name.isin(CL)].iterrows():
  box=(g[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])/np.r_[size,size]
  # Partially cropped neighbors only contribute when their centers remain inside.
  center=(box[:2]+box[2:])/2
  if not ((center>0)&(center<1)).all():continue
  box=np.clip(box,0,1);wh=box[2:]-box[:2];ar=wh.prod()
  l=xx-box[0];t=yy-box[1];rr=box[2]-xx;bb=box[3]-yy
  inside=(np.minimum.reduce([l,t,rr,bb])>0)&(abs(xx-center[0])<1.5/32)&(abs(yy-center[1])<1.5/32)&(area>ar)
  c[:,inside]=0;c[CL.index(g.class_name),inside]=1
  b[:,inside]=np.stack([l,t,rr,bb])[:,inside]
  cent[0,inside]=np.sqrt(np.minimum(l,rr)[inside]/np.maximum(l,rr)[inside]*np.minimum(t,bb)[inside]/np.maximum(t,bb)[inside])
  area[inside]=ar;pos[inside]=True
 return c,b,cent,pos

def det_loss(pred,target):
 logits,dist,ctr=pred;y,t,cent,pos=target
 p=logits.sigmoid();ce=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
 pt=p*y+(1-p)*(1-y);alpha=.25*y+.75*(1-y)
 cls=(alpha*(1-pt).square()*ce).sum()/pos.sum().clamp_min(1)
 if not pos.any():return cls
 a=dist.permute(0,2,3,1)[pos];b=t.permute(0,2,3,1)[pos]
 inter=(torch.minimum(a[:,0],b[:,0])+torch.minimum(a[:,2],b[:,2]))*(torch.minimum(a[:,1],b[:,1])+torch.minimum(a[:,3],b[:,3]))
 aa=(a[:,0]+a[:,2])*(a[:,1]+a[:,3]);bb=(b[:,0]+b[:,2])*(b[:,1]+b[:,3]);ov=inter/(aa+bb-inter).clamp_min(1e-6)
 reg=(1-ov).mean()+3*F.smooth_l1_loss(a,b,beta=.05)
 ctr_loss=F.binary_cross_entropy_with_logits(ctr[:,0][pos],cent[:,0][pos])
 return cls+2*reg+ctr_loss

def get_boxes(pred,r,kind):
 cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);size=cb[2:]-cb[:2];out=[]
 if kind=='linear_mask':
  prob=pred.sigmoid().numpy()
  for k,c in enumerate(CL):
   for cut in [.35,.5,.65]:
    n,labels,stats,cent=cv2.connectedComponentsWithStats((prob[k]>=cut).astype(np.uint8),8)
    for j in range(1,n):
     x,y,w,hh,ar=stats[j]
     if ar<4:continue
     box=np.r_[x,y,x+w,y+hh]/32*np.r_[size,size]+np.r_[cb[:2],cb[:2]]
     conf=float(prob[k][labels==j].mean());out.append(dict(class_name=c,threshold=cut,confidence=conf,**dict(zip(B,box))))
 else:
  logits,dist,ctr=pred;prob=torch.sqrt(logits.sigmoid()*ctr.sigmoid()).numpy();dist=dist.numpy()
  yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32
  for k,c in enumerate(CL):
   ids=np.argsort(prob[k].ravel())[::-1][:80];keep=[]
   for ix in ids:
    y,x=divmod(int(ix),32);conf=float(prob[k,y,x])
    if conf<.08:break
    l,t,rr,bb=dist[:,y,x];box=np.clip([xx[y,x]-l,yy[y,x]-t,xx[y,x]+rr,yy[y,x]+bb],0,1)*np.r_[size,size]+np.r_[cb[:2],cb[:2]]
    if keep and iou(box,np.array(keep)).max()>.5:continue
    keep.append(box);out.append(dict(class_name=c,threshold=0.,confidence=conf,**dict(zip(B,box))))
    if len(keep)>=8:break
 return out

def main():
 assert (A/'features/extraction_complete.json').exists()
 rec=pd.read_csv(A/'inputs/dense_records.csv');feat=np.load(A/'features/dense_grids.npy',mmap_mode='r');mask=np.load(A/'features/dense_targets.npy',mmap_mode='r')
 targets=[det_target(r) if r.split=='manual' else None for _,r in rec.iterrows()]
 logs=[]
 for fit in DAYS:
  train=rec[rec.split.eq('manual')&rec.day.eq(fit)].index.to_numpy()
  evaluation=rec[rec.split.eq('test')|((rec.split=='oof')&rec.day.ne(fit))|((rec.split=='manual')&rec.day.ne(fit))].index.to_numpy()
  # Explicit fold independence, including every object from a frame.
  assert not set(rec.loc[train,'frame_id'])&set(rec.loc[evaluation[rec.loc[evaluation,'split'].ne('test')],'frame_id'])
  frame_n=rec.loc[train].groupby('frame_id').size();prob=np.array([1/frame_n[f] for f in rec.loc[train,'frame_id']]);prob/=prob.sum()
  for kind in ['linear_mask','conv_detector']:
   name=f'{kind}_fit{fit[-2:]}';torch.manual_seed(3101);rng=np.random.default_rng(3101)
   model=Head(kind);opt=torch.optim.AdamW(model.parameters(),lr=.0007 if kind=='linear_mask' else .0004,weight_decay=.01)
   steps=450 if kind=='linear_mask' else 700;start=time.time();running=[]
   for step in range(steps):
    idx=rng.choice(train,8,replace=True,p=prob);x=torch.from_numpy(np.array(feat[idx],dtype=np.float32))
    if kind=='linear_mask':
     y=torch.from_numpy(np.array(mask[idx],dtype=np.float32));logit=model(x)
     # Balanced soft box occupancy; it must not be interpreted as segmentation truth.
     loss=F.binary_cross_entropy_with_logits(logit,y,weight=1+3*y)
    else:
     tt=tuple(torch.from_numpy(np.stack([targets[i][k] for i in idx])) for k in range(4));loss=det_loss(model(x),tt)
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();running.append(float(loss.detach()))
    if step%100==0:print('FIT',name,step,round(np.mean(running[-100:]),4),round(time.time()-start,1),flush=True)
   torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();rows=[];maps=[];mapidx=[]
   for off in range(0,len(evaluation),16):
    ids=evaluation[off:off+16]
    with torch.inference_mode():pred=model(torch.from_numpy(np.array(feat[ids],dtype=np.float32)))
    for j,i in enumerate(ids):
     r=rec.loc[i];pr=pred[j] if kind=='linear_mask' else tuple(v[j] for v in pred)
     for b in get_boxes(pr,r,kind):rows.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,**b))
     if r.split in ['oof','test']:
      maps.append((pr.sigmoid() if kind=='linear_mask' else torch.sqrt(pr[0].sigmoid()*pr[2].sigmoid())).numpy().astype(np.float16));mapidx.append(r.id)
   pd.DataFrame(rows).to_csv(A/f'features/{name}_boxes.csv',index=False)
   np.savez_compressed(A/f'features/{name}_maps.npz',maps=np.stack(maps),ids=np.array(mapidx))
   entry=dict(name=name,fit_day=fit,manual_crops=len(train),manual_frames=int(rec.loc[train].frame_id.nunique()),steps=steps,last_loss=float(np.mean(running[-100:])),seconds=time.time()-start,fit_frame_ids=sorted(rec.loc[train].frame_id.unique().tolist()))
   logs.append(entry);(A/'models/training_manifest.json').write_text(json.dumps(logs,indent=2));print('DONE',name,round(time.time()-start,1),len(rows),flush=True)
if __name__=='__main__':main()
