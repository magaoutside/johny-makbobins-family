"""FCOS-head adaptation using QFL, with/without DFL (not full original GFL).
All warm-start parameters and supervision come exclusively from the fit day.
"""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';V31=ROOT/'rascar_v31';sys.path.insert(0,str(V31))
from dense_train import torch,np,pd,nn,F,Head,det_target,CL,B,DAYS,iou

class QualityHead(nn.Module):
 def __init__(self,kind,fit):
  super().__init__();self.kind=kind
  old=Head('conv_detector');old.load_state_dict(torch.load(V31/f'models/conv_detector_fit{fit[-2:]}.pt',weights_only=True))
  self.net=old.net;self.cls=old.cls;self.reg=old.reg if kind=='qfl_point' else nn.Conv2d(64,4*33,1)
  if kind=='qfl_distribution':
   nn.init.normal_(self.reg.weight,std=.01)
   with torch.no_grad():self.reg.bias.copy_((-((torch.arange(33)-6)/3)**2/2).repeat(4))
 def forward(self,x):
  z=self.net(F.normalize(x,dim=1)*np.sqrt(384));logit=self.cls(z);raw=self.reg(z)
  if self.kind=='qfl_point':return logit,F.softplus(raw),None
  raw=raw.reshape(-1,4,33,32,32);prob=raw.softmax(2);bins=torch.arange(33,dtype=prob.dtype)[None,None,:,None,None]/32
  return logit,(prob*bins).sum(2),raw

def quality_loss(pred,tt):
 logits,dist,raw=pred;y,t,cent,pos=tt;quality=torch.zeros_like(y);iou_loss=logits.sum()*0;dfl=iou_loss
 if pos.any():
  a=dist.permute(0,2,3,1)[pos];b=t.permute(0,2,3,1)[pos]
  inter=(torch.minimum(a[:,0],b[:,0])+torch.minimum(a[:,2],b[:,2]))*(torch.minimum(a[:,1],b[:,1])+torch.minimum(a[:,3],b[:,3]))
  aa=(a[:,0]+a[:,2])*(a[:,1]+a[:,3]);bb=(b[:,0]+b[:,2])*(b[:,1]+b[:,3]);ov=inter/(aa+bb-inter).clamp_min(1e-6)
  qm=torch.zeros_like(pos,dtype=torch.float32);qm[pos]=ov.detach();quality=y*qm[:,None]
  iou_loss=(1-ov).mean()
  if raw is not None:
   target=(b*32).clamp(0,31.999);left=target.long();right=left+1;wl=right-target;wr=target-left
   lp=raw.permute(0,3,4,1,2)[pos].reshape(-1,33)
   dfl=(F.cross_entropy(lp,left.flatten(),reduction='none')*wl.flatten()+F.cross_entropy(lp,right.flatten(),reduction='none')*wr.flatten()).mean()
  else:iou_loss+=1.5*F.smooth_l1_loss(a,b,beta=.05)
 qfl=(F.binary_cross_entropy_with_logits(logits,quality,reduction='none')*abs(quality-logits.sigmoid()).square()).sum()/pos.sum().clamp_min(1)
 return qfl+2*iou_loss+.25*dfl

def decode(pred,r):
 logit,dist,raw=pred;score=logit.sigmoid().numpy();dist=dist.numpy();cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);wh=cb[2:]-cb[:2];out=[]
 entropy=None
 if raw is not None:
  p=raw.softmax(1);entropy=(-(p*p.clamp_min(1e-8).log()).sum(1)/np.log(33)).mean(0).numpy()
 for k,cls in enumerate(CL):
  keep=[]
  for ix in np.argsort(score[k].ravel())[::-1][:160]:
   y,x=divmod(int(ix),32);confidence=float(score[k,y,x])
   if confidence<.08:break
   l,t,rr,bb=dist[:,y,x];cx=(x+.5)/32;cy=(y+.5)/32
   box=np.clip([cx-l,cy-t,cx+rr,cy+bb],0,1)*np.r_[wh,wh]+np.r_[cb[:2],cb[:2]]
   if keep and iou(box,np.array(keep)).max()>.5:continue
   keep.append(box);out.append(dict(class_name=cls,threshold=0.,confidence=confidence,boundary_entropy=float(entropy[y,x]) if entropy is not None else np.nan,**dict(zip(B,box))))
   if len(keep)>=8:break
 return out

def main():
 rec=pd.read_csv(A/'inputs/dense_records.csv');feat=np.load(V31/'features/dense_grids.npy',mmap_mode='r');targets=[det_target(r) if r.split=='manual' else None for _,r in rec.iterrows()];manifest=[]
 pool=pd.read_csv(A/'inputs/pool_control_records.csv');poolfeat=np.load(A/'features/pool_control_grids.npy').astype(np.float32)
 for fit in DAYS:
  train=rec[rec.split.eq('manual')&rec.day.eq(fit)].index.to_numpy();evaluation=rec[rec.split.eq('test')|rec.day.ne(fit)].index.to_numpy()
  assert not set(rec.loc[train,'frame_id'])&set(rec.loc[evaluation,'frame_id']);counts=rec.loc[train].groupby('frame_id').size();p=np.array([1/counts[f] for f in rec.loc[train,'frame_id']]);p/=p.sum()
  for kind in ['qfl_point','qfl_distribution']:
   torch.manual_seed(3204);rng=np.random.default_rng(3204);m=QualityHead(kind,fit);opt=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.01);losses=[];start=time.time()
   for step in range(900):
    ids=rng.choice(train,8,replace=True,p=p);x=torch.from_numpy(np.array(feat[ids],dtype=np.float32));tt=tuple(torch.from_numpy(np.stack([targets[i][k] for i in ids])) for k in range(4));loss=quality_loss(m(x),tt)
    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
    if step%150==0:print('TRAIN',kind,fit,step,round(np.mean(losses[-100:]),4),round(time.time()-start,1),flush=True)
   torch.save(m.state_dict(),A/f'models/{kind}_fit{fit[-2:]}.pt');m.eval();rows=[]
   for off in range(0,len(evaluation),16):
    ids=evaluation[off:off+16]
    with torch.inference_mode():pr=m(torch.from_numpy(np.array(feat[ids],dtype=np.float32)))
    for j,i in enumerate(ids):
     r=rec.loc[i]
     for b in decode(tuple(z[j] if z is not None else None for z in pr),r):rows.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,**b))
   for i,r in pool[pool.day.ne(fit)].iterrows():
    with torch.inference_mode():pr=m(torch.from_numpy(poolfeat[i:i+1]))
    for b in decode(tuple(z[0] if z is not None else None for z in pr),r):rows.append(dict(record_id=r.id,record_split='pool',frame_id=r.frame_id,fit_day=fit,**b))
   pd.DataFrame(rows).to_csv(A/f'features/{kind}_fit{fit[-2:]}_boxes.csv',index=False)
   manifest.append(dict(kind=kind,fit_day=fit,steps=900,crops=len(train),fit_frames=sorted(rec.loc[train,'frame_id'].unique()),warm_start=f'V31 conv_detector_fit{fit[-2:]}',last_loss=float(np.mean(losses[-100:])),seconds=time.time()-start,pool_used_for_training=False));(A/'models/quality_training_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',kind,fit,round(time.time()-start,1),len(rows),flush=True)
if __name__=='__main__':main()
