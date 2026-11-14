"""Frozen DINO grid + regularized nonlinear occupancy decoder, cross-day."""
from pathlib import Path
import sys,os,time,json
sys.path.insert(0,'/kaggle/working/rascar_v56')
from dense_backbone import overlap
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
A=Path('/kaggle/working/rascar_v56');O=Path('/kaggle/working/rascar_v58');R=A.parent
for d in ['features','models','audit','logs']:(O/d).mkdir(exist_ok=True,parents=True)
backbone=sys.argv[1];device=sys.argv[2];torch.set_num_threads(2)
rec=pd.read_csv(R/'rascar_v31/inputs/dense_records.csv');x=np.load(A/f'features/{backbone}_dense.npy',mmap_mode='r');y=np.load(R/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
manual=rec.index[rec.split.eq('manual')].to_numpy();flip_index={v:len(rec)+i for i,v in enumerate(manual)}
class Head(nn.Module):
 def __init__(self,width,kind):
  super().__init__();self.width=width
  if kind=='mlp':self.net=nn.Sequential(nn.Conv2d(width,128,1),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(128,4,1))
  else:self.net=nn.Sequential(nn.Conv2d(width,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(64,4,1))
 def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(self.width))
for kind in ['mlp','spatial']:
 for seed in [3101,3102,3103]:
  evidence=[]
  for fit in ['2026-10-25','2026-10-26']:
   name=f'{backbone}_{kind}_s{seed}_fit{fit[-2:]}';dest=O/f'features/{name}_evidence.csv'
   if dest.exists():continue
   tr=rec.index[rec.split.eq('manual')&rec.day.eq(fit)].to_numpy();ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit))].to_numpy()
   assert set(rec.loc[tr,'frame_id']).isdisjoint(rec.loc[ev,'frame_id'])
   freq=rec.loc[tr].groupby('frame_id').size();prob=np.array([1/freq[f] for f in rec.loc[tr,'frame_id']]);prob/=prob.sum()
   torch.manual_seed(seed);rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+15000)
   model=Head(x.shape[1],kind).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03);losses=[];start=time.time()
   # Original image and actual horizontal RGB flip were separately encoded.
   for step in range(450):
    ix=rng.choice(tr,8,replace=True,p=prob);xx=[];yy=[]
    for i in ix:
     flip=arng.random()<.5;xx.append(x[flip_index[i] if flip else i]);yy.append(y[i,:,:,::-1] if flip else y[i])
    xb=torch.from_numpy(np.array(xx,np.float32)).to(device);yb=torch.from_numpy(np.array(yy,np.float32)).to(device)
    logits=model(xb);loss=F.binary_cross_entropy_with_logits(logits,yb,weight=1+3*yb)
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
   torch.save({k:v.cpu() for k,v in model.state_dict().items()},O/f'models/{name}.pt');model.eval();maps=[]
   for at in range(0,len(ev),16):
    with torch.inference_mode():maps.append(model(torch.from_numpy(np.array(x[ev[at:at+16]],np.float32)).to(device)).sigmoid().cpu().numpy())
   maps=np.concatenate(maps);np.savez_compressed(O/f'features/{name}_maps.npz',ids=rec.loc[ev,'id'].to_numpy(dtype=str),maps=maps)
   pd.DataFrame(rec.loc[ev,['id','split','day']]).assign(fit_day=fit,overlap=overlap(maps,rec.loc[ev])).to_csv(dest,index=False)
   info=dict(backbone=backbone,kind=kind,seed=seed,fit_day=fit,train_frames=sorted(rec.loc[tr,'frame_id'].unique()),validation_frames=sorted(rec.loc[ev[rec.loc[ev,'split'].eq('oof')],'frame_id'].unique()),steps=450,seconds=time.time()-start,loss=float(np.mean(losses[-100:])))
   (O/f'models/{name}.json').write_text(json.dumps(info,indent=2));print('DENSE_DONE',name,round(time.time()-start,1),flush=True)
 # Ensemble probability maps before decoding boxes, just like prior linear control.
 for fit in ['2026-10-25','2026-10-26']:
  maps=[];ids=None
  for seed in [3101,3102,3103]:
   d=np.load(O/f'features/{backbone}_{kind}_s{seed}_fit{fit[-2:]}_maps.npz');maps.append(d['maps']);ids=d['ids']
  rr=rec.set_index('id').loc[ids]
  rr[['split','day']].reset_index().assign(fit_day=fit,overlap=overlap(np.mean(maps,axis=0),rr)).to_csv(O/f'features/{backbone}_{kind}_mean3_fit{fit[-2:]}_evidence.csv',index=False)
print('DENSE_COMPLETE',backbone,flush=True)
