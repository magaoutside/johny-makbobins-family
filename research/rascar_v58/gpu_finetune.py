"""Train a linear DINOv2-S decoder, then unfreeze last transformer block.
Official facebook/dinov2-small weights. Train images and targets: fit-day only.
"""
from pathlib import Path
import sys,time,json
sys.path.insert(0,'/kaggle/working/rascar_v56')
from dense_backbone import overlap
import numpy as np,pandas as pd,cv2,torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from functools import lru_cache
R=Path('/kaggle/working');A=R/'rascar_v56';O=R/'rascar_v58';D=R/'rascar_data_full'
fit=sys.argv[1];device=sys.argv[2];seed=int(sys.argv[3]) if len(sys.argv)>3 else 3101
backbone=sys.argv[4] if len(sys.argv)>4 else 'vit_s'
width={'vit_s':384,'vit_b':768}[backbone]
model_id={'vit_s':'facebook/dinov2-small','vit_b':'facebook/dinov2-base'}[backbone]
print('DEVICE',torch.cuda.get_device_name(device),backbone,flush=True)
rec=pd.read_csv(R/'rascar_v31/inputs/dense_records.csv');Y=np.load(R/'rascar_v31/features/dense_targets.npy',mmap_mode='r');torch.set_num_threads(2);cv2.setNumThreads(1)
@lru_cache(maxsize=150)
def image(frame):
 im=cv2.imread(str(D/f'{frame}.jpg'));assert im is not None,frame
 return cv2.resize(im,(1280,720))[:,:,::-1].copy()
def batch(indices,flips):
 xx=[]
 for i,flip in zip(indices,flips):
  r=rec.loc[i];c=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);im=image(r.frame_id)[c[1]:c[3],c[0]:c[2]]
  if flip:im=im[:,::-1]
  xx.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
 return torch.from_numpy(np.stack(xx)).to(device).float()/255
class Network(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=AutoModel.from_pretrained(model_id);self.head=nn.Conv2d(width,4,1)
  for p in self.encoder.parameters():p.requires_grad=False
 def forward(self,x):
  out=self.encoder(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,width,32,32)
  return self.head(F.normalize(out,dim=1)*np.sqrt(width))
torch.manual_seed(seed);rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+15000);net=Network().to(device)
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
tr=rec.index[rec.split.eq('manual')&rec.day.eq(fit)].to_numpy();ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit))].to_numpy()
assert set(rec.loc[tr,'frame_id']).isdisjoint(rec.loc[ev,'frame_id'])
freq=rec.loc[tr].groupby('frame_id').size();prob=np.array([1/freq[f] for f in rec.loc[tr,'frame_id']]);prob/=prob.sum()
opt=torch.optim.AdamW(net.head.parameters(),lr=.0007,weight_decay=.01);scaler=torch.amp.GradScaler('cuda');start=time.time();losses=[]
def evaluate(stage):
 name=f'finetune_{backbone}_s{seed}_fit{fit[-2:]}_{stage}';net.eval();maps=[]
 for at in range(0,len(ev),8):
  xb=batch(ev[at:at+8],np.zeros(len(ev[at:at+8]),bool))
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):maps.append(net((xb-mean)/std).sigmoid().float().cpu().numpy())
 maps=np.concatenate(maps);np.savez_compressed(O/f'features/{name}_maps.npz',ids=rec.loc[ev,'id'].to_numpy(dtype=str),maps=maps)
 rec.loc[ev,['id','split','day']].assign(fit_day=fit,overlap=overlap(maps,rec.loc[ev])).to_csv(O/f'features/{name}_evidence.csv',index=False)
 # Last block + norm + head are sufficient with the official frozen backbone.
 state={k:v.cpu() for k,v in net.state_dict().items() if k.startswith('head.') or k.startswith('encoder.encoder.layer.11.') or k.startswith('encoder.layernorm.')}
 torch.save(state,O/f'models/{name}.pt')
 (O/f'models/{name}.json').write_text(json.dumps(dict(model=model_id,source='https://huggingface.co/'+model_id,device=torch.cuda.get_device_name(device),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),fit_day=fit,seed=seed,stage=stage,train_frames=sorted(rec.loc[tr,'frame_id'].unique()),validation_frames=sorted(rec.loc[ev[rec.loc[ev,'split'].eq('oof')],'frame_id'].unique()),seconds=time.time()-start),indent=2));print('EVAL_DONE',name,round(time.time()-start,1),flush=True)
 net.train();net.encoder.eval()
for step in range(750):
 if step==350:evaluate('tuned350')
 if step==150:
  evaluate('frozen150')
  for p in net.encoder.encoder.layer[-1].parameters():p.requires_grad=True
  for p in net.encoder.layernorm.parameters():p.requires_grad=True
  opt=torch.optim.AdamW([{'params':net.head.parameters(),'lr':.0003},{'params':[p for p in net.encoder.parameters() if p.requires_grad],'lr':.00001}],weight_decay=.03)
 ix=rng.choice(tr,6,replace=True,p=prob);flips=arng.random(len(ix))<.5;xb=batch(ix,flips)
 # Modest photometric augmentation does not change target geometry.
 brightness=torch.from_numpy(arng.uniform(-.08,.08,size=(len(ix),1,1,1)).astype(np.float32)).to(device)
 contrast=torch.from_numpy(arng.uniform(.9,1.1,size=(len(ix),1,1,1)).astype(np.float32)).to(device)
 xb=((xb-.5)*contrast+.5+brightness).clamp(0,1)
 yy=np.array([Y[i,:,:,::-1] if flip else Y[i] for i,flip in zip(ix,flips)],np.float32);yb=torch.from_numpy(yy).to(device)
 with torch.autocast('cuda',dtype=torch.float16):loss=F.binary_cross_entropy_with_logits(net((xb-mean)/std),yb,weight=1+3*yb)
 opt.zero_grad();scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],5);scaler.step(opt);scaler.update();losses.append(float(loss.detach().cpu()))
 if (step+1)%50==0:print('STEP',fit,seed,step+1,round(time.time()-start,1),np.mean(losses[-50:]),flush=True)
evaluate('tuned750');print('FINETUNE_COMPLETE',fit,seed,flush=True)
