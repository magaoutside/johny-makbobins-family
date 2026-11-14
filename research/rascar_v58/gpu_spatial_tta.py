"""True RGB horizontal flip TTA for the new three-seed spatial DINO-B heads."""
from pathlib import Path
import sys,time,json
sys.path.insert(0,'/kaggle/working/rascar_v56')
from dense_backbone import overlap,image
import numpy as np,pandas as pd,torch,cv2
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
R=Path('/kaggle/working');A=R/'rascar_v56';O=R/'rascar_v58';device='cuda:0';torch.set_num_threads(2);cv2.setNumThreads(1)
rec=pd.read_csv(R/'rascar_v31/inputs/dense_records.csv');ix=rec.index[rec.split.ne('manual')].to_numpy();x=np.load(A/'features/vit_b_dense.npy',mmap_mode='r');pos={i:j for j,i in enumerate(ix)}
model=AutoModel.from_pretrained(str(R/'rascar_v41/model_cache/vit_b'),local_files_only=True).to(device).eval()
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None];flipped=[];start=time.time()
for at in range(0,len(ix),2):
 batch=[]
 for i in ix[at:at+2]:
  r=rec.loc[i];c=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);im=image(r.frame_id)[c[1]:c[3],c[0]:c[2]][:,::-1]
  batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
 xb=torch.from_numpy(np.stack(batch)).to(device).float()/255
 with torch.inference_mode():z=model(pixel_values=(xb-mean)/std).last_hidden_state[:,1:].transpose(1,2).reshape(-1,768,32,32)
 flipped.append(z.cpu().numpy().astype(np.float16))
 if at%100==0:print('FLIP_EXTRACT',at,len(ix),round(time.time()-start,1),flush=True)
flipped=np.concatenate(flipped);del model;torch.cuda.empty_cache();np.save(O/'features/spatial_tta_flip_grids.npy',flipped)
class Head(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Conv2d(768,64,1),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(64,4,1))
 def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(768))
checks=[]
for fit in ['2026-10-25','2026-10-26']:
 ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit))].to_numpy();orig=[];flip=[]
 for seed in [3101,3102,3103]:
  name=f'vit_b_spatial_s{seed}_fit{fit[-2:]}';head=Head().to(device);head.load_state_dict(torch.load(O/f'models/{name}.pt',map_location=device,weights_only=True));head.eval();om=[];fm=[]
  for at in range(0,len(ev),16):
   bi=ev[at:at+16]
   with torch.inference_mode():
    om.append(head(torch.from_numpy(np.array(x[bi],np.float32)).to(device)).sigmoid().cpu().numpy())
    fm.append(head(torch.from_numpy(np.array(flipped[[pos[i] for i in bi]],np.float32)).to(device)).sigmoid().cpu().numpy()[:,:,:,::-1])
  om=np.concatenate(om);fm=np.concatenate(fm);old=np.load(O/f'features/{name}_maps.npz');assert np.array_equal(old['ids'],rec.loc[ev,'id'].to_numpy());err=float(abs(old['maps']-om).max());assert err<1e-6
  checks.append(dict(name=name,max_map_replay_error=err));orig.append(om);flip.append(fm)
  for w in [.5,1.]:rec.loc[ev,['id','split','day']].assign(fit_day=fit,overlap=overlap((1-w)*om+w*fm,rec.loc[ev])).to_csv(O/f'features/tta_spatial_w{w}_s{seed}_fit{fit[-2:]}_evidence.csv',index=False)
 for w in [.5,1.]:
  maps=(1-w)*np.mean(orig,0)+w*np.mean(flip,0)
  rec.loc[ev,['id','split','day']].assign(fit_day=fit,overlap=overlap(maps,rec.loc[ev])).to_csv(O/f'features/tta_spatial_w{w}_mean3_fit{fit[-2:]}_evidence.csv',index=False)
(O/'audit/spatial_checkpoint_replay.json').write_text(json.dumps(checks,indent=2));print('SPATIAL_TTA_COMPLETE',round(time.time()-start,1),flush=True)
