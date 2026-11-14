"""Automatic, day-isolated pseudo-label candidates. Visual review is POOL only.
Test candidates stay automatic and are reserved for a separate transductive fit.
"""
from vision import *
import argparse,time,gc,joblib
from ultralytics import YOLO
import yolo_runtime
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--select-only',action='store_true');p.add_argument('--geometry-only',action='store_true');p.add_argument('--resume-geometry',action='store_true');args=p.parse_args()
out=A/'pool_expansion';out.mkdir(exist_ok=True)
q,te,h=inputs();old=pd.read_csv(A/'inputs/records.csv');rng=np.random.default_rng(4142)
machine=pd.read_csv(D/'machine_predictions.csv').rename(columns={'detection_id':'id'});machine=machine[machine.class_name.isin(CL)].copy();machine['day']=machine.frame_id.map(day)
machine['ts']=pd.to_datetime(machine.frame_id.str.split('_').str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64')/1e9
used=set(old.frame_id)|set(h.frame_id);used_ts={d:np.array([pd.to_datetime(f.split('_')[-2]+f.split('_')[-1],format='%Y%m%d%H%M%S').timestamp() for f in used if day(f)==d]) for d in DAYS}
selected=[]
for (d,c),g in machine[machine.day.isin(DAYS)&~machine.frame_id.isin(used)].groupby(['day','class_name']):
 g=g[g.confidence.ge(.65 if c=='nude_hands' else .75)].copy();g['draw']=rng.random(len(g));g['cell']=((g.x1+g.x2)/2/160).astype(int).astype(str)+'_'+((g.y1+g.y2)/2/90).astype(int).astype(str)
 # Round-robin spatial strata, with time separation; avoid a single seated person.
 g['within_cell']=g.groupby('cell').draw.rank(method='first');g=g.sort_values(['within_cell','draw']);times=[];n=0
 for _,r in g.iterrows():
  if len(used_ts[d]) and abs(used_ts[d]-r.ts).min()<120:continue
  if times and min(abs(np.array(times)-r.ts))<120:continue
  selected.append(r);times.append(r.ts);n+=1
  if n>=75:break
pool=pd.DataFrame(selected).drop_duplicates('id');pool['split']='new_pool'
test=te.copy();test['split']='auto_test';rec=pd.concat([pool,test],ignore_index=True)
for ix,r in rec.iterrows():
 b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],4);cc=(b[:2]+b[2:])/2
 rec.loc[ix,CB]=np.r_[np.maximum(0,np.floor(cc-wh*.95)),np.minimum([1280,720],np.ceil(cc+wh*.95))]
rec.to_csv(out/'candidates.csv',index=False)
print('SELECTED',pool.groupby(['day','class_name']).size().to_dict(),'test',len(test),flush=True)
if args.select_only:raise SystemExit
assert torch.backends.mps.is_available();device='mps';evidence=rec[['id','frame_id','day','class_name','split','confidence']+B].copy();imgcache={}
def pixels(indices):
 batch=[]
 for ix in indices:
  r=rec.iloc[ix]
  if r.frame_id not in imgcache:imgcache[r.frame_id]=cv2.resize(cv2.imread(str(D/f'{r.frame_id}.jpg')),(1280,720))
  b=r[CB].to_numpy(int);im=imgcache[r.frame_id][b[1]:b[3],b[0]:b[2],::-1]
  batch.append(cv2.resize(im,(224,224),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
 return np.stack(batch)
class Geometry(nn.Module):
 def __init__(self,dim,grid=16):
  super().__init__();self.dim=dim;self.grid=grid;self.net=nn.Sequential(nn.Conv2d(dim+7,96,1),nn.GroupNorm(8,96),nn.GELU(),nn.Conv2d(96,48,3,padding=1),nn.GroupNorm(8,48),nn.GELU(),nn.Conv2d(48,1,1))
 def forward(self,x,m,e):
  n=len(x);g=self.grid;xy=torch.stack(torch.meshgrid(torch.linspace(-1,1,g,device=x.device),torch.linspace(-1,1,g,device=x.device),indexing='ij')).expand(n,-1,-1,-1)
  cls=e[:,-4:,None,None].expand(-1,-1,g,g);return self.net(torch.cat([F.normalize(x,dim=1)*np.sqrt(self.dim),m.reshape(n,1,g,g),xy,cls],1))
if args.resume_geometry:
 evidence=pd.read_csv(out/'geometry_evidence.csv');assert evidence.id.tolist()==rec.id.tolist()
for key in ([] if args.resume_geometry else ['vit_b','vit_l']):
 model=load(key).to(device);dim=model.config.hidden_size;mask,extra=aux(rec,16);feat=np.empty((len(rec),dim,16,16),np.float16)
 mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
 with torch.inference_mode():
  for off in range(0,len(rec),8):
   ix=np.arange(off,min(off+8,len(rec)));x=(torch.tensor(pixels(ix),device=device).float()/255-mean)/std;z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,dim,16,16);feat[ix]=z.cpu().numpy().astype(np.float16)
   if off%160==0:print('POOL FEATURES',key,off,len(rec),flush=True)
 del model;gc.collect();torch.mps.empty_cache();np.save(out/f'{key}_patches.npy',feat)
 for fit in DAYS:
  use=rec.day.eq(fit)|rec.split.eq('auto_test');ids=np.flatnonzero(use);net=Geometry(dim).to(device);net.load_state_dict(torch.load(A/f'models/{key}_geometry_fit{fit[-2:]}.pt',map_location=device,weights_only=True));net.eval()
  with torch.inference_mode():
   for off in range(0,len(ids),24):
    ii=ids[off:off+24];pr=net(torch.tensor(feat[ii].astype(np.float32),device=device),mask[ii].to(device),extra[ii].to(device)).sigmoid().cpu().numpy()[:,0]
    for ix,pm in zip(ii,pr):
     r=rec.iloc[ix];cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];n,lab,stats,cent=cv2.connectedComponentsWithStats((pm>=.5).astype(np.uint8),8);boxes=[];confs=[]
     for j in range(1,n):
      x,y,w,hh,ar=stats[j]
      if ar<2:continue
      boxes.append(np.array([x,y,x+w,y+hh])/16*np.tile(size,2)+np.tile(cb[:2],2));confs.append(float(pm[lab==j].mean()))
     ov=iou(r[B],boxes)[0] if boxes else []
     k=int(np.argmax(ov)) if len(ov) else None;tag=f'{key}_{fit[-2:]}'
     evidence.loc[ix,tag+'_iou']=ov[k] if k is not None else 0.;evidence.loc[ix,tag+'_conf']=confs[k] if k is not None else 0.
     if k is not None:
      for col,val in zip(B,boxes[k]):evidence.loc[ix,tag+'_'+col]=val
  del net;gc.collect();torch.mps.empty_cache()
 evidence.to_csv(out/'geometry_evidence.csv',index=False)
if args.geometry_only:raise SystemExit
for fit in DAYS:
 model=YOLO(str(A/f'yolo_runs/fit{fit[-2:]}/weights/last.pt'));ids=np.flatnonzero(rec.day.eq(fit)|rec.split.eq('auto_test'))
 for off in range(0,len(ids),8):
  ii=ids[off:off+8];images=[]
  for ix in ii:
   r=rec.iloc[ix];b=r[CB].to_numpy(int)
   if r.frame_id not in imgcache:imgcache[r.frame_id]=cv2.resize(cv2.imread(str(D/f'{r.frame_id}.jpg')),(1280,720))
   images.append(imgcache[r.frame_id][b[1]:b[3],b[0]:b[2]].copy())
  pred=model.predict(images,imgsz=640,device='mps',conf=.1,iou=.7,verbose=False)
  for ix,det in zip(ii,pred):
   r=rec.iloc[ix];cb=r[CB].to_numpy(float);classes=det.boxes.cls.cpu().numpy();m=classes==CL.index(r.class_name);boxes=det.boxes.xyxy.cpu().numpy()[m]+np.tile(cb[:2],2);conf=det.boxes.conf.cpu().numpy()[m];ov=iou(r[B],boxes)[0] if len(boxes) else []
   k=int(np.argmax(ov)) if len(ov) else None;tag=f'yolo_{fit[-2:]}'
   evidence.loc[ix,tag+'_iou']=ov[k] if k is not None else 0.;evidence.loc[ix,tag+'_conf']=conf[k] if k is not None else 0.
  if off%160==0:print('POOL YOLO',fit,off,len(ids),flush=True)
 del model;gc.collect();torch.mps.empty_cache()
evidence['geometry_admit']=False
for ix,r in evidence.iterrows():
 days=[r.day[-2:]] if r.split=='new_pool' else ['25','26']
 agree=[]
 for d in days:
  agree.append(min(r[f'vit_b_{d}_iou'],r[f'vit_l_{d}_iou'])>=.68 and min(r[f'vit_b_{d}_conf'],r[f'vit_l_{d}_conf'])>=.7 and r[f'yolo_{d}_iou']>=.6 and r[f'yolo_{d}_conf']>=.35)
 evidence.loc[ix,'geometry_admit']=all(agree)
evidence.to_csv(out/'evidence.csv',index=False)
print('AUTO ADMITTED',evidence[evidence.geometry_admit].groupby(['split','class_name']).size().to_dict(),flush=True)
