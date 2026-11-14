"""Frozen dense DINOv2-S grids at 448px. No test annotations or targets."""
from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parent.parent; A=ROOT/'rascar_v31'
os.environ.setdefault('MPLCONFIGDIR',str(A/'model_cache/mpl'))
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,h,B,CL,DAYS
from features23 import image
import numpy as np,pandas as pd,cv2,torch
from transformers import AutoModel

def build_records():
 rng=np.random.default_rng(3101);rows=[]
 for j,r in h.iterrows():
  if r.class_name not in CL+['phone','full_plate','empty_plate','hair','personal_items']:continue
  for aug in range(2 if r.class_name in CL else 1):
   b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
   center=cc+rng.uniform(-.35,.35,2)*wh;size=wh*rng.uniform(2.,3.2)
   lo=np.maximum(0,np.floor(center-size/2));hi=np.minimum([1280,720],np.ceil(center+size/2))
   rows.append(dict(id=f'manual_{j}_{aug}',frame_id=r.frame_id,day=r.day,class_name=r.class_name,split='manual',origin_id=f'manual_{j}',**dict(zip(B,b)),**dict(zip(['crop_x1','crop_y1','crop_x2','crop_y2'],np.r_[lo,hi]))))
 for split,df in [('oof',q),('test',te)]:
  for _,r in df.iterrows():
   b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;size=wh*2.4
   lo=np.maximum(0,np.floor(cc-size/2));hi=np.minimum([1280,720],np.ceil(cc+size/2))
   day=pd.to_datetime(r.frame_id.split('_')[-2],format='%Y%m%d').strftime('%Y-%m-%d')
   rows.append(dict(id=r.id,frame_id=r.frame_id,day=day,class_name=r.class_name,split=split,origin_id=r.id,**dict(zip(B,b)),**dict(zip(['crop_x1','crop_y1','crop_x2','crop_y2'],np.r_[lo,hi]))))
 d=pd.DataFrame(rows);d['feature_index']=np.arange(len(d));return d

def occupancy(r,grid=32):
 cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);wh=(cb[2:]-cb[:2])/grid
 xx=cb[0]+np.arange(grid)*wh[0];yy=cb[1]+np.arange(grid)*wh[1]
 out=np.zeros((len(CL),grid,grid),np.float32)
 for c,cls in enumerate(CL):
  for b in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(cls)][B].to_numpy(float):
   wx=np.maximum(0,np.minimum(xx+wh[0],b[2])-np.maximum(xx,b[0]))/wh[0]
   wy=np.maximum(0,np.minimum(yy+wh[1],b[3])-np.maximum(yy,b[1]))/wh[1]
   out[c]=np.maximum(out[c],wy[:,None]*wx[None,:])
 return out

def main():
 torch.set_num_threads(3);torch.set_num_interop_threads(1)
 rec=build_records();rec.to_csv(A/'inputs/dense_records.csv',index=False)
 target=np.stack([occupancy(r) if r.split!='test' else np.zeros((4,32,32),np.float32) for _,r in rec.iterrows()])
 np.save(A/'features/dense_targets.npy',target.astype(np.float16))
 model=AutoModel.from_pretrained(str(A/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
 shape=(len(rec),384,32,32);dst=np.lib.format.open_memmap(A/'features/dense_grids.npy',mode='w+',dtype=np.float16,shape=shape)
 mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
 start=time.time()
 for off in range(0,len(rec),4):
  batch=[]
  for _,r in rec.iloc[off:off+4].iterrows():
   b=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);im=image(r.frame_id)[b[1]:b[3],b[0]:b[2],::-1]
   batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
  x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
  with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:,:].transpose(1,2).reshape(-1,384,32,32)
  dst[off:off+len(batch)]=z.numpy().astype(np.float16)
  if off%80==0:print('EXTRACT',off,len(rec),round(time.time()-start,1),flush=True)
 dst.flush();(A/'features/extraction_complete.json').write_text(json.dumps(dict(count=len(rec),shape=shape,seconds=time.time()-start,backbone='facebook/dinov2-small',input=448,grid=32,manual_box_supervision=True,test_annotations_used=False),indent=2))
 print('DONE',len(rec),time.time()-start,flush=True)
if __name__=='__main__':main()
