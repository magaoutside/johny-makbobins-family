"""Correct the equal-minimum-crop ablation; prepare ALL hand candidates for review."""
from pathlib import Path
import sys,gc,json
import numpy as np,pandas as pd,torch
from PIL import Image,ImageDraw
from ultralytics import YOLO
A=Path('/kaggle/working/rascar_v20');D=Path('/kaggle/working/competition_data');fit,device=sys.argv[1:3];device=int(device)
B=['x1','y1','x2','y2'];torch.set_num_threads(2);paths={p.stem:p for p in D.rglob('*.jpg')}
out=A/'pool_review'/fit;out.mkdir(parents=True,exist_ok=True)
rows=pd.read_csv(A/'pool_v2'/fit/'pseudo_label_ledger.csv')
def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4);inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-9)
model=YOLO(A/f'weights/hand_{fit}.pt')
hands=rows.class_name.eq('nude_hands')
for n,(ix,r) in enumerate(rows[hands&rows.confidence.ge(.75)&rows.hand_iou.ge(.6)&rows.semantic_conf.ge(.8)].iterrows()):
 im=Image.open(paths[r.frame_id]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC);b=r[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;pp=[];crops=[]
 for scale,minimum in [(4,192),(6,288)]:
  side=max(float(wh.max()*scale),minimum);crop=np.clip(np.r_[cc-side/2,cc+side/2],0,[1280,720,1280,720]).astype(int);crops.append(crop.tolist())
  rr=model.predict(np.array(im.crop(tuple(crop)))[:,:,::-1],imgsz=800,device=device,conf=.15,iou=.5,verbose=False)[0]
  bb=rr.boxes.xyxy.cpu().numpy()+crop[[0,1,0,1]];conf=rr.boxes.conf.cpu().numpy();ov=iou(b,bb)[0] if len(bb) else np.array([])
  if len(ov):j=int(ov.argmax());pp.append((bb[j],float(conf[j]),float(ov[j])))
 assert crops[0]!=crops[1]
 rows.loc[ix,['hand_iou','hand_conf','hand_stability']]=[min(p[2] for p in pp),min(p[1] for p in pp),float(iou(pp[0][0],pp[1][0])[0,0])] if len(pp)==2 else [0.,0.,0.]
 if n%100==0:print('INDEPENDENT_CONTEXT_CHECK',fit,n,flush=True)
rows['review_candidate']=False
m=hands&rows.confidence.ge(.75)&rows.hand_iou.ge(.65)&rows.hand_conf.ge(.4)&rows.hand_stability.ge(.7)&rows.semantic_conf.ge(.8)&~rows.overlapping_source
rows.loc[m,'review_candidate']=True
rows.loc[rows.class_name.eq('cup')&rows.accepted,'review_candidate']=True
rows.to_csv(out/'review_ledger.csv',index=False)
candidate=rows[rows.review_candidate&rows.source.eq('machine_pool')].sort_values(['class_name','frame_id','id']).copy();candidate['review_index']=np.arange(len(candidate));candidate.to_csv(out/'all_review_candidates.csv',index=False)
for start in range(0,len(candidate),24):
 canvas=Image.new('RGB',(1200,1200),'white');draw=ImageDraw.Draw(canvas)
 for j,r in enumerate(candidate.iloc[start:start+24].itertuples()):
  im=Image.open(paths[r.frame_id]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC);b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2
  # Show context AND enlarged box side by side; red border belongs to query.
  for view,side in enumerate([max(110,wh.max()*3.5),max(45,wh.max()*1.35)]):
   ci=im.copy();ImageDraw.Draw(ci).rectangle(tuple(b),outline='red',width=2);ci=ci.crop(tuple(np.r_[cc-side/2,cc+side/2]));ci=ci.resize((145,145),Image.Resampling.BICUBIC);canvas.paste(ci,(j%4*300+view*150,j//4*200+35))
  draw.text((j%4*300+4,j//4*200+5),f'{start+j} {r.class_name} IoU {r.hand_iou:.2f}',fill='black')
 canvas.save(out/f'all_{start//24:02d}.jpg',quality=92)
cal=rows[rows.review_candidate&rows.source.eq('held_day_calibration')]
print('REVIEW_READY',fit,'candidate counts',candidate.class_name.value_counts().to_dict(),'held accepted',len(cal),'incorrect',cal.error.sum(),flush=True)
