"""Review complete left-seat ROIs, not isolated automatically accepted boxes."""
from pathlib import Path
import json,zipfile,gc
import numpy as np,pandas as pd,torch
from PIL import Image,ImageDraw,ImageFont
from ultralytics import YOLO
A=Path('/kaggle/working/rascar_v22');roots=json.loads((A/'roots.json').read_text())
paths={p.stem:p for p in Path(roots['images']).rglob('*.jpg')}
rows=pd.read_csv(A/'inputs/pool_selected.csv');machine=pd.read_csv(A/'inputs/pool_machine_hands.csv')
B=['x1','y1','x2','y2'];dest=A/'pool_review';dest.mkdir(exist_ok=True)
def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4);z=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return z/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-z,1e-9)
font=ImageFont.load_default(size=18);allboxes=[];cards=[]
for day,g in rows.groupby('day'):
 weights=list(Path(roots['checkpoints']).glob(f'**/yolo_hands/{day}/expanded/weights/last.pt'));assert len(weights)==1,weights
 model=YOLO(weights[0])
 for r in g.itertuples():
  im=Image.open(paths[r.frame_id]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)
  roi=np.array([64,152,400,424]);crop=im.crop(tuple(roi));arr=np.asarray(crop)[:,:,::-1].copy()
  pred=[]
  for scale in [640,960]:
   p=model.predict(arr,imgsz=scale,conf=.12,iou=.5,device=0,verbose=False)[0]
   pred.append((p.boxes.xyxy.cpu().numpy()+np.tile(roi[:2],2),p.boxes.conf.cpu().numpy()))
  local=[]
  # Use only stable teacher boxes as provisional complete-ROI annotations.
  for j,(box,conf) in enumerate(zip(*pred[0])):
   ov=iou(box,pred[1][0])[0]
   if not len(ov):continue
   k=int(ov.argmax());stable=ov[k];confidence=min(conf,pred[1][1][k])
   if stable<.6 or confidence<.3:continue
   bb=(box+pred[1][0][k])/2
   if min(bb[:2]-roi[:2])<3 or min(roi[2:]-bb[2:])<3:continue
   local.append(dict(review_index=r.review_index,frame_id=r.frame_id,day=day,confidence=float(confidence),stability=float(stable),**dict(zip(B,bb))))
  local=sorted(local,key=lambda v:v['x1'])
  card=Image.new('RGB',(672,600),'white');d=ImageDraw.Draw(card)
  d.text((5,3),f'ROI {r.review_index} / {day} / {r.frame_id[-6:]}',fill='black',font=font)
  # Blue = proposed training boxes; orange = original machine proposals.
  disp=crop.resize((672,544));dr=ImageDraw.Draw(disp)
  for p in machine[machine.frame_id.eq(r.frame_id)].itertuples():
   bb=np.array([getattr(p,c) for c in B]);cc=(bb[:2]+bb[2:])/2
   if ((cc>roi[:2])&(cc<roi[2:])).all():dr.rectangle(tuple((bb-np.tile(roi[:2],2))*2),outline='orange',width=1)
  for j,p in enumerate(local):
   p['box_index']=j;p.update(dict(zip(['roi_'+c for c in B],roi)));allboxes.append(p)
   bb=np.array([p[c] for c in B]);v=(bb-np.tile(roi[:2],2))*2;dr.rectangle(tuple(v),outline='#00ddff',width=2);dr.text(tuple(v[:2]+[0,-19]),f'{j} ({p["confidence"]:.2f})',fill='#00ddff',font=font)
  card.paste(disp,(0,45));card.save(dest/f'roi_{r.review_index:03d}.jpg',quality=94);cards.append((r.review_index,card))
  print('REVIEW_ROI',r.review_index,len(local),flush=True)
 del model;gc.collect();torch.cuda.empty_cache()
pd.DataFrame(allboxes).to_csv(dest/'provisional_boxes.csv',index=False)
cards.sort(key=lambda x:x[0])
for start in range(0,len(cards),6):
 sheet=Image.new('RGB',(2016,1200),'white')
 for k,(_,card) in enumerate(cards[start:start+6]):sheet.paste(card,((k%3)*672,(k//3)*600))
 sheet.save(dest/f'sheet_{start//6:02d}.jpg',quality=92)
with zipfile.ZipFile(A/'pool_review.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in dest.glob('*'):z.write(p,'pool_review/'+p.name)
print('POOL_REVIEW_DONE',len(rows),'ROIs',len(allboxes),'provisional boxes',flush=True)
