from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v31';sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import q,te,B,CL,iou
from features23 import image
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw

def gallery(numbers,out):
 audit=pd.read_csv(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv');rec=pd.read_csv(A/'inputs/dense_records.csv').set_index('id')
 sheets={};dets={}
 for day in ['25','26']:
  p=np.load(A/f'features/linear_mask_fit{day}_maps.npz');sheets[day]={k:v for k,v in zip(p['ids'],p['maps'])}
  dets[day]=pd.read_csv(A/f'features/conv_detector_fit{day}_boxes.csv')
 canvas=Image.new('RGB',(1200,300*len(numbers)),'#191919');draw=ImageDraw.Draw(canvas)
 for row,n in enumerate(numbers):
  a=audit[audit.review_no.eq(n)].iloc[0];r=rec.loc[a.id];cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int)
  roi=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];base=Image.fromarray(roi);scale=min(390/base.width,245/base.height);size=(int(base.width*scale),int(base.height*scale));base=base.resize(size)
  for col,day in enumerate(['image','25','26']):
   xx=col*400;yy=row*300+45;im=base.copy()
   if day!='image':
    heat=cv2.resize(sheets[day][a.id][CL.index(r.class_name)].astype(np.float32),size);color=cv2.applyColorMap((heat*255).astype(np.uint8),cv2.COLORMAP_JET)[:,:,::-1];im=Image.blend(im,Image.fromarray(color),.4)
   canvas.paste(im,(xx,yy));draw.text((xx+4,row*300+5),f'#{n} {r.class_name} V22 {a.v22_score:.3f} {day}',fill='white');draw.text((xx+4,row*300+22),a.issue_v29[:50],fill='white')
   def box(b,color):
    c=(np.array(b)-np.r_[cb[:2],cb[:2]])*scale+np.array([xx,yy,xx,yy]);draw.rectangle(tuple(c),outline=color,width=2)
   box(r[B].to_numpy(float),'#ff3030')
   if day!='image':
    d=dets[day];z=d[d.record_id.eq(a.id)&d.class_name.eq(r.class_name)&d.confidence.ge(.3)]
    if len(z):
     ov=iou(r[B],z[B]).ravel();best=z.iloc[ov.argmax()];box(best[B].to_numpy(float),'#00ff60');draw.text((xx+5,yy+size[1]+2),f'det IoU {ov.max():.2f}, conf {best.confidence:.2f}',fill='white')
 canvas.save(A/f'audit/{out}.jpg')
if __name__=='__main__':
 gallery([112,185,266,145,204,73],'dense_hard_hands_tray')
 gallery([133,65,67,136,137,157],'dense_hard_person_cups')
