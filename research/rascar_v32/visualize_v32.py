"""Visual QA: original queries, independent hand masks and conditional masks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';V31=ROOT/'rascar_v31'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import B,iou
from features23 import image
import numpy as np,pandas as pd,cv2
from PIL import Image,ImageDraw

def main():
    audit=pd.read_csv(ROOT/'rascar_v29/audit/all_362_fresh_audit.csv');rec=pd.read_csv(V31/'inputs/dense_records.csv').set_index('id')
    masks={};boxes={}
    for fit in ['25','26']:
        ar=np.load(A/f'features/conditional_projection_fit{fit}_maps.npz');masks[fit]=dict(zip(ar['ids'],ar['maps']))
        boxes[fit]=pd.read_csv(A/f'features/conditional_projection_fit{fit}_boxes.csv')
    for numbers,name in [([112,185,266,145,204,222],'hands'),([133,65,67,136,157,73],'people_cups_tray')]:
        canvas=Image.new('RGB',(1200,300*len(numbers)),'#171717');draw=ImageDraw.Draw(canvas)
        for row,number in enumerate(numbers):
            ar=audit[audit.review_no.eq(number)].iloc[0];r=rec.loc[ar.id];cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int)
            roi=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];scale=min(390/roi.shape[1],235/roi.shape[0]);size=(int(roi.shape[1]*scale),int(roi.shape[0]*scale));base=Image.fromarray(roi).resize(size)
            for col,fit in enumerate(['image','25','26']):
                x=400*col;y=300*row+45;im=base.copy()
                if fit!='image':
                    p=cv2.resize(masks[fit][ar.id].astype(np.float32),size);color=cv2.applyColorMap((p*255).astype(np.uint8),cv2.COLORMAP_JET)[:,:,::-1];im=Image.blend(im,Image.fromarray(color),.35)
                canvas.paste(im,(x,y));draw.text((x+4,y-40),f'#{number} {r.class_name}, fit={fit}',fill='white');draw.text((x+4,y-23),f'V22 score={ar.v22_score:.3f}; RED=query GREEN=reference',fill='white')
                def box(b,color):
                    coords=(np.array(b)-np.r_[cb[:2],cb[:2]])*scale+np.array([x,y,x,y]);draw.rectangle(tuple(coords),outline=color,width=2)
                box(r[B].to_numpy(float),'#ff3737')
                if fit!='image':
                    z=boxes[fit];z=z[z.record_id.eq(ar.id)&z.threshold.eq(.5)]
                    if len(z):
                        ov=iou(r[B],z[B]).ravel();best=z.iloc[ov.argmax()];box(best[B],'#20ff70');draw.text((x+4,y+size[1]+4),f'IoU={ov.max():.3f}, mask confidence={best.confidence:.3f}',fill='white')
        canvas.save(A/f'audit/conditional_{name}.jpg')
    # Direct independent-mask output: close and wide views of the same query.
    canvas=Image.new('RGB',(1200,300*4),'#171717');draw=ImageDraw.Draw(canvas)
    for col,(tag,ctx) in enumerate([('handseg',2.4),('handseg',4.),('handseg_wide',8.)]):
        rows=pd.read_csv(A/f'inputs/{tag}_records.csv');maps=np.load(A/f'features/{tag}_maps.npy',mmap_mode='r')
        for j,number in enumerate([112,185,266,204]):
            ar=audit[audit.review_no.eq(number)].iloc[0];rr=rows[rows.id.eq(ar.id)&rows.context.eq(ctx)];r=rr.iloc[0];p=maps[rr.index[0]].astype(np.float32)
            cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);roi=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];scale=min(390/roi.shape[1],240/roi.shape[0]);size=(int(roi.shape[1]*scale),int(roi.shape[0]*scale));im=Image.fromarray(roi).resize(size)
            prob=cv2.resize(p,size);over=np.asarray(im).copy();over[prob>.5]=(over[prob>.5]*.5+np.array([0,255,0])*.5).astype(np.uint8)
            x=col*400;y=j*300+40;canvas.paste(Image.fromarray(over),(x,y));b=(r[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])*scale+np.array([x,y,x,y]);draw.rectangle(tuple(b),outline='#ff3737',width=2);draw.text((x+4,y-30),f'#{number} DeepLab context={ctx}; green=mask>0.5',fill='white')
    canvas.save(A/'audit/independent_handseg_failures.jpg')
if __name__=='__main__':main()
