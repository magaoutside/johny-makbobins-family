"""Prepare TRAIN-only annotation panels, without model scores/boxes overlaid."""
from common38 import *
from PIL import Image,ImageDraw
import cv2

def main():
    pool=pd.read_csv(ROOT/'rascar_v37/inputs/blind_pool.csv');lab=json.loads((ROOT/'rascar_v37/inputs/blind_review.json').read_text())
    pool=pool[pool.review_index.isin(lab['present'])].copy()
    assert not set(pool.frame_id)&set(te.frame_id)
    pool.to_csv(A/'inputs/geometry_review_frames.csv',index=False)
    for _,r in pool.iterrows():
        raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))[:,:,::-1]
        im=Image.fromarray(raw);draw=ImageDraw.Draw(im)
        for x in range(0,1280,100):draw.line((x,0,x,719),fill=(130,130,130),width=1);draw.text((x+2,2),str(x),fill='white')
        for y in range(0,720,100):draw.line((0,y,1279,y),fill=(130,130,130),width=1);draw.text((2,y+2),str(y),fill='white')
        canvas=Image.new('RGB',(1280,1160),'#202020');canvas.paste(im,(0,35));dr=ImageDraw.Draw(canvas)
        dr.text((5,4),f'TRAIN review {r.review_index}  {r.class_name}  {r.frame_id}',fill='white')
        b=r[B].to_numpy(float);cc=(b[:2]+b[2:])/2;size=np.maximum((b[2:]-b[:2])*3,[100,80]);lo=np.maximum(0,np.floor(cc-size/2)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+size/2)).astype(int)
        crop=Image.fromarray(raw[lo[1]:hi[1],lo[0]:hi[0]]);scale=min(1200/crop.width,380/crop.height);crop=crop.resize((int(crop.width*scale),int(crop.height*scale)));canvas.paste(crop,(40,780))
        for x in range(((lo[0]+9)//10)*10,hi[0],10):
            cx=40+(x-lo[0])*scale;dr.line((cx,780,cx,780+crop.height-1),fill='#777777');dr.text((cx+1,763),str(x),fill='white')
        for y in range(((lo[1]+9)//10)*10,hi[1],10):
            cy=780+(y-lo[1])*scale;dr.line((40,cy,40+crop.width-1,cy),fill='#777777');dr.text((4,cy),str(y),fill='white')
        canvas.save(A/f'audit/train_geometry_{int(r.review_index):02d}.jpg')

if __name__=='__main__':main()
