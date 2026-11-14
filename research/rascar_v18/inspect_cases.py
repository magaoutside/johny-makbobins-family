from pathlib import Path
import sys,math
import pandas as pd,numpy as np
from PIL import Image,ImageDraw,ImageFont
A=Path(__file__).resolve().parent;B=['x1','y1','x2','y2'];name=sys.argv[1];numbers=[int(x) for x in sys.argv[2].split(',')];label=sys.argv[3] if len(sys.argv)>3 else name
d=pd.read_csv(A/'results/reviews'/f'{name}.csv');heads=pd.read_csv(A.parent/'rascar_v16/wevis_heads_references.csv');hands=pd.read_csv(A/'frozen_hand_references.csv');body=pd.read_csv(A/'person_body_ridge_test.csv').set_index('id')
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',17);out=Image.new('RGB',(480*4,400*math.ceil(len(numbers)/4)),'#eef2f6')
for k,n in enumerate(numbers):
    r=d[d.review_no.eq(n)].iloc[0];b=r[B].to_numpy(float);im=Image.open(A.parent/'rascar_v16/test_images'/f'{r.frame_id}.jpg').convert('RGB');pad=np.maximum((b[2:]-b[:2])*.65,70);roi=b+np.r_[-pad,pad];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
    crop=im.crop(tuple(roi));scale=min(470/crop.width,330/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)));draw=ImageDraw.Draw(crop)
    if r.class_name=='person':
        rr=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy();color='#26aaff';ref=body.loc[r.id,['ref_'+c for c in B]].to_numpy(float)
        if body.loc[r.id,'eligible']:draw.rectangle(tuple((ref-np.tile(roi[:2],2))*scale),outline='#dc61ff',width=2)
    else:rr=hands[hands.frame_id.eq(r.frame_id)][B].to_numpy();color='#30d24f'
    for bb in rr:draw.rectangle(tuple((bb-np.tile(roi[:2],2))*scale),outline=color,width=2)
    draw.rectangle(tuple((b-np.tile(roi[:2],2))*scale),outline='#ff2828',width=3);xx=(k%4)*480;yy=(k//4)*400;out.paste(crop,(xx+(480-crop.width)//2,yy+60));draw=ImageDraw.Draw(out)
    draw.text((xx+8,yy+5),f'#{n} {r.class_name} {r.score:.3f} -> {r.new_score:.3f}',font=font,fill='black');draw.text((xx+8,yy+30),str(r.issue)[:48],font=font,fill='black')
out.save(A/f'{label}.jpg',quality=94);print(A/f'{label}.jpg')
