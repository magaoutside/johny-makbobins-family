"""Render query boxes with independent detector references for manual inspection."""
from pathlib import Path
import math, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

A=Path(__file__).resolve().parent
B=['x1','y1','x2','y2']
experiment=sys.argv[1] if len(sys.argv)>1 else 'combined_head_geometry_and_hands'
d=pd.read_csv(A/'results/reviews'/f'{experiment}.csv')
heads=pd.read_csv(A/'wevis_heads_references.csv')
hands=pd.read_csv(A/'wevis_hands_references.csv')
font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',17) if Path('/System/Library/Fonts/Helvetica.ttc').exists() else ImageFont.load_default()
groups={
 'head_cases':[67,101,96,158,244,255,289,76,141,155,149,208],
 'hand_cases':[248,333,266,354,183,195,298,111,190,191,192,239],
 'geometry_regressions':[44,55,62,85,106,124,186,209,229,52,54,87],
 'geometry_regressions_2':[33,34,35,42,84,88,95,108,223,263,271,283],
 'geometry_regressions_3':[310,316,323,329,338,91,92,120,214,7,9,16],
}
if len(sys.argv)>1:
    nums=d.loc[d.delta.abs()>1e-9].sort_values('review_no').review_no.tolist()
    groups={f'final_changes_{i//12+1}':nums[i:i+12] for i in range(0,len(nums),12)}
    ambiguous=d[d.visual_status.eq('uncertain')&(d.delta.abs()>.02)]
    nums=d.loc[d.frame_id.isin(ambiguous.frame_id)&d.class_name.eq('nude_hands')].sort_values('review_no').review_no.tolist()
    groups.update({f'final_ambiguous_groups_{i//12+1}':nums[i:i+12] for i in range(0,len(nums),12)})
for name,nums in groups.items():
    out=Image.new('RGB',(4*400,math.ceil(len(nums)/4)*350),'#f1f4f8')
    for k,no in enumerate(nums):
        r=d[d.review_no.eq(no)].iloc[0];b=r[B].to_numpy(float)
        im=Image.open(A/'test_images'/f'{r.frame_id}.jpg').convert('RGB')
        pad=np.maximum((b[2:]-b[:2])*.5,60);roi=b+np.r_[-pad,pad];roi[[0,2]]=np.clip(roi[[0,2]],0,1280);roi[[1,3]]=np.clip(roi[[1,3]],0,720)
        crop=im.crop(tuple(roi));scale=min(390/crop.width,285/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)))
        draw=ImageDraw.Draw(crop)
        for df,color in [(heads[heads.frame_id.eq(r.frame_id)],'#19aeff'),(hands[hands.query_id.eq(r.id)],'#31d14f')]:
            if color=='#19aeff' and r.class_name!='person':continue
            for ref in df[B].to_numpy():draw.rectangle(tuple((ref-np.tile(roi[:2],2))*scale),outline=color,width=2)
        draw.rectangle(tuple((b-np.tile(roi[:2],2))*scale),outline='red',width=3)
        x=(k%4)*400;y=(k//4)*350;out.paste(crop,(x+(400-crop.width)//2,y+55))
        draw=ImageDraw.Draw(out);draw.text((x+6,y+5),f'#{no} {r.class_name}: {r.score:.3f} -> {r.new_score:.3f}',font=font,fill='black');draw.text((x+6,y+28),str(r.issue)[:38],font=font,fill='black')
    out.save(A/f'{name}.jpg',quality=94)
    print(name)
