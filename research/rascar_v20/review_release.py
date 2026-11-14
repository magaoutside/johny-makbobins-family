"""Render all changes, frozen priority cases, and protected controls for review."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd, math
A=Path(__file__).resolve().parent
r=pd.read_csv(A/'results/reviews/release_main.csv')
e=pd.read_csv(A/'release_main_evidence_test.csv').set_index('id')
ref=pd.read_csv(A/'hand_expanded_sahi_test.csv').set_index('id')
priority=r.diagnostic_flag.isin(['clear_error_low_score','likely_error_low_score','plausible_correct_high_score'])
selected=r[(r.delta.abs()>1e-9)|priority|r.review_no.isin([66,234,272,139,190,192,193,32])].sort_values('review_no')
dest=A/'audit/final_review';dest.mkdir(exist_ok=True)
selected.to_csv(dest/'reviewed_cases.csv',index=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
for page,start in enumerate(range(0,len(selected),12)):
    canvas=Image.new('RGB',(1200,1140),'white');draw=ImageDraw.Draw(canvas)
    for k,(_,row) in enumerate(selected.iloc[start:start+12].iterrows()):
        x=(k%4)*300;y=(k//4)*380
        im=Image.open(A.parent/'rascar_v16/test_images'/f'{row.frame_id}.jpg').convert('RGB').resize((1280,720))
        b=[float(row[c]) for c in ['x1','y1','x2','y2']]
        cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
        size=max(b[2]-b[0],b[3]-b[1])*1.6
        size=max(size,125 if row.class_name=='nude_hands' else 180)
        l=max(0,min(1280-size,cx-size/2));t=max(0,min(720-size,cy-size/2))
        crop=im.crop((l,t,l+size,t+size)).resize((296,296))
        cd=ImageDraw.Draw(crop);scale=296/size
        cd.rectangle(tuple((v-(l if j%2==0 else t))*scale for j,v in enumerate(b)),outline='yellow',width=2)
        if row.class_name=='nude_hands' and e.loc[row.id,'sahi_hand_rescue']:
            rr=ref.loc[row.id];bb=[rr['ref_'+c] for c in ['x1','y1','x2','y2']]
            cd.rectangle(tuple((v-(l if j%2==0 else t))*scale for j,v in enumerate(bb)),outline='cyan',width=2)
        canvas.paste(crop,(x,y+78))
        draw.text((x+3,y+3),f'#{row.review_no} {row.class_name}',font=font,fill='black')
        draw.text((x+3,y+23),f'{row.baseline_v18:.3f} -> {row.new_score:.3f}',font=font,fill='black')
        draw.text((x+3,y+43),str(row.visual_status),font=font,fill='black')
        draw.text((x+3,y+60),str(row.issue)[:33],font=font,fill='black')
    canvas.save(dest/f'page_{page+1}.jpg',quality=92)
print(f'{len(selected)} cases, {math.ceil(len(selected)/12)} sheets')
