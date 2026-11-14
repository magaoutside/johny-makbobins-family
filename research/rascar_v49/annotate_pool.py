"""Saved visual silver review; bounds are approximate, not organizer GT."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from PIL import Image,ImageDraw,ImageFont
A=Path(__file__).resolve().parent;D=A.parent/'rascar_data_full';B=['x1','y1','x2','y2'];CB=['crop_x1','crop_y1','crop_x2','crop_y2']
# Coordinates relative to the crop origin printed on reviewed sheets.
BOXES={0:[41,35,106,79],1:[58,45,132,108],2:[47,33,104,94],3:[45,38,100,73],4:[14,28,86,79],5:[40,30,119,81],7:[37,40,100,77],8:[32,29,83,79],9:[57,25,101,74],10:[35,35,100,74],11:[48,36,119,92],12:[40,40,100,73],13:[41,28,105,77],15:[42,42,98,70],16:[47,38,108,72],17:[39,32,92,77],18:[42,40,100,82],20:[34,36,107,78],26:[42,31,119,84],28:[21,34,80,76]}
SKIP={6:'hand/object occlusion; tray edge uncertain',14:'hands cover border; incomplete geometry',19:'overexposed boundary',21:'hand occlusion and ambiguous outer border',22:'cannot confidently separate tray from table',23:'occlusion and crowded table',24:'possible neighboring partial tray',25:'two trays and crop truncation',27:'two overlapping trays with partly hidden boundary',29:'unclear instance and boundary'}
# Overlay QA rejected these estimates before any training or model selection.
for i,why in [(9,'oblique plate/tray boundary remains ambiguous after overlay QA'),(17,'hand obscures left tray boundary after overlay QA')]:
    BOXES.pop(i);SKIP[i]=why
def main():
    r=pd.read_csv(A/'pool_review/candidates.csv');rows=[];boxes=[]
    for _,x in r.iterrows():
        i=int(x.review_index);d=x.to_dict();d['status']='accepted_silver' if i in BOXES else 'excluded_ambiguous';d['notes']='Visible tray; approximate outer bounding rectangle; ignore uncertain border in loss' if i in BOXES else SKIP[i];d['edge_uncertainty_px']=6 if i in [1,2,4,9,11,16,20,28] else 4;rows.append(d)
        if i in BOXES:
            b=np.array(BOXES[i])+np.tile(x[CB[:2]].to_numpy(float),2);boxes.append(dict(review_index=i,frame_id=x.frame_id,day=x.day,role=x.role,class_name='tray',edge_uncertainty_px=d['edge_uncertainty_px'],**dict(zip(B,b))))
    pd.DataFrame(rows).to_csv(A/'pool_review/reviews.csv',index=False);pd.DataFrame(boxes).to_csv(A/'pool_review/boxes.csv',index=False)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    for off in range(0,len(boxes),6):
        sheet=Image.new('RGB',(1200,900),'#151a23');draw=ImageDraw.Draw(sheet)
        for k,b in enumerate(boxes[off:off+6]):
            x=r[r.review_index.eq(b['review_index'])].iloc[0];cb=x[CB].to_numpy(int);im=Image.open(D/f'{x.frame_id}.jpg').convert('RGB').resize((1280,720)).crop(tuple(cb));s=min(380/im.width,380/im.height);im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.BICUBIC);xx=k%3*400+10;yy=k//3*450+55;sheet.paste(im,(xx,yy))
            for bb,color in [(x[B].to_numpy(float),'#ffda55'),(np.array([b[v] for v in B]),'#66ef99')]:draw.rectangle(((bb-np.tile(cb[:2],2))*s+np.tile([xx,yy],2)).tolist(),outline=color,width=2)
            draw.text((xx,yy-48),f'#{b["review_index"]} {b["role"]}',font=font,fill='white');draw.text((xx,yy-26),f'silver +/-{b["edge_uncertainty_px"]} coord px',font=font,fill='white')
        sheet.save(A/f'pool_review/annotated_{off//6:02d}.jpg',quality=95)
    summary=dict(reviewed=len(r),accepted=len(boxes),accepted_by_role=pd.DataFrame(boxes).role.value_counts().to_dict(),excluded=len(SKIP),test_annotation=False,training='development only, own fit day; uncertain edges ignored',warning='Silver visual estimates at native 640x360, not exact manual GT; holdout contains only six frames and may lack robust error examples.')
    (A/'pool_review/review_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
