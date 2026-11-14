from pathlib import Path
import csv, json, math
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]; A=Path(__file__).resolve().parent
def read(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
t=read(ROOT/'rascar_data_full/test.csv')
base={r['id']:float(r['score']) for r in read(ROOT/'rascar_v52/release/submission_v52_candidate_purged_a20.csv')}
new={r['id']:float(r['score']) for r in read(ROOT/'rascar_v56/release/submission_v56_tabicl_hands_a40.csv')}
t.sort(key=lambda r:(r['frame_id'],r['class_name'],float(r['x1']),float(r['y1']),r['id']))
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',19)
small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',15)
for n,r in enumerate(t,1):
    r.update(review_no=n,score_v52=base[r['id']],score_a40=new[r['id']],delta=new[r['id']]-base[r['id']])
    im=Image.open(ROOT/'rascar_data_full'/f"{r['frame_id']}.jpg").convert('RGB')
    b=[float(r[k]) for k in ['x1','y1','x2','y2']];w=b[2]-b[0];h=b[3]-b[1];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
    ew=max(w*(1.65 if r['class_name']=='person' else 2.5),95);eh=max(h*(1.85 if r['class_name']=='person' else 2.5),85)
    roi=[int(max(0,cx-ew/2)),int(max(0,cy-eh/2)),int(min(1280,cx+ew/2)),int(min(720,cy+eh/2))]
    cr=im.crop(roi);s=min(490/cr.width,320/cr.height);cr=cr.resize((round(cr.width*s),round(cr.height*s)),Image.Resampling.LANCZOS)
    tile=Image.new('RGB',(500,400),'#101825');d=ImageDraw.Draw(tile);ox=(500-cr.width)//2;oy=76+(320-cr.height)//2;tile.paste(cr,(ox,oy))
    d.rectangle(((b[0]-roi[0])*s+ox,(b[1]-roi[1])*s+oy,(b[2]-roi[0])*s+ox,(b[3]-roi[1])*s+oy),outline='#ffdd33',width=2)
    d.text((6,4),f"#{n:03d} {r['class_name']}  V52 {r['score_v52']:.3f}  A40 {r['score_a40']:.3f}",font=font,fill='white')
    d.text((6,29),r['id']+'  '+r['frame_id'][-15:],font=small,fill='#cbd5e1')
    d.text((6,50),f"box {w:.0f} x {h:.0f}   detector confidence {float(r['confidence']):.2f}",font=small,fill='#cbd5e1')
    tile.save(A/'crops'/f'{n:03d}.jpg',quality=94)
for p in range(math.ceil(len(t)/12)):
    sh=Image.new('RGB',(1500,1600),'#101825')
    for k,r in enumerate(t[p*12:(p+1)*12]):sh.paste(Image.open(A/'crops'/f"{r['review_no']:03d}.jpg"),((k%3)*500,(k//3)*400))
    sh.save(A/'pages'/f'test_{p+1:02d}.jpg',quality=94)
(A/'manifest.json').write_text(json.dumps(t,ensure_ascii=False,indent=2))
print('Rendered',len(t),'test boxes',len({r['frame_id'] for r in t}),'frames')
