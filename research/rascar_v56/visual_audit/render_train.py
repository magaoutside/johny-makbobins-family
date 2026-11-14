from render import ROOT,A,read,font,small
from PIL import Image,ImageDraw
import json,math
q=read(ROOT/'rascar_best_audit_2026-11-14/verification/v52_error_audit.csv')
hh=read(ROOT/'rascar_data_full/human_boxes_train.csv')
for r in hh:
    r['class_name']=r['class_name'].replace('nude_hand','nude_hands') if r['class_name']=='nude_hand' else r['class_name']
    r['box']=[(float(r['cx'])-float(r['w'])/2)*1280,(float(r['cy'])-float(r['h'])/2)*720,(float(r['cx'])+float(r['w'])/2)*1280,(float(r['cy'])+float(r['h'])/2)*720]
def iou(a,b):
    v=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return v/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-v+1e-9)
for n,r in enumerate(q,1):
    r['train_no']=n;b=[float(r[k]) for k in ['x1','y1','x2','y2']]
    gt=[h['box'] for h in hh if h['frame_id']==r['frame_id'] and h['class_name']==r['class_name']]
    g=max(gt,key=lambda z:iou(b,z)) if gt else None
    if g and iou(b,g)<0.01:g=None
    lo=[min(b[i],g[i] if g else b[i]) for i in [0,1]];hi=[max(b[i],g[i] if g else b[i]) for i in [2,3]]
    w=hi[0]-lo[0];h=hi[1]-lo[1];cx=(lo[0]+hi[0])/2;cy=(lo[1]+hi[1])/2
    ew=max(w*1.5,95);eh=max(h*1.5,85);roi=[int(max(0,cx-ew/2)),int(max(0,cy-eh/2)),int(min(1280,cx+ew/2)),int(min(720,cy+eh/2))]
    im=Image.open(ROOT/'rascar_data_full'/f"{r['frame_id']}.jpg").convert('RGB').crop(roi);s=min(490/im.width,310/im.height);im=im.resize((round(im.width*s),round(im.height*s)),Image.Resampling.LANCZOS)
    tile=Image.new('RGB',(500,400),'#101825');d=ImageDraw.Draw(tile);ox=(500-im.width)//2;oy=86+(310-im.height)//2;tile.paste(im,(ox,oy))
    for box,color in [(b,'#ffdd33')]+([(g,'#36f395')] if g else []):d.rectangle(((box[0]-roi[0])*s+ox,(box[1]-roi[1])*s+oy,(box[2]-roi[0])*s+ox,(box[3]-roi[1])*s+oy),outline=color,width=2)
    d.text((6,3),f"T{n:03d} {r['class_name']} E={r['error']} s={float(r['score']):.3f}",font=font,fill='white')
    d.text((6,27),f"wrong pairs {r['wrong_pairs']}   IoU {float(r['max_gt_iou']):.3f}",font=small,fill='#cbd5e1')
    d.text((6,47),r['id']+' '+r['frame_id'][-15:],font=small,fill='#cbd5e1')
    d.text((6,65),'Yellow=query; green=nearest overlapping official GT',font=small,fill='#cbd5e1')
    tile.save(A/'crops'/f'T{n:03d}.jpg',quality=94)
for p in range(math.ceil(len(q)/12)):
    sh=Image.new('RGB',(1500,1600),'#101825')
    for k,r in enumerate(q[p*12:(p+1)*12]):sh.paste(Image.open(A/'crops'/f"T{r['train_no']:03d}.jpg"),((k%3)*500,(k//3)*400))
    sh.save(A/'pages'/f'train_{p+1:02d}.jpg',quality=94)
(A/'train_manifest.json').write_text(json.dumps(q,ensure_ascii=False,indent=2))
print('Rendered train',len(q))
