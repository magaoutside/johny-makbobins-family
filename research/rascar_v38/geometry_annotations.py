"""Freeze cautious TRAIN-only local geometry notes before model comparison.

These are approximate visual boxes with explicit edge uncertainty, not official
full-frame annotations. Do not use them as exact training targets or test labels.
"""
from common38 import *
from PIL import Image,ImageDraw
import cv2

NOTES={
7:([880,240,980,330],[[941,254,965,286],[900,291,920,319]],3),
8:([1000,225,1055,280],[[1012,239,1042,268]],3),
9:([715,290,780,347],[[739,303,768,334]],3),
15:([565,228,610,265],[[578,236,598,259]],3),
17:([585,228,628,260],[[599,235,617,256]],3),
18:([545,276,594,320],[[559,285,581,312]],3),
21:([1145,215,1190,257],[[1156,225,1178,250]],3),
25:([980,245,1060,326],[[996,255,1049,318]],4),
27:([880,235,990,325],[[896,246,976,312]],4),
29:([535,250,620,330],[[550,258,604,322]],4),
31:([545,255,630,330],[[562,266,619,319]],4),
33:([260,240,385,335],[[299,252,367,285],[277,290,345,323]],4),
34:([545,255,630,330],[[558,268,617,317]],4),
38:([545,250,630,318],[[560,259,614,305]],4),
39:([535,252,630,308],[[550,259,620,299]],4),
40:([545,250,630,318],[[560,261,615,305]],4),
42:([500,210,545,252],[[512,221,539,248]],3),
48:([775,180,825,218],[[783,193,810,207]],3),
52:([562,196,600,228],[[574,204,593,225]],3),
62:([530,210,580,275],[[539,221,557,240],[551,245,573,270]],3),
}

def main():
    pool=pd.read_csv(A/'inputs/geometry_review_frames.csv').set_index('review_index');out=[]
    for index,(roi,boxes,tol) in NOTES.items():
        r=pool.loc[index];out.append(dict(review_index=index,id=r.id,frame_id=r.frame_id,day=r.day,class_name=r.class_name,roi=roi,boxes=boxes,edge_uncertainty_px=tol,scope='local reviewed region only',used_for_training=False))
    dest=A/'inputs/train_geometry_notes.json'
    payload=dict(coordinate_system='1280x720; source pool images upscaled from 640x360',source='visual TRAIN-only annotation before V38 model comparison',
      selection='V37 presence-positive pool, biased diagnostic subset; no absence-AP claim',records=out,
      rejected_review_indices=[int(x) for x in pool.index if x not in NOTES],rejection_reason='uncertain instance boundary, occlusion, motion blur, or insufficient coverage for complete local class annotation')
    if dest.exists():assert json.loads(dest.read_text())==payload
    else:dest.write_text(json.dumps(payload,indent=2,ensure_ascii=False))
    for start in range(0,len(out),10):
        batch=out[start:start+10];canvas=Image.new('RGB',(1600,750),'#202020');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(batch):
            roi=np.array(r['roi']);lo=roi[:2]-8;hi=roi[2:]+8
            raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f"{r['frame_id']}.jpg")),(1280,720))[:,:,::-1]
            im=Image.fromarray(raw[lo[1]:hi[1],lo[0]:hi[0]]);scale=min(310/im.width,300/im.height);im=im.resize((int(im.width*scale),int(im.height*scale)))
            x=j%5*320;y=j//5*375;canvas.paste(im,(x,y+50));draw.text((x+3,y+3),f"TRAIN {r['review_index']} {r['class_name']} +/-{r['edge_uncertainty_px']}px",fill='white')
            for b in r['boxes']:draw.rectangle(tuple((np.array(b)-np.r_[lo,lo])*scale+[x,y+50,x,y+50]),outline='#39ff76',width=2)
            draw.rectangle(tuple((roi-np.r_[lo,lo])*scale+[x,y+50,x,y+50]),outline='#ffcc00',width=1)
        canvas.save(A/f'audit/geometry_notes_{start//10+1}.jpg')
    print('ANNOTATED',len(out),'regions',sum(len(r['boxes']) for r in out),'boxes; not used for training')

if __name__=='__main__':main()
