"""Diagnostic overlays only; rendered pictures are never training inputs."""
from common23 import *
from PIL import Image,ImageDraw,ImageFont
import math
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)

def render(rows,name,score_col='new_score',gt=False,ref=None):
    rows=list(rows.itertuples());paths=[]
    for start in range(0,len(rows),6):
        page=Image.new('RGB',(1200,760),'#eeeeee')
        for j,r in enumerate(rows[start:start+6]):
            b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2
            size=max(140,wh.max()*1.6);cr=np.r_[cc-size/2,cc+size/2];cr[0]=np.clip(cr[0],0,max(0,1280-size));cr[1]=np.clip(cr[1],0,max(0,720-size));cr[2:]=cr[:2]+size
            folder='images' if gt else 'test_images';im=Image.open(ROOT/f'rascar_v16/{folder}/{r.frame_id}.jpg').convert('RGB')
            pic=Image.new('RGB',(400,380),'white');pic.paste(im.crop(tuple(cr)).resize((390,315)),(5,60));dr=ImageDraw.Draw(pic)
            label=f'#{int(r.review_no)}' if hasattr(r,'review_no') else r.id[:8]
            old=getattr(r,'baseline_v22',getattr(r,'score',0));new=getattr(r,score_col,old)
            dr.text((5,3),f'{label} {r.class_name}: {old:.3f} -> {new:.3f}',font=font,fill='black')
            status=f'GT error={r.error}' if gt else r.visual_status+' / '+r.issue
            dr.text((5,27),status,font=font,fill='black')
            def box(bb,color,width=2):
                xy=(np.asarray(bb)-np.tile(cr[:2],2))*np.tile([390/size,315/size],2)+[5,60,5,60]
                dr.rectangle(tuple(xy),outline=color,width=width)
            if gt:
                for bb in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy():box(bb,'lime')
            if ref is not None and r.id in ref.index:
                z=ref.loc[r.id]
                if all('ref_'+c in z.index for c in B):box(z[['ref_'+c for c in B]].to_numpy(float),'cyan')
                elif 'ref_y2' in z.index:box([b[0],z.ref_y1,b[2],z.ref_y2],'cyan')
            box(b,'yellow',3)
            page.paste(pic,((j%3)*400,(j//3)*380))
        path=A/f'audit/{name}_{start//6:02}.jpg';page.save(path,quality=92);paths.append(str(path))
    return paths

if __name__=='__main__':
    f=align(A/'features/table_edge_far_0.25_oof.csv',q);d=q.copy();d['baseline_v22']=base_oof;d['new_score']=f.score.to_numpy()
    bad=d[d.error.eq(0)&d.new_score.gt(d.baseline_v22+.01)]
    print(render(bad,'table_correct_oof',gt=True,ref=f))
    for name in ['table_edge_far_w0.25','combined_catboost_cup_rescue_w0.4','rule_hand_distal_shape_w0.4']:
        d=pd.read_csv(A/f'results/reviews/{name}.csv');d=d[d.delta.abs()>.02].sort_values('review_no')
        print(render(d,name))
