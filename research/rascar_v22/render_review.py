from common22 import *
from PIL import Image,ImageDraw,ImageFont
import math
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
d=pd.read_csv(R/'reviews/final_union.csv');d=d[abs(d.delta)>.02].sort_values(['class_name','review_no'])
ref=pd.read_csv(A/'zoom_targeted_test.csv').set_index('id');out=A/'audit/final_visual';out.mkdir(exist_ok=True)
def card(r):
 im=Image.open(ROOT/'rascar_v16/test_images'/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720))
 b=np.array([r.x1,r.y1,r.x2,r.y2]);wh=b[2:]-b[:2];size=max(100,float(wh.max()*2.5));cc=(b[:2]+b[2:])/2;l=max(0,min(1280-size,cc[0]-size/2));t=max(0,min(720-size,cc[1]-size/2));cr=np.array([l,t,l+size,t+size])
 im=im.crop(tuple(cr)).resize((380,300));pic=Image.new('RGB',(400,360),'white');pic.paste(im,(10,55));draw=ImageDraw.Draw(pic)
 draw.text((10,4),f'#{int(r.review_no)} {r.class_name} {r.baseline_v20:.3f} -> {r.new_score:.3f}',font=font,fill='black');draw.text((10,26),r.issue+' / '+r.visual_status,font=font,fill='black')
 def box(bb,color):
  xy=(np.asarray(bb)-np.tile(cr[:2],2))*np.tile([380/size,300/size],2)+[10,55,10,55];draw.rectangle(tuple(xy),outline=color,width=2)
 box(b,'yellow')
 if r.class_name=='nude_hands':
  rr=ref.loc[r.id]
  if rr.overlap>.1:box(rr[['ref_'+c for c in B]].to_numpy(float),'cyan')
 return pic
rr=list(d.itertuples())
for start in range(0,len(rr),6):
 canvas=Image.new('RGB',(1200,720),'#eeeeee')
 for j,r in enumerate(rr[start:start+6]):canvas.paste(card(r),((j%3)*400,(j//3)*360))
 canvas.save(out/f'changed_{start//6:02d}.jpg',quality=93)
d.to_csv(out/'reviewed_changes.csv',index=False);print('significant changed rows',len(d),'sheets',math.ceil(len(d)/6))
# Also inspect the people harmed by the strongest joint expert.
z=pd.read_csv(R/'reviews/conditioned_cb4_guarded_w0.15.csv');z=z[z.visual_status.isin(['clear_error','likely_error'])&z.delta.lt(-.02)].sort_values('delta').head(6)
canvas=Image.new('RGB',(1200,720),'#eeeeee')
for j,r in enumerate(z.itertuples()):canvas.paste(card(r),((j%3)*400,(j//3)*360))
canvas.save(out/'rejected_person_examples.jpg',quality=93)
