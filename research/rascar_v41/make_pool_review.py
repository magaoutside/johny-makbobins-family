"""Render POOL candidates for review. Never produces a test-label worksheet."""
from common import *
from PIL import Image,ImageDraw,ImageFont
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--geometry-review',action='store_true');args=parser.parse_args()
p=A/'pool_expansion';e=pd.read_csv(p/('geometry_evidence.csv' if args.geometry_review else 'evidence.csv'))
if args.geometry_review:
 e=e[e.split.eq('new_pool')].copy();scores=[]
 for _,r in e.iterrows():
  d=r.day[-2:];a,b=r[f'vit_b_{d}_iou'],r[f'vit_l_{d}_iou'];cf=min(r[f'vit_b_{d}_conf'],r[f'vit_l_{d}_conf'])
  scores.append(min(a,b) if min(a,b)>=.55 and max(a,b)>=.7 and cf>=.55 else -1)
 e['review_quality']=scores;e=e[e.review_quality>=0];pieces=[]
 for c,g in e.groupby('class_name'):pieces.append(g.sort_values('review_quality',ascending=False).head(24 if c=='person' else 48))
 g=pd.concat(pieces).sort_values(['class_name','day','review_quality'],ascending=[True,True,False]).reset_index(drop=True)
else:g=e[e.split.eq('new_pool')&e.geometry_admit].copy().reset_index(drop=True)
g['review_index']=np.arange(len(g));g.to_csv(p/'review_candidates.csv',index=False)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
for start in range(0,len(g),12):
 sheet=Image.new('RGB',(1200,1200),'#202020');draw=ImageDraw.Draw(sheet)
 for k,(_,r) in enumerate(g.iloc[start:start+12].iterrows()):
  x=(k%3)*400;y=(k//3)*300;im=Image.open(D/f'{r.frame_id}.jpg').convert('RGB').resize((1280,720));b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;size=np.maximum(wh*2,80);lo=np.maximum(0,cc-size/2);hi=np.minimum([1280,720],cc+size/2)
  crop=im.crop(tuple(np.r_[lo,hi].astype(int)));scale=min(390/crop.width,255/crop.height);crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.BICUBIC);sx=crop.width/(hi[0]-lo[0]);sy=crop.height/(hi[1]-lo[1]);sheet.paste(crop,(x,y+40));bb=(b-np.tile(lo,2))*[sx,sy,sx,sy]+[x,y+40,x,y+40];draw.rectangle(tuple(bb),outline='#ffd34e',width=2)
  label=f"#{int(r.review_index)} {r.class_name} {r.day[-2:]} conf={r.confidence:.2f}"
  draw.text((x+5,y+3),label,font=font,fill='white');draw.text((x+5,y+21),r.frame_id.split('_')[-1],font=font,fill='#aaaaaa')
 sheet.save(p/f'review_{start//12:02d}.jpg',quality=94)
print('REVIEW POOL',len(g),g.class_name.value_counts().to_dict(),flush=True)
