"""Coordinate-labelled native-image crops for a separate, nonofficial audit."""
from pathlib import Path
import json, math, argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

A=Path(__file__).resolve().parent; ROOT=A.parent
q=pd.read_csv(A/'inputs/review_manifest.csv')
parser=argparse.ArgumentParser();parser.add_argument('--numbers');parser.add_argument('--name',default='cups')
args=parser.parse_args()
out=A/'audit/measurement';out.mkdir(exist_ok=True)
font=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',13)
large=ImageFont.truetype('/System/Library/Fonts/Monaco.ttf',18)
changes=pd.read_csv(A/'audit/v27_changed_queries.csv')
nos=list(map(int,args.numbers.split(','))) if args.numbers else q.loc[q.class_name.eq('cup'),'review_no'].tolist()
manifest=[]
tiles=[]
for no in nos:
    r=q.set_index('review_no').loc[no]
    im=Image.open(ROOT/'rascar_v16/test_images'/f'{r.frame_id}.jpg').convert('RGB')
    box=r[['x1','y1','x2','y2']].to_numpy(float)
    side=max(70,math.ceil(max(box[2:]-box[:2])*1.6/10)*10)
    cx,cy=(box[:2]+box[2:])/2
    x0=max(0,min(1280-side,math.floor((cx-side/2)/5)*5))
    y0=max(0,min(720-side,math.floor((cy-side/2)/5)*5))
    s=400/side
    tile=Image.new('RGB',(480,485),'#f4f4f4');d=ImageDraw.Draw(tile)
    tile.paste(im.crop((x0,y0,x0+side,y0+side)).resize((400,400)),(55,50))
    d.text((10,5),f'#{no} {r.class_name} | coordinates',font=large,fill='black')
    step=10 if side<120 else 20
    for x in range(math.ceil(x0/step)*step,x0+side+1,step):
        xx=55+(x-x0)*s
        d.line((xx,42,xx,50),fill='black');d.text((xx-13,27),str(x),font=font,fill='black')
    for y in range(math.ceil(y0/step)*step,y0+side+1,step):
        yy=50+(y-y0)*s
        d.line((47,yy,55,yy),fill='black');d.text((3,yy-7),str(y),font=font,fill='black')
    d.rectangle((55+(box[0]-x0)*s,50+(box[1]-y0)*s,55+(box[2]-x0)*s,50+(box[3]-y0)*s),outline='#ff4545',width=1)
    d.text((8,457),f'Q: '+','.join(f'{v:.1f}' for v in box),font=font,fill='black')
    tile.save(out/f'{no:03d}.png')
    tiles.append(tile);manifest.append(dict(review_no=no,x0=x0,y0=y0,scale=s))
for i in range(0,len(tiles),4):
    page=Image.new('RGB',(960,970),'white')
    for k,t in enumerate(tiles[i:i+4]):page.paste(t,((k%2)*480,(k//2)*485))
    page.save(out/f'{args.name}_{i//4:02d}.jpg',quality=95)
(out/f'{args.name}_transforms.json').write_text(json.dumps(manifest,indent=2))
print('Rendered',len(nos),'queries on',math.ceil(len(nos)/4),'pages')
