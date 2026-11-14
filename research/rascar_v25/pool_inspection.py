from common25 import *
from PIL import Image,ImageDraw,ImageFont
p=pd.read_csv(V22/'inputs/reviewed_pool_hands.csv');c=pd.read_csv(A/'inputs/reviewed_pool_cups.csv');d=pd.concat([p,c]).reset_index(drop=True)
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
for start in range(0,len(d),8):
 canvas=Image.new('RGB',(1200,600),'white')
 for n,r in enumerate(d.iloc[start:start+8].itertuples()):
  im=Image.open(A/f'gpu/results/pool_images/{r.frame_id}.jpg').convert('RGB').resize((1280,720));b=np.array([getattr(r,x) for x in B]);cc=(b[:2]+b[2:])/2;size=max(120,(b[2:]-b[:2]).max()*3);cr=np.r_[cc-size/2,cc+size/2]
  tile=Image.new('RGB',(300,300),'white');tile.paste(im.crop(cr).resize((300,260)),(0,40));dr=ImageDraw.Draw(tile);dr.text((3,3),f'{start+n} {r.class_name} {r.day[-2:]}',font=font,fill='black');z=(b-np.tile(cr[:2],2))*np.tile([300/size,260/size],2)+[0,40,0,40];dr.rectangle(z.tolist(),outline='yellow',width=2);canvas.paste(tile,((n%4)*300,(n//4)*300))
 canvas.save(A/f'audit/pool_confirm_{start//8}.jpg')
d.to_csv(A/'audit/pool_review_order.csv',index=False)
