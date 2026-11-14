from common36 import *
from PIL import Image,ImageDraw
import cv2,argparse

def plot(name,split,kind='dense',decoder='vote',confidence=.55):
 changes=pd.read_csv(A/f'audit/{name}_changes.csv');changes=changes[changes.split.eq(split)].sort_values('delta',key=lambda s:abs(s),ascending=False)
 features={fit:pd.read_csv(A/f'features/{kind}_{decoder}_c{confidence}_{split}_fit{fit[-2:]}.csv').set_index('id') for fit in DAYS}
 for off in range(0,len(changes),12):
  rr=changes.iloc[off:off+12];w=500;hh=350
  canvas=Image.new('RGB',(w*3,hh*int(np.ceil(len(rr)/3))),'#202020');dr=ImageDraw.Draw(canvas)
  for k,(_,r) in enumerate(rr.iterrows()):
   b=r[B].to_numpy(float);pad=np.maximum((b[2:]-b[:2])*.6,40);lo=np.maximum(0,np.floor(b[:2]-pad)).astype(int);hi=np.minimum([1280,720],np.ceil(b[2:]+pad)).astype(int)
   raw=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));im=Image.fromarray(raw[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min((w-5)/im.width,(hh-68)/im.height)
   im=im.resize((int(im.width*scale),int(im.height*scale)));x=k%3*w;y=k//3*hh;canvas.paste(im,(x,y+63))
   def box(bb,color,width=2):
    if not np.isfinite(bb).all():return
    c=(np.asarray(bb)-np.r_[lo,lo])*scale+[x,y+63,x,y+63];c[[0,2]]=np.clip(c[[0,2]],x,x+im.width-1);c[[1,3]]=np.clip(c[[1,3]],y+63,y+63+im.height-1)
    if c[2]>c[0] and c[3]>c[1]:dr.rectangle(tuple(c),outline=color,width=width)
   if split=='oof':
    for bb in h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy():
     if iou(b,bb)[0,0]>.01:box(bb,'#66ee55',3)
   for fit in DAYS:
    if r.id in features[fit].index:
     f=features[fit].loc[r.id];box(f[['ref_'+b for b in B]].to_numpy(float),'#ffce44')
   box(b,'#ff4444',3)
   dr.text((x+4,y+3),f'{r.id[:8]} {r.class_name}'+(f' GTerror={int(r.error)}' if split=='oof' else ''),fill='white')
   dr.text((x+4,y+20),f'V31 {r.v31_score:.4f} -> {r.new_score:.4f} ({r.delta:+.4f})',fill='white')
   dr.text((x+4,y+37),'red=query yellow=vote reference'+(' green=official GT' if split=='oof' else ''),fill='white')
  canvas.save(A/f'audit/{name}_{split}_{off//12+1}.jpg')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='dense_vote_c0.55_person_w0.25');ap.add_argument('--kind',default='dense');ap.add_argument('--decoder',default='vote');ap.add_argument('--confidence',type=float,default=.55);args=ap.parse_args()
 for split in ROWS:plot(args.name,split,args.kind,args.decoder,args.confidence)
