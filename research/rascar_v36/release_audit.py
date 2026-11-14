from common36 import *
from PIL import Image,ImageDraw
import cv2

def plot(name,kind,confidence):
 d=pd.read_csv(A/f'audit/{name}_changes.csv');d=d[d.split.eq('test')].sort_values('delta',key=lambda s:abs(s),ascending=False)
 refs=[pd.read_csv(A/f'features/{kind}_vote_c{confidence}_test_fit{fit[-2:]}.csv').set_index('id') for fit in DAYS]
 for off in range(0,len(d),12):
  rows=d.iloc[off:off+12];canvas=Image.new('RGB',(1500,380*int(np.ceil(len(rows)/3))),'#202020');dr=ImageDraw.Draw(canvas)
  for j,(_,r) in enumerate(rows.iterrows()):
   b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;pad=1.8 if r.class_name=='nude_hands' else 1.1;lo=np.maximum(0,np.floor(cc-pad*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+pad*wh)).astype(int)
   im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));roi=Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(495/roi.width,315/roi.height);roi=roi.resize((int(roi.width*scale),int(roi.height*scale)));x=j%3*500;y=j//3*380;canvas.paste(roi,(x,y+63))
   def box(bb,color):
    bb=(bb-np.r_[lo,lo])*scale+[x,y+63,x,y+63];bb[[0,2]]=np.clip(bb[[0,2]],x,x+roi.width-1);bb[[1,3]]=np.clip(bb[[1,3]],y+63,y+63+roi.height-1)
    if bb[2]>bb[0] and bb[3]>bb[1]:dr.rectangle(tuple(bb),outline=color,width=2)
   if r.class_name=='person':
    for f in refs:
     z=f.loc[r.id];bb=z[['ref_'+v for v in B]].to_numpy(float)
     if np.isfinite(bb).all():box(bb,'#ffcc33')
   box(b,'#ff4444');dr.text((x+4,y+4),r.id+' '+r.class_name,fill='white');dr.text((x+4,y+23),f'{r.v31_score:.4f} -> {r.new_score:.4f} ({r.delta:+.4f})',fill='white');dr.text((x+4,y+42),'red=query; yellow=person reference',fill='white')
  canvas.save(A/f'audit/{name}_test_{off//12+1}.jpg')

if __name__=='__main__':
 plot('person_strict_pool','dense_identity',.55)
 plot('person_multiscale_pool','multiscale',.4)
 plot('strict_pose_pool','dense_identity',.55)
 plot('multiscale_pose_pool','multiscale',.4)
