from common36 import *
from PIL import Image,ImageDraw
import cv2

def main():
 name='siglip_hand_c0.1_p0.15_w0.25';d=pd.read_csv(A/f'audit/{name}_changes.csv');ev=pd.read_csv(A/'audit/siglip_predictions.csv');ev=ev[ev.C.eq(.1)]
 for split in ['oof','test']:
  rows=d[d.split.eq(split)].sort_values('delta',ascending=False)
  for off in range(0,len(rows),12):
   rr=rows.iloc[off:off+12];canvas=Image.new('RGB',(1500,380*int(np.ceil(len(rr)/4))),'#202020');dr=ImageDraw.Draw(canvas)
   for j,(_,r) in enumerate(rr.iterrows()):
    b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2;lo=np.maximum(0,np.floor(cc-1.8*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+1.8*wh)).astype(int)
    im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));roi=Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(370/roi.width,310/roi.height);roi=roi.resize((int(roi.width*scale),int(roi.height*scale)));x=j%4*375;y=j//4*380;canvas.paste(roi,(x,y+67));box=(b-np.r_[lo,lo])*scale+[x,y+67,x,y+67];dr.rectangle(tuple(box),outline='#ff4444',width=2)
    dr.text((x+4,y+3),r.id[:10]+(f' official error={int(r.error)}' if split=='oof' else ''),fill='white');dr.text((x+4,y+21),f'{r.v31_score:.4f} -> {r.new_score:.4f}',fill='white')
    probs=ev[ev.id.eq(r.id)].p_hand.tolist();dr.text((x+4,y+40),'P(hand): '+', '.join(f'{p:.3f}' for p in probs),fill='white')
   canvas.save(A/f'audit/{name}_{split}_{off//12+1}.jpg')

if __name__=='__main__':main()
