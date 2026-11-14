from common36 import *
from PIL import Image,ImageDraw
import cv2

def main():
 dd=pd.read_csv(A/'audit/pose_guard_decisions.csv');dd=dd[dd.veto].drop_duplicates('id');w=500;hh=380;canvas=Image.new('RGB',(w*3,hh*int(np.ceil(len(dd)/3))),'#202020');dr=ImageDraw.Draw(canvas)
 for j,(_,z) in enumerate(dd.iterrows()):
  r=ROWS[z.split].set_index('id').loc[z.id];b=r[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;lo=np.maximum(0,np.floor(cc-1.1*wh)).astype(int);hi=np.minimum([1280,720],np.ceil(cc+1.1*wh)).astype(int)
  im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));roi=Image.fromarray(im[lo[1]:hi[1],lo[0]:hi[0],::-1]);scale=min(495/roi.width,310/roi.height);roi=roi.resize((int(roi.width*scale),int(roi.height*scale)));x=j%3*w;y=j//3*hh;canvas.paste(roi,(x,y+65));box=(b-np.r_[lo,lo])*scale+[x,y+65,x,y+65];dr.rectangle(tuple(box),outline='#ff5555',width=2)
  for pt in json.loads(z.supported_lower_body_points):
   px,py=(np.array(pt)-lo)*scale+[x,y+65];dr.ellipse((px-3,py-3,px+3,py+3),fill='#55ffff')
  dr.text((x+4,y+3),z.id+' '+z.split,fill='white');dr.text((x+4,y+22),f'New penalty {z.before_score:.3f} vetoed; V31 {z.v31_score:.3f}',fill='white');dr.text((x+4,y+41),'cyan=consistent knee/ankle'+(f'; official error={int(r.error)}' if z.split=='oof' else ''),fill='white')
 canvas.save(A/'audit/pose_guard_vetoes.jpg')

if __name__=='__main__':main()
