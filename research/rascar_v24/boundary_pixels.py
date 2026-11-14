"""Image-edge evidence, with dense pixel shifts around annotated boundaries.

No image identifiers, absolute location, fitted day statistics, or test labels
are model features. The entire held day is absent from model fitting.
"""
from common24 import *
from features23 import image
import cv2
from skimage.feature import hog
from functools import lru_cache
cv2.setNumThreads(2)

def descriptor(fid,b):
    b=np.asarray(b,float);wh=b[2:]-b[:2]
    xs=np.linspace(b[0]-.5*wh[0],b[2]+.5*wh[0],96,endpoint=False)+wh[0]/96
    ys=np.linspace(b[1]-.5*wh[1],b[3]+.5*wh[1],96,endpoint=False)+wh[1]/96
    xx,yy=np.meshgrid(xs,ys)
    im=cv2.remap(image(fid),xx.astype('float32'),yy.astype('float32'),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype('float32')/255
    smooth=cv2.GaussianBlur(gray,(3,3),.6)
    gx=cv2.Scharr(smooth,cv2.CV_32F,1,0)/32;gy=cv2.Scharr(smooth,cv2.CV_32F,0,1)/32
    mag=np.hypot(gx,gy);norm=max(float(np.quantile(mag,.8)),.015)
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB).astype(float)/255
    f=dict(log_width=np.log(wh[0]),log_height=np.log(wh[1]),aspect=wh[0]/wh[1],
           contrast=float(gray.std()),blur=float(cv2.Laplacian(smooth,cv2.CV_32F).var()),color=float(lab[:,:,1:].std(axis=(0,1)).mean()))
    for side,axis,edge,direction in [('left',0,24,1),('right',0,72,-1),('top',1,24,1),('bottom',1,72,-1)]:
        signed=gx if axis==0 else gy
        profiles=[]
        for k,delta in enumerate(range(-12,13,2)):
            at=edge+direction*delta
            strip=signed[27:69,at-1:at+2] if axis==0 else signed[at-1:at+2,27:69]
            v=float(abs(strip).mean()/norm);profiles.append(v)
            f[f'edge_{side}_{k}']=v
        profiles=np.array(profiles)
        f[f'peak_{side}']=(profiles.argmax()-6)/6
        f[f'center_prominence_{side}']=float(profiles[5:8].max()/(profiles.max()+.01))
        for key,arr in [('gray',gray),('chroma',lab[:,:,1:])]:
            a=edge-direction*4;c=edge+direction*4
            outside=arr[27:69,a-2:a+2] if axis==0 else arr[a-2:a+2,27:69]
            inside=arr[27:69,c-2:c+2] if axis==0 else arr[c-2:c+2,27:69]
            f[f'jump_{key}_{side}']=float(np.mean(abs(inside.mean(axis=(0,1))-outside.mean(axis=(0,1)))))
    hv=hog(gray,orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),feature_vector=True)
    f.update({f'hog_{i}':float(v) for i,v in enumerate(hv)})
    return f

def training():
    rng=np.random.default_rng(2401);out=[]
    def add(fid,day,cls,b,gt,kind,weight=1.,error=None,ident=None):
        b=np.clip(b,0,[1280,720,1280,720])
        if min(b[2:]-b[:2])<5:return
        target=float(iou(b,gt).max()) if len(gt) else 0.
        out.append(dict(id=ident or f'boundary24_{len(out)}',frame_id=fid,day=day,class_name=cls,
            error=int(target<.6) if error is None else error,target_iou=target,kind=kind,weight=weight,**dict(zip(B,b))))
    for (day,fid,cls),g in h[h.class_name.isin(CL)].groupby(['day','frame_id','class_name']):
        gt=g[B].to_numpy()
        for b in gt:
            wh=b[2:]-b[:2];add(fid,day,cls,b,gt,'gt',2.)
            for side in range(4):
                for size in [-8,-4,4,8] if cls!='person' else [-.25,-.1,.1,.25]:
                    bb=b.copy();bb[side]+=size if cls!='person' else size*wh[side%2]
                    add(fid,day,cls,bb,gt,'pixel_border')
            for _ in range(16):
                add(fid,day,cls,b+rng.uniform(-.4,.4,4)*np.tile(wh,2),gt,'random_border')
            for peer in gt:
                if np.array_equal(peer,b):continue
                if np.linalg.norm(((peer[:2]+peer[2:])-b[:2]-b[2:])/2/wh)<3:
                    add(fid,day,cls,np.r_[np.minimum(b[:2],peer[:2]),np.maximum(b[2:],peer[2:])],gt,'merge',1.5)
        other=h[h.frame_id.eq(fid)&h.class_name.ne(cls)&h.class_name.isin(CL+['phone','empty_plate','full_plate'])]
        for _,r in other.sample(min(8,len(other)),random_state=2401).iterrows():add(fid,day,cls,r[B].to_numpy(float),gt,'wrong_class')
    for r in q.itertuples():
        gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy()
        add(r.frame_id,r.day,r.class_name,np.array([getattr(r,c) for c in B]),gt,'real_query',4.,r.error,r.id)
    return pd.DataFrame(out)

if __name__=='__main__':
    tr=training();tr.to_csv(A/'inputs/boundary_training.csv',index=False)
    for split,rows in [('train',tr),('oof',q),('test',te)]:
        path=A/f'features/boundary_{split}.csv'
        if path.exists():continue
        out=[]
        for j,r in enumerate(rows.itertuples()):
            out.append(dict(id=r.id,**descriptor(r.frame_id,[getattr(r,c) for c in B])))
            if j%1000==0:print('BOUNDARY',split,j,len(rows),flush=True)
        d=pd.DataFrame(out);assert np.isfinite(d.drop(columns='id')).all().all();d.to_csv(path,index=False)
    print('DONE',tr.groupby(['day','class_name']).size().to_dict(),flush=True)
