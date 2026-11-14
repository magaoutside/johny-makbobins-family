"""Separate joined occupancy regions with independently predicted object centers.

Maps describe box occupancy, not segmentation GT. Watershed is an experimental
instance separation step. No manual test boxes or test-specific exceptions.
"""
from dense_train import *
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

def separate(prob,det,r):
 cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);wh=cb[2:]-cb[:2];G=128;yy,xx=np.mgrid[:G,:G];out=[]
 for k,cls in enumerate(CL):
  p=cv2.resize(prob[k],(G,G),interpolation=cv2.INTER_LINEAR)
  refs=det[det.class_name.eq(cls)&det.confidence.ge(.3)].sort_values('confidence',ascending=False)
  for cut in [.35,.5,.65]:
   foreground=p>=cut;markers=np.zeros((G,G),np.int32);seedrefs={};j=0
   for _,ref in refs.iterrows():
    b=(ref[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])/np.r_[wh,wh]*G;center=(b[:2]+b[2:])/2;size=np.maximum(b[2:]-b[:2],2)
    dist=((xx-center[0])/size[0])**2+((yy-center[1])/size[1])**2
    eligible=foreground&(abs(xx-center[0])<.3*size[0])&(abs(yy-center[1])<.3*size[1])
    if not eligible.any():continue
    val=np.where(eligible,p-.3*dist,-1);y,x=np.unravel_index(val.argmax(),val.shape)
    if markers[y,x]:continue
    j+=1;markers[y,x]=j;seedrefs[j]=ref
   if j==0:continue
   d=distance_transform_edt(foreground);elevation=-p-.1*d/max(float(d.max()),1)
   regions=watershed(elevation,markers,mask=foreground)
   for lab,ref in seedrefs.items():
    ys,xs=np.where(regions==lab)
    if len(xs)<64:continue
    box=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1])/G*np.r_[wh,wh]+np.r_[cb[:2],cb[:2]]
    out.append(dict(class_name=cls,threshold=cut,confidence=float(p[regions==lab].mean()),seed_confidence=float(ref.confidence),seed_mask_iou=float(iou(box,ref[B])[0,0]),seed_count=j,**dict(zip(B,box))))
 return out

def main():
 rec=pd.read_csv(A/'inputs/dense_records.csv');feat=np.load(A/'features/dense_grids.npy',mmap_mode='r')
 for fit in DAYS:
  m=Head('linear_mask');m.load_state_dict(torch.load(A/f'models/linear_mask_fit{fit[-2:]}.pt',weights_only=True));m.eval()
  det=pd.read_csv(A/f'features/conv_detector_fit{fit[-2:]}_boxes.csv');groups={k:v for k,v in det.groupby('record_id')}
  ev=rec[rec.split.eq('test')|rec.day.ne(fit)].index.to_numpy();out=[];start=time.time()
  for off in range(0,len(ev),16):
   ids=ev[off:off+16]
   with torch.inference_mode():p=m(torch.from_numpy(np.array(feat[ids],dtype=np.float32))).sigmoid().numpy()
   for i,z in zip(ids,p):
    r=rec.loc[i];d=groups.get(r.id,det.iloc[:0])
    for box in separate(z,d,r):out.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,**box))
  pd.DataFrame(out).to_csv(A/f'features/watershed_fit{fit[-2:]}_boxes.csv',index=False);print('WATERSHED',fit,len(out),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
