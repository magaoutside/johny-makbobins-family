"""Prepare extra, day-separated empty-location crops; reuse frozen V31 grids."""
from pathlib import Path
import sys, os, json, time
ROOT=Path(__file__).resolve().parent.parent; A=ROOT/'rascar_v33'; V31=ROOT/'rascar_v31'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import h,q,B,CL,DAYS
from features23 import image
import numpy as np, pandas as pd, cv2, torch
from transformers import AutoModel
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
for p in ['inputs','features','models','results/oof','results/submissions','audit','logs']:(A/p).mkdir(parents=True,exist_ok=True)

def main():
    rng=np.random.default_rng(3301); rows=[]
    for day in DAYS:
        for fid in sorted(h[h.day.eq(day)].frame_id.unique()):
            present=set(h[h.frame_id.eq(fid)].class_name)
            for cls in CL:
                if cls in present:continue
                templates=h[h.day.eq(day)&h.class_name.eq(cls)]
                for k in range(2):
                    r=templates.iloc[int(rng.integers(len(templates)))]; b=r[B].to_numpy(float)
                    wh=np.maximum(b[2:]-b[:2],16); center=(b[:2]+b[2:])/2
                    lo=np.maximum(0,np.floor(center-1.2*wh)); hi=np.minimum([1280,720],np.ceil(center+1.2*wh))
                    rows.append(dict(id=f'empty_{len(rows)}',frame_id=fid,day=day,class_name=cls,split='empty',origin_id=f'empty_{len(rows)}',template_frame=r.frame_id,**dict(zip(B,b)),**dict(zip(CB,np.r_[lo,hi]))))
    rec=pd.DataFrame(rows);rec['feature_index']=np.arange(len(rec));rec.to_csv(A/'inputs/empty_records.csv',index=False)
    torch.set_num_threads(3);cv2.setNumThreads(1)
    model=AutoModel.from_pretrained(str(V31/'model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    dst=np.lib.format.open_memmap(A/'features/empty_grids.npy',mode='w+',dtype=np.float16,shape=(len(rec),384,32,32))
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);start=time.time()
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            b=r[CB].to_numpy(int);im=image(r.frame_id)[b[1]:b[3],b[0]:b[2],::-1]
            batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
        with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32)
        dst[off:off+len(batch)]=z.numpy().astype(np.float16)
        if off%80==0:print('EXTRACT',off,len(rec),round(time.time()-start,1),flush=True)
    dst.flush()
    (A/'inputs/empty_manifest.json').write_text(json.dumps(dict(n_crops=len(rec),n_frames=int(rec.frame_id.nunique()),by_day_class=rec.groupby(['day','class_name']).size().to_dict().__str__(),rule='Class absent from ALL manual annotations in frame; same-day occupied-location template; absence assumes complete manual class annotation.',test_images_or_labels_used=False,seconds=time.time()-start),indent=2))
    print('DONE',len(rec),round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
