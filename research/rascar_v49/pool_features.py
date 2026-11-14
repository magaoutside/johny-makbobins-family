from pathlib import Path
import os,json,time
A=Path(__file__).resolve().parent;ROOT=A.parent
os.environ.setdefault('MPLCONFIGDIR',str(A/'features/mpl'))
import numpy as np,pandas as pd,torch,cv2
from transformers import AutoModel,SiglipVisionModel
def main():
    assert torch.backends.mps.is_available();torch.set_num_threads(3);cv2.setNumThreads(1);rec=pd.read_csv(A/'pool_review/reviews.csv');rec=rec[rec.status.eq('accepted_silver')].reset_index(drop=True);rec.to_csv(A/'inputs/pool_records.csv',index=False)
    for key in ['small','siglip2']:
        path=ROOT/('rascar_v31/model_cache/dinov2-small' if key=='small' else 'rascar_v36/model_cache/siglip2-base-patch16-224');model=(AutoModel if key=='small' else SiglipVisionModel).from_pretrained(str(path),local_files_only=True).eval().to('mps');dim=384 if key=='small' else 768;size=448 if key=='small' else 512
        mean=torch.tensor([.485,.456,.406] if key=='small' else [.5]*3,device='mps').view(1,3,1,1);std=torch.tensor([.229,.224,.225] if key=='small' else [.5]*3,device='mps').view(1,3,1,1);out=[]
        for off in range(0,len(rec),4):
            images=[]
            for _,r in rec.iloc[off:off+4].iterrows():
                im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720));b=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);im=im[b[1]:b[3],b[0]:b[2],::-1];images.append(cv2.resize(im,(size,size),interpolation=cv2.INTER_CUBIC if key=='small' else cv2.INTER_LINEAR).transpose(2,0,1).copy())
            x=(torch.tensor(np.stack(images),device='mps').float()/255-mean)/std
            with torch.inference_mode():
                z=model(pixel_values=x).last_hidden_state[:,1:] if key=='small' else model(pixel_values=x,interpolate_pos_encoding=True).last_hidden_state
                out.append(z.transpose(1,2).reshape(-1,dim,32,32).cpu().numpy().astype(np.float16))
        np.save(A/f'features/{key}_pool.npy',np.concatenate(out));del model;torch.mps.empty_cache();print('EXTRACTED',key,len(rec),flush=True)
    (A/'features/pool_complete.json').write_text(json.dumps({'count':len(rec),'backbones':['facebook/dinov2-small','google/siglip2-base-patch16-224'],'records_file':'inputs/pool_records.csv'},indent=2))
if __name__=='__main__':main()
