"""Frozen DINOv2 ViT-S layers 3/6/9/12 on the exact V31 crops.
Earlier layers add local appearance; last layer remains a semantics baseline.
No annotation/test-label fitting in feature extraction.
"""
from common38 import *
import torch,cv2,time,os
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ.setdefault('MPLCONFIGDIR',str(A/'features/mpl'))
from transformers import AutoModel
torch.set_num_threads(3);cv2.setNumThreads(1)
CB=['crop_'+b for b in B]

def main():
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');rec.to_csv(A/'inputs/multilevel_records.csv',index=False)
    model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True).eval()
    layers=[3,6,9,12];shape=(len(rec),len(layers)*384,32,32)
    dest=A/'features/multilevel_grids.npy';done=A/'features/multilevel_complete.json'
    if done.exists():assert np.load(dest,mmap_mode='r').shape==shape;print('CACHED',shape);return
    bank=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float16,shape=shape)
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1);start=time.time()
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            cb=r[CB].to_numpy(int);im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))
            im=im[cb[1]:cb[3],cb[0]:cb[2],::-1];batch.append(cv2.resize(im,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        with torch.inference_mode():
            z=model(pixel_values=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std,output_hidden_states=True)
            # Apply the frozen backbone's LayerNorm consistently at every depth.
            out=torch.cat([model.layernorm(z.hidden_states[l])[:,1:].transpose(1,2).reshape(-1,384,32,32) for l in layers],1)
        bank[off:off+len(batch)]=out.numpy().astype(np.float16)
        if off%120==0:print('MULTILEVEL',off,len(rec),round(time.time()-start,1),flush=True)
    bank.flush();done.write_text(json.dumps(dict(layers=layers,shape=shape,seconds=time.time()-start,backbone='facebook/dinov2-small',source='https://github.com/facebookresearch/dinov2',test_targets_used=False),indent=2))

if __name__=='__main__':main()
