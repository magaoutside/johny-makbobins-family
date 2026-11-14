"""Exact V31 crop/input recipe, replacing only frozen DINOv2-S with B or L."""
from common46 import *
import argparse,time,cv2,torch
from functools import lru_cache
from transformers import AutoModel,SiglipVisionModel
def main():
    p=argparse.ArgumentParser();p.add_argument('key',choices=['vit_b','vit_l','siglip2']);p.add_argument('--device',default='mps');args=p.parse_args()
    torch.set_num_threads(3);cv2.setNumThreads(1);rec=records();dim={'vit_b':768,'vit_l':1024,'siglip2':768}[args.key]
    assert args.device!='mps' or torch.backends.mps.is_available(),'Local GPU is required.'
    if args.key=='siglip2':
        modelpath=ROOT/'rascar_v36/model_cache/siglip2-base-patch16-224'
        source=json.loads((modelpath/'source.json').read_text());source['url']='https://huggingface.co/'+source['repo_id'];source['sha256']=hashlib.sha256((modelpath/'model.safetensors').read_bytes()).hexdigest()
        model=SiglipVisionModel.from_pretrained(str(modelpath),local_files_only=True).eval().to(args.device)
    else:
        sources=json.loads((ROOT/'rascar_v41/model_cache/sources.json').read_text());source=next(x for x in sources if x['name']==args.key)
        modelpath=ROOT/'rascar_v41/model_cache'/args.key
        assert hashlib.sha256((modelpath/'model.safetensors').read_bytes()).hexdigest()==source['sha256']
        model=AutoModel.from_pretrained(str(modelpath),local_files_only=True,trust_remote_code=False).eval().to(args.device)
    dst=np.lib.format.open_memmap(A/f'features/{args.key}_grids.npy',mode='w+',dtype=np.float16,shape=(len(rec),dim,32,32))
    mean=torch.tensor([.5]*3 if args.key=='siglip2' else [.485,.456,.406],device=args.device).view(1,3,1,1);std=torch.tensor([.5]*3 if args.key=='siglip2' else [.229,.224,.225],device=args.device).view(1,3,1,1)
    size=512 if args.key=='siglip2' else 448
    @lru_cache(maxsize=12)
    def image(fid):
        im=cv2.imread(str(D/f'{fid}.jpg'));assert im is not None
        return cv2.resize(im,(1280,720))
    start=time.time();batch=4 if args.key=='vit_b' else 2
    for off in range(0,len(rec),batch):
        images=[]
        for _,r in rec.iloc[off:off+batch].iterrows():
            b=r[CB].to_numpy(int);im=image(r.frame_id)[b[1]:b[3],b[0]:b[2],::-1]
            images.append(cv2.resize(im,(size,size),interpolation=cv2.INTER_LINEAR if args.key=='siglip2' else cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=(torch.from_numpy(np.stack(images)).to(args.device).float()/255-mean)/std
        with torch.inference_mode():
            z=model(pixel_values=x,interpolate_pos_encoding=True).last_hidden_state if args.key=='siglip2' else model(pixel_values=x).last_hidden_state[:,1:]
            z=z.transpose(1,2).reshape(-1,dim,32,32)
        z=z.cpu().numpy().astype(np.float16);assert np.isfinite(z).all();dst[off:off+len(z)]=z
        if off%80==0:print('EXTRACT',args.key,off,len(rec),round(time.time()-start,1),flush=True)
    dst.flush();manifest=dict(source=source,count=len(rec),shape=list(dst.shape),device=args.device,seconds=time.time()-start,input=size,grid=32,position_interpolation=args.key=='siglip2',record_sha256=hashlib.sha256((ROOT/'rascar_v31/inputs/dense_records.csv').read_bytes()).hexdigest())
    (A/f'features/{args.key}_complete.json').write_text(json.dumps(manifest,indent=2));print('DONE',args.key,manifest['seconds'],flush=True)
if __name__=='__main__':main()
