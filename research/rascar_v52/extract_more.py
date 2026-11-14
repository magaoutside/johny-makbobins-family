"""Frozen detector embeddings, distinct from target-fitted model outputs."""
from prepare import *
import argparse,torch,time,os
from functools import lru_cache
def main():
    p=argparse.ArgumentParser();p.add_argument('key',choices=['owl','yolo']);args=p.parse_args();rows=pd.read_csv(A/'inputs/rows.csv');rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id').loc[rows.id];torch.set_num_threads(2);start=time.time();out=[]
    @lru_cache(maxsize=12)
    def image(fid):return cv2.resize(cv2.imread(str(D/f'{fid}.jpg')),(1280,720))
    if args.key=='owl':
        from transformers import Owlv2ForObjectDetection,Owlv2Processor
        from PIL import Image
        assert torch.backends.mps.is_available();path=ROOT/'rascar_v50/model_cache/owlv2';model=Owlv2ForObjectDetection.from_pretrained(str(path),local_files_only=True).eval().to('mps');proc=Owlv2Processor.from_pretrained(str(path),local_files_only=True)
        for j,(_,r) in enumerate(rec.iterrows()):
            cb=r[CB].to_numpy(int);im=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];x=proc.image_processor(images=Image.fromarray(im),return_tensors='pt')['pixel_values'].to('mps')
            with torch.inference_mode():f,_=model.image_embedder(x)
            z=f[0].float().cpu().numpy().transpose(2,0,1);side=max(im.shape[:2]);n=z.shape[1];yy,xx=np.mgrid[:n,:n];xx=(xx+.5)*side/n+cb[0];yy=(yy+.5)*side/n+cb[1];bb=r[B].to_numpy(float);valid=(xx<cb[2])&(yy<cb[3]);mask=(xx>=bb[0])&(xx<=bb[2])&(yy>=bb[1])&(yy<=bb[3])&valid;inside=z[:,mask].mean(1);outside=z[:,valid&~mask].mean(1);out.append(np.r_[z[:,valid].mean(1),inside,inside-outside])
            if j%40==0:print('OWL EMB',j,len(rec),round(time.time()-start,1),flush=True)
    else:
        os.environ['YOLO_CONFIG_DIR']=str(A/'features/yolo_config')
        from ultralytics import YOLO
        net=YOLO(str(ROOT/'rascar_v41/model_cache/yolo26n.pt')).model.eval();layers=[]
        for m in net.model:
            if m.type.endswith(('C3k2','C2PSA')):layers.append(m.i)
        # Multi-scale visual features before Detect; fixed pretrained weights.
        layers=layers[-3:];assert layers;print('YOLO LAYERS',layers,flush=True)
        for off in range(0,len(rec),16):
            ims=[]
            for _,r in rec.iloc[off:off+16].iterrows():
                cb=r[CB].to_numpy(int);im=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2],::-1];ims.append(cv2.resize(im,(320,320)).transpose(2,0,1).copy())
            with torch.inference_mode():z=net(torch.tensor(np.stack(ims)).float()/255,embed=layers)
            out.extend([v.numpy() for v in z]);print('YOLO EMB',off,len(rec),round(time.time()-start,1),flush=True)
    f=np.asarray(out,np.float32);assert len(f)==len(rows) and np.isfinite(f).all();np.save(A/f'features/{args.key}.npy',f);(A/f'features/{args.key}_manifest.json').write_text(json.dumps(dict(backbone='google/owlv2-base-patch16-ensemble' if args.key=='owl' else 'pretrained YOLO26n',shape=list(f.shape),seconds=time.time()-start,row_order_sha256=hashlib.sha256((A/'inputs/rows.csv').read_bytes()).hexdigest(),competition_fine_tuning=False,pool_fit=False),indent=2));print('DONE',args.key,f.shape,flush=True)
if __name__=='__main__':main()
