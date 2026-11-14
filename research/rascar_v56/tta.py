"""Actual test-time RGB transforms; align frozen-head probabilities in image space.

Training labels are never used to build the views or predictions. OOF evaluation
uses only heads trained on the other day. The full grid of recipes is saved.
"""
from common56 import *
import torch,cv2,argparse
from torch.nn import functional as F
from transformers import AutoModel
from functools import lru_cache

VIEWS=['identity','flip','left','right','up','down','tight','wide','hires']
MODEL_SPECS=[(recipe,seed,fit) for recipe,seeds in [('control450',[3101,3102,3103]),('flip450',[3101,3102,3103]),('aug900',[3101])] for seed in seeds for fit in DAYS]

@lru_cache(maxsize=60)
def image(fid):
    im=cv2.imread(str(D/f'{fid}.jpg'));assert im is not None
    return cv2.resize(im,(1280,720))[:,:,::-1].copy()

def crop_for(r,view):
    if view in ['identity','flip','hires']:return r[CB].to_numpy(int)
    b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
    scale=2.0 if view=='tight' else 2.8 if view=='wide' else 2.4
    shifts={'left':[-.18,0],'right':[.18,0],'up':[0,-.18],'down':[0,.18]}
    cc+=np.array(shifts.get(view,[0,0]))*wh
    return np.r_[np.maximum(0,np.floor(cc-wh*scale/2)),np.minimum([1280,720],np.ceil(cc+wh*scale/2))].astype(int)

def align(p,crop,original,flip=False):
    if flip:p=p[:,::-1].copy()
    yy,xx=np.mgrid[:64,:64]
    xx=original[0]+(xx+.5)/64*(original[2]-original[0]);yy=original[1]+(yy+.5)/64*(original[3]-original[1])
    grid=p.shape[-1]
    mx=((xx-crop[0])/(crop[2]-crop[0])*grid-.5).astype(np.float32)
    my=((yy-crop[1])/(crop[3]-crop[1])*grid-.5).astype(np.float32)
    valid=(xx>=crop[0])&(xx<crop[2])&(yy>=crop[1])&(yy<crop[3])
    return cv2.remap(p.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE),valid

def extract():
    torch.set_num_threads(3);cv2.setNumThreads(1)
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');rows=rec[rec.split.isin(['oof','test'])].reset_index(drop=True)
    rows.to_csv(A/'inputs/tta_rows.csv',index=False)
    weights=[];bias=[];man=[]
    for recipe,seed,fit in MODEL_SPECS:
        path=ROOT/f'rascar_v55/models/{recipe}_s{seed}_fit{fit[-2:]}.pt'
        state=torch.load(path,weights_only=True);weights.append(state['net.weight']);bias.append(state['net.bias'])
        manifest=json.loads(path.with_suffix('.json').read_text())
        assert all(fid.split('_')[-2]==fit.replace('-','') for fid in manifest['train_frames'])
        man.append(dict(recipe=recipe,seed=seed,fit=fit,path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (A/'inputs/tta_models.json').write_text(json.dumps(man,indent=2))
    weight=torch.cat(weights);bias=torch.cat(bias);nmodels=len(MODEL_SPECS)
    model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True,trust_remote_code=False).eval()
    old=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');oldindex=rec.reset_index().set_index('id')['index'].loc[rows.id].to_numpy()
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    start=time.time();completed=[]
    for view in VIEWS:
        path=A/f'features/tta_{view}.npy';validpath=A/f'features/tta_{view}_valid.npy'
        if (A/f'features/tta_{view}_complete.json').exists():completed.append(view);continue
        out=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(nmodels,len(rows),64,64))
        coverage=np.lib.format.open_memmap(validpath,mode='w+',dtype=np.uint8,shape=(len(rows),64,64))
        size=560 if view=='hires' else 448;grid=size//14
        for off in range(0,len(rows),4):
            rr=rows.iloc[off:off+4];crops=[crop_for(r,view) for _,r in rr.iterrows()]
            if view=='identity':features=torch.from_numpy(np.array(old[oldindex[off:off+len(rr)]],np.float32))
            else:
                batch=[]
                for (_,r),cb in zip(rr.iterrows(),crops):
                    im=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2]]
                    if view=='flip':im=im[:,::-1].copy()
                    batch.append(cv2.resize(im,(size,size),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
                x=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std
                with torch.inference_mode():features=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,grid,grid)
            with torch.inference_mode():prob=F.conv2d(F.normalize(features,dim=1)*np.sqrt(384),weight,bias).sigmoid().reshape(-1,nmodels,4,grid,grid).numpy()
            for j,(_,r) in enumerate(rr.iterrows()):
                cls=CL.index(r.class_name);original=r[CB].to_numpy(float)
                for k in range(nmodels):
                    pp,v=align(prob[j,k,cls],crops[j],original,view=='flip');out[k,off+j]=pp
                coverage[off+j]=v
            if off%100==0:print('TTA',view,off,len(rows),round(time.time()-start,1),flush=True)
        out.flush();coverage.flush();(A/f'features/tta_{view}_complete.json').write_text(json.dumps(dict(view=view,rows=len(rows),heads=nmodels,seconds=time.time()-start),indent=2));completed.append(view)
    (A/'features/tta_complete.json').write_text(json.dumps(dict(views=completed,seconds=time.time()-start),indent=2))

def decode(p,r):
    _,lab,stats,_=cv2.connectedComponentsWithStats((p>=.5).astype(np.uint8),8);boxes=[]
    crop=r[CB].to_numpy(float)
    for x,y,w,h,area in stats[1:]:
        if area>=16:boxes.append(np.array([x,y,x+w,y+h])/64*np.tile(crop[2:]-crop[:2],2)+np.tile(crop[:2],2))
    return float(iou(r[B],boxes).max()) if boxes else np.nan

def evaluate():
    q,t,h=inputs();rows=pd.read_csv(A/'inputs/tta_rows.csv');models=json.loads((A/'inputs/tta_models.json').read_text())
    groups={'identity':['identity'],'flip':['identity','flip'],'shift':['identity','left','right','up','down'],
            'scale':['identity','tight','wide'],'geometry':['identity','left','right','up','down','tight','wide'],
            'all8':VIEWS[:-1],'hires':['identity','hires'],'hires_only':['hires']}
    modelgroups={'control1':('control450',[3101]),'control3':('control450',[3101,3102,3103]),'flip3':('flip450',[3101,3102,3103]),'aug900':('aug900',[3101])}
    source={v:np.load(A/f'features/tta_{v}.npy',mmap_mode='r') for v in VIEWS if (A/f'features/tta_{v}_complete.json').exists()}
    masks={v:np.load(A/f'features/tta_{v}_valid.npy',mmap_mode='r') for v in source}
    results=[]
    for mname,(recipe,seeds) in modelgroups.items():
        for gname,views in groups.items():
            if not set(views)<=set(source):continue
            evidence=[]
            for fit in DAYS:
                use=[i for i,m in enumerate(models) if m['recipe']==recipe and m['seed'] in seeds and m['fit']==fit]
                total=sum(np.mean(source[v][use].astype(np.float32),axis=0)*masks[v] for v in views)
                den=sum(masks[v].astype(np.float32) for v in views);prob=total/np.maximum(den,1)
                for i,r in rows.iterrows():
                    if r.split=='oof' and r.day==fit:continue
                    evidence.append(dict(id=r.id,split=r.split,fit_day=fit,overlap=decode(prob[i],r)))
            name=f'tta_{mname}_{gname}';e=pd.DataFrame(evidence);e.to_csv(A/f'features/{name}_evidence.csv',index=False)
            oo=e[e.split.eq('oof')].set_index('id').loc[q.id,'overlap'].to_numpy()
            tt=np.stack([e[e.split.eq('test')&e.fit_day.eq(f)].set_index('id').loc[t.id,'overlap'].to_numpy() for f in DAYS])
            results.extend(dense_variants(name,oo,tt,dict(family='tta',recipe=recipe,views=gname)))
            print('EVAL',name,max(z['ap'] for z in results[-15:]),flush=True)
    d=pd.DataFrame(results).sort_values('ap',ascending=False);d.to_csv(A/'results/tta_experiments.csv',index=False);print(d.head(20).to_string(index=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evaluate-only',action='store_true');args=parser.parse_args()
    if not args.evaluate_only:extract()
    evaluate()
