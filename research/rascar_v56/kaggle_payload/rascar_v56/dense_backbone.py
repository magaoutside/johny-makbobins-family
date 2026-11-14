"""Upgrade dense occupancy backbone with local DINOv2 base/large weights.

Fixed manual crops, opposite-day evaluation, synchronized horizontal RGB flips.
"""
from common56 import *
import torch,cv2,argparse
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from functools import lru_cache

@lru_cache(maxsize=50)
def image(frame):
    im=cv2.imread(str(D/f'{frame}.jpg'));assert im is not None
    return cv2.resize(im,(1280,720))[:,:,::-1].copy()

def extract(backbone,device,rec):
    width={'vit_b':768,'vit_l':1024}[backbone];path=A/f'features/{backbone}_dense.npy';meta=A/f'features/{backbone}_dense_progress.json'
    manual=rec.index[rec.split.eq('manual')].to_numpy();records=[(i,False) for i in rec.index]+[(i,True) for i in manual]
    n=len(records);shape=(n,width,32,32);off=json.loads(meta.read_text())['completed'] if meta.exists() else 0
    if off==n:return
    out=np.lib.format.open_memmap(path,mode='r+' if off else 'w+',dtype=np.float16,shape=shape)
    model,loading=AutoModel.from_pretrained(str(ROOT/f'rascar_v41/model_cache/{backbone}'),local_files_only=True,output_loading_info=True)
    assert not loading['missing_keys'];model.eval().to(device)
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    start=time.time();batch_size=2 if backbone=='vit_b' else 1
    for at in range(off,n,batch_size):
        batch=[]
        for idx,flip in records[at:at+batch_size]:
            r=rec.loc[idx];c=r[CB].to_numpy(int);x=image(r.frame_id)[c[1]:c[3],c[0]:c[2]]
            if flip:x=x[:,::-1]
            batch.append(cv2.resize(x,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        x=(torch.from_numpy(np.stack(batch)).to(device).float()/255-mean)/std
        with torch.inference_mode():z=model(pixel_values=x).last_hidden_state[:,1:].transpose(1,2).reshape(-1,width,32,32)
        out[at:at+len(batch)]=z.cpu().numpy().astype(np.float16)
        if at%50==0 or at+batch_size>=n:
            out.flush();meta.write_text(json.dumps(dict(completed=min(at+batch_size,n),shape=shape,backbone=backbone,device=device,seconds=time.time()-start,record_sha256=hashlib.sha256((ROOT/'rascar_v31/inputs/dense_records.csv').read_bytes()).hexdigest())))
            print('EXTRACT',backbone,at,n,round(time.time()-start,1),flush=True)
    del model;image.cache_clear()
    if device=='mps':torch.mps.empty_cache()

class Head(nn.Module):
    def __init__(self,width):super().__init__();self.width=width;self.net=nn.Conv2d(width,4,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(self.width))

def train(backbone,device,rec,seeds,recipes):
    x=np.load(A/f'features/{backbone}_dense.npy',mmap_mode='r');y=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
    manual=rec.index[rec.split.eq('manual')].to_numpy();flip_index={v:len(rec)+i for i,v in enumerate(manual)}
    for recipe in recipes:
        for seed in seeds:
            for fit in DAYS:
                name=f'{backbone}_{recipe}_s{seed}_fit{fit[-2:]}';output=A/f'features/{name}_maps.npz'
                if output.exists():continue
                tr=rec.index[rec.split.eq('manual')&rec.day.eq(fit)].to_numpy();ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit))].to_numpy()
                assert set(rec.loc[tr,'frame_id']).isdisjoint(rec.loc[ev,'frame_id'])
                freq=rec.loc[tr].groupby('frame_id').size();prob=np.array([1/freq[f] for f in rec.loc[tr,'frame_id']]);prob/=prob.sum()
                torch.manual_seed(seed);rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+15000)
                model=Head(x.shape[1]).to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01);losses=[];start=time.time()
                for step in range(450):
                    ix=rng.choice(tr,8,replace=True,p=prob);xx=[];yy=[]
                    for i in ix:
                        flip=recipe=='flip450' and arng.random()<.5
                        xx.append(x[flip_index[i] if flip else i]);yy.append(y[i,:,:,::-1] if flip else y[i])
                    xb=torch.from_numpy(np.array(xx,np.float32)).to(device);yb=torch.from_numpy(np.array(yy,np.float32)).to(device)
                    loss=F.binary_cross_entropy_with_logits(model(xb),yb,weight=1+3*yb)
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach().cpu()))
                torch.save({k:v.cpu() for k,v in model.state_dict().items()},A/f'models/{name}.pt');model.eval();maps=[]
                for at in range(0,len(ev),16):
                    with torch.inference_mode():maps.append(model(torch.from_numpy(np.array(x[ev[at:at+16]],np.float32)).to(device)).sigmoid().cpu().numpy())
                np.savez_compressed(output,ids=rec.loc[ev,'id'].to_numpy(dtype=str),maps=np.concatenate(maps))
                manifest=dict(backbone=backbone,recipe=recipe,seed=seed,fit_day=fit,steps=450,train_frames=sorted(rec.loc[tr,'frame_id'].unique()),validation_frames=sorted(rec.loc[ev[rec.loc[ev,'split'].eq('oof')],'frame_id'].unique()),loss=float(np.mean(losses[-100:])),seconds=time.time()-start)
                (A/f'models/{name}.json').write_text(json.dumps(manifest,indent=2));print('TRAIN',name,round(time.time()-start,1),flush=True)

def overlap(prob,rows):
    values=[]
    for p,(_,r) in zip(prob,rows.iterrows()):
        crop=r[CB].to_numpy(float);k=CL.index(r.class_name);_,lab,stats,_=cv2.connectedComponentsWithStats((p[k]>=.5).astype(np.uint8),8);boxes=[]
        for x,y,w,h,area in stats[1:]:
            if area>=4:boxes.append(np.array([x,y,x+w,y+h])/32*np.tile(crop[2:]-crop[:2],2)+np.tile(crop[:2],2))
        values.append(float(iou(r[B],boxes).max()) if boxes else np.nan)
    return np.array(values)

def evaluate(backbone,rec,seeds,recipes):
    q,t,h=inputs();results=[];rec=rec.set_index('id')
    for recipe in recipes:
        for mix in [0.]:
            evidence=[]
            for fit in DAYS:
                maps=[];ids=None
                for seed in seeds:
                    d=np.load(A/f'features/{backbone}_{recipe}_s{seed}_fit{fit[-2:]}_maps.npz');maps.append(d['maps']);ids=d['ids']
                pred=np.mean(maps,axis=0)
                if mix:
                    old=[]
                    for seed in seeds:
                        d=np.load(ROOT/f'rascar_v55/features/{recipe}_s{seed}_fit{fit[-2:]}_maps.npz');assert np.array_equal(ids,d['ids']);old.append(d['maps'])
                    pred=(1-mix)*pred+mix*np.mean(old,axis=0)
                r=rec.loc[ids];evidence.append(r[['split','day']].reset_index().assign(fit_day=fit,overlap=overlap(pred,r)))
            e=pd.concat(evidence);name=f'{backbone}_{recipe}_mean{len(seeds)}_small{int(mix*100)}';e.to_csv(A/f'features/{name}_evidence.csv',index=False)
            oo=e[e.split.eq('oof')].set_index('id').loc[q.id,'overlap'].to_numpy();tt=np.stack([g.set_index('id').loc[t.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')])
            results+=dense_variants(name,oo,tt,dict(backbone=backbone,recipe=recipe,small_mix=mix,seeds=len(seeds)))
    pd.DataFrame(results).sort_values('ap',ascending=False).to_csv(A/f'results/dense_{backbone}.csv',index=False);print(pd.DataFrame(results).sort_values('ap',ascending=False).head(10).to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=['vit_b','vit_l'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seeds',nargs='+',type=int,default=[3101,3102,3103]);p.add_argument('--recipes',nargs='+',default=['control450','flip450']);p.add_argument('--evaluate-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(3);cv2.setNumThreads(1);rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv')
    if not args.evaluate_only:extract(args.backbone,args.device,rec);train(args.backbone,args.device,rec,args.seeds,args.recipes)
    evaluate(args.backbone,rec,args.seeds,args.recipes)

if __name__=='__main__':main()
