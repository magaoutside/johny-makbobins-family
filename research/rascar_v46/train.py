"""Frozen-backbone linear occupancy: same V31 seed/sampling/loss/steps."""
from common46 import *
import argparse,time,torch,cv2
from torch import nn
import torch.nn.functional as F
class Dense(nn.Module):
    def __init__(self,dim):super().__init__();self.dim=dim;self.net=nn.Conv2d(dim,4,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(self.dim))
def decode(prob,r):
    cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];k=CL.index(r.class_name);p=prob[k]
    _,labels,stats,_=cv2.connectedComponentsWithStats((p>=.5).astype(np.uint8),8);refs=[]
    for j,(x,y,w,hh,area) in enumerate(stats[1:],1):
        if area>=4:refs.append((np.array([x,y,x+w,y+hh])/32*np.tile(size,2)+np.tile(cb[:2],2),float(p[labels==j].mean())))
    if refs:
        ov=iou(r[B],[v[0] for v in refs])[0];j=ov.argmax();bb,confidence=refs[j];overlap=float(ov[j])
    else:bb=np.full(4,np.nan);confidence=0.;overlap=np.nan
    return dict(overlap=overlap,confidence=confidence,references=len(refs),**dict(zip(['ref_'+b for b in B],bb)))
def main():
    p=argparse.ArgumentParser();p.add_argument('key',choices=['vit_b','vit_l','siglip2','small_control']);p.add_argument('--seed',type=int,default=3101);args=p.parse_args()
    torch.set_num_threads(3);cv2.setNumThreads(1);rec=records();q,te,h=inputs()
    path=ROOT/'rascar_v31/features/dense_grids.npy' if args.key=='small_control' else A/f'features/{args.key}_grids.npy'
    if args.key!='small_control':
        complete=json.loads((A/f'features/{args.key}_complete.json').read_text())
        assert complete['record_sha256']==hashlib.sha256((ROOT/'rascar_v31/inputs/dense_records.csv').read_bytes()).hexdigest()
    feat=np.load(path,mmap_mode='r');target=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r');manifest=[];output=[]
    for fit in DAYS:
        ix=rec.index[rec.split.eq('manual')&rec.day.eq(fit)].to_numpy()
        ev=rec.index[rec.split.eq('test')|(rec.split.isin(['oof','manual'])&rec.day.ne(fit)&rec.class_name.isin(CL))].to_numpy()
        assert not set(rec.loc[ix,'frame_id'])&set(rec.loc[ev,'frame_id'])
        counts=rec.loc[ix].groupby('frame_id').size();prob=np.array([1/counts[f] for f in rec.loc[ix,'frame_id']]);prob/=prob.sum()
        torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=Dense(feat.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01);losses=[];start=time.time()
        for step in range(450):
            chosen=rng.choice(ix,8,replace=True,p=prob);x=torch.from_numpy(np.array(feat[chosen],np.float32));y=torch.from_numpy(np.array(target[chosen],np.float32));z=model(x)
            loss=F.binary_cross_entropy_with_logits(z,y,weight=1+3*y);assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            if step%100==0:print('TRAIN',args.key,args.seed,fit,step,round(np.mean(losses[-100:]),5),round(time.time()-start,1),flush=True)
        name=f'{args.key}_s{args.seed}_fit{fit[-2:]}';torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();maps=[];ids=[]
        for off in range(0,len(ev),16):
            indices=ev[off:off+16]
            with torch.inference_mode():pr=model(torch.from_numpy(np.array(feat[indices],np.float32))).sigmoid().numpy()
            for j,i in enumerate(indices):
                r=rec.loc[i];output.append(dict(id=r.id,origin_id=r.origin_id,frame_id=r.frame_id,day=r.day,split=r.split,source='query' if r.split=='oof' else r.split,class_name=r.class_name,fit_day=fit,key=args.key,seed=args.seed,**decode(pr[j],r)))
                if r.split in ['oof','test']:maps.append(pr[j].astype(np.float16));ids.append(r.id)
        np.savez_compressed(A/f'features/{name}_maps.npz',ids=np.array(ids),maps=np.stack(maps));manifest.append(dict(name=name,fit_day=fit,seed=args.seed,fit_frames=sorted(rec.loc[ix].frame_id.unique()),manual_crops=len(ix),steps=450,last_loss=float(np.mean(losses[-100:])),seconds=time.time()-start))
        print('DONE',name,round(time.time()-start,1),flush=True)
    pd.DataFrame(output).to_csv(A/f'features/{args.key}_s{args.seed}_evidence.csv',index=False);(A/f'models/{args.key}_s{args.seed}_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
