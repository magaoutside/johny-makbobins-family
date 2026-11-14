"""Instance-specific rectangle localization on detector features, not union masks."""
from common43 import *
import torch,time,argparse
from torch import nn
import torch.nn.functional as F
class Local(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(263,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,1,1))
    def forward(self,z,mask,classes):
        n=len(z);xy=torch.linspace(0,1,28,device=z.device);yy,xx=torch.meshgrid(xy,xy,indexing='ij');coord=torch.stack([xx,yy])[None].expand(n,-1,-1,-1)
        cls=F.one_hot(classes,4).float()[:,:,None,None].expand(-1,-1,28,28)
        return self.net(torch.cat([z,mask[:,None],coord,cls],1))[:,0]
def masks(boxes,cb):
    b=(boxes-np.tile(cb[:,:2],2))/np.tile(cb[:,2:]-cb[:,:2],2);p=(np.arange(28)+.5)/28
    wx=(np.minimum(b[:,2,None],p+1/56)-np.maximum(b[:,0,None],p-1/56)).clip(0)*28
    wy=(np.minimum(b[:,3,None],p+1/56)-np.maximum(b[:,1,None],p-1/56)).clip(0)*28
    return (wy[:,:,None]*wx[:,None,:]).astype(np.float32)
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['clean','pool'],required=True);args=p.parse_args()
    torch.set_num_threads(4);device='mps';assert torch.backends.mps.is_available()
    manifest_cache=json.loads((A/'features/local_manifest.json').read_text())
    r=pd.read_csv(A/'inputs/records.csv');assert manifest_cache['rows']==len(r)
    z=np.load(A/'features/local_grids.npy',mmap_mode='r');cb=np.load(A/'features/local_context.npy');b=r[B].to_numpy(float)
    qm=masks(b,cb);gt=r[['gt_'+v for v in B]].to_numpy(float);target=masks(np.nan_to_num(gt),cb);target[r.gt_iou.fillna(0).lt(.1).to_numpy()]=0
    cls=np.array([CL.index(c) for c in r.class_name]);manifest=[];pred=[]
    for fit in DAYS:
        torch.manual_seed(4302);rng=np.random.default_rng(4302);model=Local().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.02);start=time.time();losses=[]
        groups={s:r.index[r.day.eq(fit)&r.source.eq(s)].to_numpy() for s in ['query','synthetic','pool']}
        def sample(s,n):
            ix=groups[s];counts=r.loc[ix].groupby(['frame_id','class_name']).id.transform('size').to_numpy();pw=1/counts;pw/=pw.sum();return rng.choice(ix,n,p=pw)
        for step in range(600):
            ix=np.r_[sample('query',8),sample('synthetic',12 if args.variant=='pool' else 16),sample('pool',4) if args.variant=='pool' else []].astype(int)
            x=torch.tensor(np.asarray(z[ix]),dtype=torch.float32,device=device);y=torch.tensor(target[ix],device=device)
            logits=model(x,torch.tensor(qm[ix],device=device),torch.tensor(cls[ix],device=device));prob=logits.sigmoid()
            # Full box outside the crop stays outside: target does not redefine GT.
            weight=torch.tensor(r.loc[ix,'weight'].to_numpy(np.float32),device=device)
            pixel=F.binary_cross_entropy_with_logits(logits,y,reduction='none').mean((1,2))
            dice=1-(2*(prob*y).sum((1,2))+1)/((prob+y).sum((1,2))+1)
            loss=(weight*(pixel+dice)).mean();assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
            if step%150==0:print('TRAIN local',args.variant,fit,step,round(np.mean(losses[-75:]),5),round(time.time()-start,1),flush=True)
        torch.save(model.cpu().state_dict(),A/f'models/local_{args.variant}_fit{fit[-2:]}.pt');model.to(device).eval();jobs=r.index[r.source.isin(['query','test'])].to_numpy()
        allmaps=np.zeros((len(jobs),28,28),np.float16)
        with torch.inference_mode():
          for off in range(0,len(jobs),32):
            ix=jobs[off:off+32];x=torch.tensor(np.asarray(z[ix]),dtype=torch.float32,device=device);probs=model(x,torch.tensor(qm[ix],device=device),torch.tensor(cls[ix],device=device)).sigmoid().cpu().numpy();allmaps[off:off+len(ix)]=probs
            for j,i in enumerate(ix):
                # One bbox over the selected instance map; track multiple components
                # for ambiguity rather than silently using masks as exact GT.
                import cv2
                mask=probs[j]>=.5;num,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
                candidates=[]
                for k in range(1,num):
                    xx,yy,ww,hh,area=stats[k]
                    if area<2:continue
                    box=cb[i,:2]+np.array([xx,yy])/28*(cb[i,2:]-cb[i,:2]);end=cb[i,:2]+np.array([xx+ww,yy+hh])/28*(cb[i,2:]-cb[i,:2]);bb=np.r_[box,end]
                    candidates.append((float(iou(b[i],bb)[0,0]),bb,float(probs[j][labels==k].mean())))
                if candidates:ov,ref,conf=max(candidates,key=lambda x:x[0])
                else:ov=np.nan;ref=np.full(4,np.nan);conf=0.
                v=r.loc[i];pred.append(dict(id=v.id,frame_id=v.frame_id,day=v.day,class_name=v.class_name,source=v.source,fit_day=fit,variant=args.variant,ref_iou=ov,mask_confidence=conf,components=len(candidates),**dict(zip(['ref_'+x for x in B],ref))))
        np.savez_compressed(A/f'features/local_{args.variant}_fit{fit[-2:]}_maps.npz',ids=r.loc[jobs,'id'].to_numpy(str),maps=allmaps)
        pd.DataFrame(pred).to_csv(A/f'results/local_{args.variant}_predictions.csv',index=False)
        tids=np.r_[groups['query'],groups['synthetic'],groups['pool'] if args.variant=='pool' else []].astype(int)
        manifest.append(dict(variant=args.variant,fit_day=fit,seed=4302,steps=600,fit_frames=sorted(r.loc[tids,'frame_id'].unique()),final_loss=float(np.mean(losses[-75:])),seconds=time.time()-start))
        (A/f'models/local_{args.variant}_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',args.variant,fit,time.time()-start,flush=True)
        del model;torch.mps.empty_cache()
if __name__=='__main__':main()
