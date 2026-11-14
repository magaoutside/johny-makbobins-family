from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v43'))
from common43 import *
A=ROOT/'rascar_v44'
import torch,cv2
from torch import nn
import torch.nn.functional as F
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
class Head(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Conv2d(384,1,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(384))[:,0]
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['control','silver'],required=True);args=p.parse_args()
    torch.set_num_threads(4);device='mps';assert torch.backends.mps.is_available()
    assert (A/'features/manifest.json').exists()
    old=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');new=pd.read_csv(A/'inputs/pool_records.csv')
    oz=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');oy=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
    nz=np.load(A/'features/pool_grids.npy',mmap_mode='r');ny=np.load(A/'features/pool_targets.npy');nv=np.load(A/'features/pool_valid.npy')
    preds=[];manifests=[]
    for fit in DAYS:
        torch.manual_seed(4401);rng=np.random.default_rng(4401);pool_rng=np.random.default_rng(4402);m=Head();weights=torch.load(ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt',map_location='cpu',weights_only=True)
        with torch.no_grad():m.net.weight.copy_(weights['net.weight'][1:2]);m.net.bias.copy_(weights['net.bias'][1:2])
        m.to(device);opt=torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.01)
        oi=old.index[old.split.eq('manual')&old.day.eq(fit)].to_numpy();ni=new.index[new.split.eq('train')&new.day.eq(fit)].to_numpy();counts=old.loc[oi].frame_id.value_counts();pw=np.array([1/counts[f] for f in old.loc[oi,'frame_id']]);pw/=pw.sum()
        assert not set(old.loc[oi,'frame_id'])&set(new[new.split.eq('reserved_holdout')].frame_id)
        losses=[];start=time.time()
        for step in range(350):
            ix=rng.choice(oi,8,p=pw);x=torch.tensor(np.asarray(oz[ix]),dtype=torch.float32,device=device);y=torch.tensor(np.asarray(oy[ix,1]),dtype=torch.float32,device=device)
            log=m(x);loss=F.binary_cross_entropy_with_logits(log,y,weight=1+3*y)
            if args.variant=='silver':
                j=pool_rng.choice(ni,4);nx=torch.tensor(np.asarray(nz[j]),dtype=torch.float32,device=device);target=torch.tensor(ny[j],dtype=torch.float32,device=device);valid=torch.tensor(nv[j],dtype=torch.float32,device=device)
                pl=F.binary_cross_entropy_with_logits(m(nx),target,weight=(1+3*target)*valid,reduction='none');pool_loss=pl.sum()/valid.sum().clamp_min(1)
                loss=loss+.35*pool_loss
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step();losses.append(float(loss.detach()))
            if step%100==0:print('TRAIN',args.variant,fit,step,round(np.mean(losses[-50:]),5),round(time.time()-start,1),flush=True)
        torch.save(m.cpu().state_dict(),A/f'models/{args.variant}_fit{fit[-2:]}.pt');m.to(device).eval()
        jobs=[('queries',old.index[old.split.isin(['oof','test'])&old.class_name.eq('nude_hands')].to_numpy(),old,oz),('pool',new.index[new.split.eq('reserved_holdout')].to_numpy(),new,nz)]
        for mode,ids,rec,features in jobs:
            for off in range(0,len(ids),16):
                ix=ids[off:off+16]
                with torch.inference_mode():prob=m(torch.tensor(np.asarray(features[ix]),dtype=torch.float32,device=device)).sigmoid().cpu().numpy()
                for k,i in enumerate(ix):
                    r=rec.loc[i];cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];n,lab,stats,_=cv2.connectedComponentsWithStats((prob[k]>=.5).astype(np.uint8),8);refs=[]
                    for c in range(1,n):
                        x,y,w,hh,area=stats[c]
                        if area<4:continue
                        box=np.r_[x,y,x+w,y+hh]/32*np.tile(size,2)+np.tile(cb[:2],2);refs.append((float(iou(r[B],box)[0,0]),box,float(prob[k][lab==c].mean())))
                    if refs:ov,b,confidence=max(refs,key=lambda x:x[0])
                    else:ov=np.nan;b=np.full(4,np.nan);confidence=0
                    preds.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,split=r.split,fit_day=fit,variant=args.variant,overlap=ov,confidence=confidence,references=len(refs),**dict(zip(['ref_'+v for v in B],b))))
        pd.DataFrame(preds).to_csv(A/f'results/{args.variant}_predictions.csv',index=False)
        manifests.append(dict(variant=args.variant,fit_day=fit,seed=4401,steps=350,lr=.0001,new_loss_weight=.35 if args.variant=='silver' else 0,old_train_frames=sorted(old.loc[oi,'frame_id'].unique()),new_train_frames=sorted(new.loc[ni,'frame_id'].unique()) if args.variant=='silver' else [],reserved_pool_excluded=True,test_targets=False,loss=float(np.mean(losses[-50:])),seconds=time.time()-start))
        (A/f'models/{args.variant}_manifest.json').write_text(json.dumps(manifests,indent=2));print('DONE',args.variant,fit,flush=True)
if __name__=='__main__':main()
