from common43 import *
import torch,copy,time,argparse
from torch import nn
import torch.nn.functional as F

class Refiner(nn.Module):
    def __init__(self,base):
        super().__init__();self.visual=nn.Sequential(*copy.deepcopy(list(base.roi_heads.box_head.children())))
        for p in self.visual[:3].parameters():p.requires_grad=False
        self.prior=nn.Linear(1024,16)
        w=base.roi_heads.box_predictor.bbox_pred.weight.detach().reshape(91,4,1024)
        b=base.roi_heads.box_predictor.bbox_pred.bias.detach().reshape(91,4)
        with torch.no_grad():
            self.prior.weight.zero_();self.prior.bias.zero_()
            for cls,coco in [(0,1),(2,47)]:
                scale=torch.tensor([10.,10.,5.,5.]);self.prior.weight[cls*4:(cls+1)*4]=w[coco]/scale[:,None];self.prior.bias[cls*4:(cls+1)*4]=b[coco]/scale
        self.head=nn.Sequential(nn.LayerNorm(2056),nn.Linear(2056,128),nn.GELU(),nn.Dropout(.25),nn.Linear(128,24))
        with torch.no_grad():self.head[-1].weight.mul_(.01);self.head[-1].bias.zero_()
    def train(self,mode=True):
        super().train(mode)
        for layer in self.visual.modules():
            if isinstance(layer,nn.BatchNorm2d):layer.eval()
        return self
    def forward(self,x,meta,cls):
        n=len(x);z=x.reshape(-1,256,7,7)
        with torch.no_grad():z=self.visual[:3](z)
        z=self.visual[3:](z).reshape(n,2,1024)
        out=self.head(torch.cat([z.flatten(1),meta],1)).reshape(n,4,6)[torch.arange(n,device=x.device),cls]
        prior=self.prior(z[:,0]).reshape(n,4,4)[torch.arange(n,device=x.device),cls]
        return out[:,:2],out[:,2:]+prior

def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['clean','pool'],required=True);args=p.parse_args()
    torch.set_num_threads(4);device='mps';assert torch.backends.mps.is_available()
    r=pd.read_csv(A/'inputs/records.csv');maps=np.load(A/'features/roi_maps.npy',mmap_mode='r')
    b=r[B].to_numpy(np.float32);wh=b[:,2:]-b[:,:2];cc=(b[:,2:]+b[:,:2])/2
    meta=np.c_[wh/[1280,720],cc/[1280,720],np.eye(4)[[CL.index(c) for c in r.class_name]]].astype(np.float32)
    cls=np.array([CL.index(c) for c in r.class_name]);delta=geometry(b,r[['gt_'+v for v in B]].to_numpy(np.float32))
    base=detector();manifest=[];all_predictions=[]
    for fit in DAYS:
        torch.manual_seed(4301);rng=np.random.default_rng(4301);model=Refiner(base).to(device).train();start=time.time()
        optimizer=torch.optim.AdamW([{'params':[p for p in model.visual.parameters() if p.requires_grad],'lr':1e-5},{'params':list(model.prior.parameters())+list(model.head.parameters()),'lr':2e-4}],weight_decay=.03)
        groups={s:r.index[r.day.eq(fit)&r.source.eq(s)].to_numpy() for s in ['query','synthetic','pool']}
        def sample(source,n):
            ix=groups[source];counts=r.loc[ix].groupby(['frame_id','class_name']).id.transform('size').to_numpy();pw=1/counts;pw/=pw.sum()
            return rng.choice(ix,n,p=pw)
        train_ids=np.r_[groups['query'],groups['synthetic'],groups['pool'] if args.variant=='pool' else []].astype(int)
        losses=[]
        for step in range(400):
            ix=np.r_[sample('query',12),sample('synthetic',10 if args.variant=='pool' else 12),sample('pool',2) if args.variant=='pool' else []].astype(int)
            x=torch.tensor(np.asarray(maps[ix]),dtype=torch.float32,device=device)
            mm=torch.tensor(meta[ix],device=device);cl=torch.tensor(cls[ix],device=device);logits,pred_delta=model(x,mm,cl)
            y=torch.tensor(r.loc[ix,'target'].to_numpy(np.float32),device=device);quality=torch.tensor(r.loc[ix,'gt_iou'].to_numpy(np.float32),device=device)
            weights=torch.tensor(r.loc[ix,'weight'].to_numpy(np.float32),device=device)
            ld=F.smooth_l1_loss(pred_delta,torch.tensor(delta[ix],device=device),beta=.1,reduction='none').mean(1)
            loss=(weights*(F.binary_cross_entropy_with_logits(logits[:,0],y,reduction='none')+3*F.smooth_l1_loss(logits[:,1].sigmoid(),quality,beta=.1,reduction='none')+ld*(quality>.1))).mean()
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step();losses.append(float(loss.detach()))
            if step%100==0:print('TRAIN',args.variant,fit,step,round(np.mean(losses[-50:]),5),round(time.time()-start,1),flush=True)
        model.eval();torch.save(model.cpu().state_dict(),A/f'models/{args.variant}_fit{fit[-2:]}.pt');model.to(device)
        # Save fit and held predictions for overfit diagnosis, only held is OOF.
        jobs=r.index[r.source.isin(['query','test'])].to_numpy();out=[]
        with torch.inference_mode():
          for off in range(0,len(jobs),32):
            ix=jobs[off:off+32];x=torch.tensor(np.asarray(maps[ix]),dtype=torch.float32,device=device)
            logits,dp=model(x,torch.tensor(meta[ix],device=device),torch.tensor(cls[ix],device=device));prob=logits.sigmoid().cpu().numpy()
            refined=decode(torch.tensor(b[ix],device=device),dp).cpu().numpy();refined=np.clip(refined,[0,0,0,0],[1280,720,1280,720])
            for j,i in enumerate(ix):
              row=r.loc[i];ov=float(iou(b[i],refined[j])[0,0]);out.append(dict(id=row.id,frame_id=row.frame_id,day=row.day,source=row.source,class_name=row.class_name,fit_day=fit,variant=args.variant,p_error=float(prob[j,0]),pred_iou=float(prob[j,1]),ref_iou=ov,**dict(zip(['ref_'+v for v in B],refined[j]))))
        all_predictions.extend(out);pd.DataFrame(all_predictions).to_csv(A/f'results/{args.variant}_predictions.csv',index=False)
        manifest.append(dict(variant=args.variant,fit_day=fit,seed=4301,steps=400,fit_frames=sorted(r.loc[train_ids,'frame_id'].unique()),source_counts=r.loc[train_ids,'source'].value_counts().to_dict(),final_loss=float(np.mean(losses[-50:])),seconds=time.time()-start))
        (A/f'models/{args.variant}_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',args.variant,fit,round(time.time()-start,1),flush=True)
        del model;torch.mps.empty_cache()
if __name__=='__main__':main()
