"""Small side-distribution head, inspired by SABL, NOT a SABL reproduction.

Independent side logits, sub-bin offsets, object presence and query quality.
All targets are built from fit-day manual boxes, never test visual annotations.
Two ablations: manual crops only; plus empty locations and fit-day raw queries.
"""
from pathlib import Path
import sys,json,time,argparse
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v33';V31=ROOT/'rascar_v31'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import h,q,B,CL,DAYS,iou
import numpy as np,pandas as pd,torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3)
CB=['crop_x1','crop_y1','crop_x2','crop_y2']
yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32

class SideHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk=nn.Sequential(nn.Conv2d(392,48,1),nn.GroupNorm(6,48),nn.GELU(),nn.Conv2d(48,48,3,padding=1),nn.GroupNorm(6,48),nn.GELU())
        self.att=nn.Conv2d(48,4,1)
        self.sides=nn.ModuleList([nn.Sequential(nn.Conv1d(48,32,3,padding=1),nn.GELU(),nn.Conv1d(32,2,1)) for _ in range(4)])
        self.quality=nn.Sequential(nn.Linear(96,48),nn.GELU(),nn.Linear(48,2))
    def forward(self,x,c):
        z=self.trunk(torch.cat([F.normalize(x,dim=1)*np.sqrt(384),c],1));att=self.att(z);pred=[]
        for k in range(4):
            dim=2 if k%2==0 else 3
            v=(z*att[:,k:k+1].softmax(dim)).sum(dim);pred.append(self.sides[k](v))
        pred=torch.stack(pred,1)
        qual=self.quality(torch.cat([z.mean((2,3)),z.amax((2,3))],1))
        return pred[:,:,0],pred[:,:,1].tanh()/32,qual

class RegionalHead(SideHead):
    """Quality sees inside/outside/edge pools and explicit side/query geometry.

    Added after the plain global pooling head failed the synthetic stress test.
    Second output is P(query IoU >= .6), not regressed IoU.
    """
    def __init__(self):
        super().__init__()
        self.quality=nn.Sequential(nn.Linear(313,96),nn.GELU(),nn.Dropout(.1),nn.Linear(96,2))
    def forward(self,x,c):
        z=self.trunk(torch.cat([F.normalize(x,dim=1)*np.sqrt(384),c],1));att=self.att(z);pred=[]
        for k in range(4):
            dim=2 if k%2==0 else 3
            pred.append(self.sides[k]((z*att[:,k:k+1].softmax(dim)).sum(dim)))
        pred=torch.stack(pred,1);logits=pred[:,:,0];offset=pred[:,:,1].tanh()/32
        near=(c[:,:4].amin(1)>=-.25);inside=(c[:,:4].amin(1)>=0)
        masks=[inside,near&~inside]+[(c[:,k].abs()<.125)&near for k in range(4)]
        pooled=[(z*m[:,None]).sum((2,3))/m.sum((1,2)).clamp_min(1)[:,None] for m in masks]
        # The central derivatives of the relative-coordinate channels recover
        # the query exactly for the centered and bounded-jitter crops used here.
        wx=(1/32)/(c[:,0,16,17]-c[:,0,16,16]).clamp_min(.001)
        wy=(1/32)/(c[:,1,17,16]-c[:,1,16,16]).clamp_min(.001)
        query=torch.stack([16.5/32-wx*c[:,0,16,16],16.5/32-wy*c[:,1,16,16],16.5/32+wx*c[:,2,16,16],16.5/32+wy*c[:,3,16,16]],1)
        p=logits.softmax(-1).detach();bins=(torch.arange(32,device=x.device)+.5)/32
        mean=(p*bins).sum(-1);std=((p*(bins-mean[:,:,None]).square()).sum(-1)+1e-8).sqrt();entropy=-(p*p.clamp_min(1e-8).log()).sum(-1)/np.log(32)
        wh=(query[:,2:]-query[:,:2]).clamp_min(.03);diff=(mean-query)/torch.cat([wh,wh],1)
        inter=(torch.minimum(mean[:,2:],query[:,2:])-torch.maximum(mean[:,:2],query[:,:2])).clamp_min(0).prod(1)
        union=(mean[:,2:]-mean[:,:2]).clamp_min(0).prod(1)+wh.prod(1)-inter
        overlap=inter/union.clamp_min(1e-5)
        geom=torch.cat([query,mean,std,diff,entropy,c[:,4:,0,0],overlap[:,None]],1)
        qual=self.quality(torch.cat(pooled+[geom],1))
        return logits,offset,qual

def normbox(box,cb):return (np.asarray(box,float)-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]
def cond(box,cls):
    wh=np.maximum(box[2:]-box[:2],.03)
    d=np.stack([(xx-box[0])/wh[0],(yy-box[1])/wh[1],(box[2]-xx)/wh[0],(box[3]-yy)/wh[1]])
    one=np.zeros((4,32,32));one[cls]=1
    return np.concatenate([np.clip(d,-4,4),one]).astype(np.float32)

class Data:
    def __init__(self):
        r=pd.read_csv(V31/'inputs/dense_records.csv');r['bank']=0
        e=pd.read_csv(A/'inputs/empty_records.csv');e['bank']=1
        self.rec=pd.concat([r,e],ignore_index=True)
        self.feat=[np.load(V31/'features/dense_grids.npy',mmap_mode='r'),np.load(A/'features/empty_grids.npy',mmap_mode='r')]
        if (A/'features/pseudo_grids.npy').exists():
            pool=pd.read_csv(A/'inputs/pseudo_records.csv');pool['bank']=2
            self.rec=pd.concat([self.rec,pool],ignore_index=True);self.feat.append(np.load(A/'features/pseudo_grids.npy',mmap_mode='r'))
        if (A/'features/rare_grids.npy').exists():
            rare=pd.read_csv(A/'inputs/rare_records.csv');rare['bank']=len(self.feat)
            self.rec=pd.concat([self.rec,rare],ignore_index=True);self.feat.append(np.load(A/'features/rare_grids.npy',mmap_mode='r'))
        self.box=self.rec[B].to_numpy(float);self.cb=self.rec[CB].to_numpy(float)
        self.gt={};self.by_frame={f:{c:d[d.class_name.eq(c)][B].to_numpy(float) for c in CL} for f,d in h.groupby('frame_id')}
        for i,r in self.rec.iterrows():
            # TEST NEVER receives targets, even during target preprocessing.
            if r.split=='test' or r.split.startswith('pool_'):continue
            self.gt[i]=self.by_frame[r.frame_id]
    def features(self,ids):
        return torch.from_numpy(np.stack([self.feat[int(self.rec.iloc[i].bank)][int(self.rec.iloc[i].feature_index)].astype(np.float32) for i in ids]))
    def target(self,i,rng):
        r=self.rec.iloc[i];cb=self.cb[i];original=self.box[i]
        # Positive manual crops anchor an INSTANCE, avoiding a union of nearby hands.
        pseudo=r.split.startswith('pool_');presence_only=pseudo and r.split.endswith('presence');pos=r.split in ['manual','pool_geometry'] and r.class_name in CL
        cls=CL.index(r.class_name) if r.class_name in CL else int(rng.integers(4))
        if r.split in ['manual','pool_geometry']:
            b=normbox(original,cb);wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2+rng.normal(0,.2,2)*wh
            size=wh*np.exp(rng.normal(0,.32,2));query=np.r_[center-size/2,center+size/2]
            # Independent edge corruption models short people and clipped fingers.
            if rng.random()<.4:query[rng.integers(4)]+=rng.uniform(-.3,.3)*wh[rng.integers(2)]
            query=np.clip(query,0,1);query[2:]=np.maximum(query[2:],query[:2]+.02)
        else:query=normbox(original,cb)
        gt=self.gt[i][CL[cls]] if not pseudo else np.empty((0,4))
        pixelq=query*np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]+np.r_[cb[:2],cb[:2]]
        if pos or presence_only:target=original;present=1.
        elif len(gt):
            ov=iou(pixelq,gt).ravel();j=ov.argmax();target=gt[j]
            present=float(ov[j]>=.05)
        else:target=original;present=0.
        b=normbox(target,cb);valid=present and np.all(b>=0) and np.all(b<=1) and not presence_only
        quality=(float(iou(pixelq,target)[0,0]) if present else 0.) if not presence_only else -1.
        return cond(query,cls),np.clip(b,0,1).astype(np.float32),float(valid),np.array([present,quality],np.float32),np.float32(.3 if pseudo else 1.)

def loss_fn(out,target,regional=False):
    logits,off,qual=out;b,valid,y,rowweight=target;ymask=y>=0
    if regional:
        y=y.clone();y[:,1]=torch.sigmoid((y[:,1]-.6)/.04)*y[:,0]
    ce=F.binary_cross_entropy_with_logits(qual,y.clamp_min(0),reduction='none')
    loss=(ce*ymask*rowweight[:,None]).sum()/(ymask*rowweight[:,None]).sum().clamp_min(1)*(2 if regional else 1)
    m=valid>0
    if m.any():
        loc=(b[m]*32-.5).clamp(0,31);left=loc.floor().long();right=(left+1).clamp_max(31);fraction=loc-left
        lp=logits[m].log_softmax(-1)
        ce=-(lp.gather(2,left[:,:,None])[:,:,0]*(1-fraction)+lp.gather(2,right[:,:,None])[:,:,0]*fraction).mean(1)
        index=loc.round().long();res=b[m]-(index+.5)/32
        reg=F.smooth_l1_loss(off[m].gather(2,index[:,:,None])[:,:,0],res,beta=.02,reduction='none').mean(1)
        loss=loss+((ce+8*reg)*rowweight[m]).sum()/rowweight[m].sum().clamp_min(1)
    return loss

def decode(out,rows):
    logits,offset,quality=out;p=logits.softmax(-1).numpy();offset=offset.numpy();quality=quality.sigmoid().numpy();outrows=[]
    for j,(_,r) in enumerate(rows.iterrows()):
        cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];query=normbox(r[B].to_numpy(float),cb)
        modes=p[j].argmax(1);loc=(modes+.5)/32+offset[j,np.arange(4),modes]
        loc=np.clip(loc,0,1);valid=bool(np.all(loc[2:]>loc[:2]));ref=loc*np.r_[size,size]+np.r_[cb[:2],cb[:2]]
        mean=((np.arange(32)[None,:]+.5)/32*p[j]).sum(1)
        std=np.sqrt((((np.arange(32)[None,:]+.5)/32-mean[:,None])**2*p[j]).sum(1))
        # Distribution-derived IoU interval (not calibrated coverage).
        lo=np.array([np.searchsorted(np.cumsum(v),.1) for v in p[j]])/32
        hi=(np.array([min(31,np.searchsorted(np.cumsum(v),.9)) for v in p[j]])+1)/32
        # A query may be correct anywhere inside uncertain side intervals.
        optimistic=np.clip(query,lo,hi);optimistic[2:]=np.maximum(optimistic[2:],optimistic[:2]+1e-4)
        outrows.append(dict(id=r.id,split=r.split,frame_id=r.frame_id,day=r.day,class_name=r.class_name,
            presence=float(quality[j,0]),predicted_quality=float(quality[j,1]),ref_iou=float(iou(query,loc)[0,0]) if valid else 0.,valid=valid,
            optimistic_iou=float(iou(query,optimistic)[0,0]),mean_side_std=float(std.mean()),side_peak_min=float(p[j].max(1).min()),
            **{f'ref_{k}':v for k,v in zip(B,ref)},**{f'std_{k}':v for k,v in zip(B,std)}))
    return outrows,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--kinds',nargs='+',default=['manual','hardneg']);ap.add_argument('--fit-days',nargs='+',default=DAYS);args=ap.parse_args();data=Data();rec=data.rec
    path=A/'models/training_manifest.json';manifest=json.loads(path.read_text()) if path.exists() else []
    for fit in args.fit_days:
        for kind in args.kinds:
            torch.manual_seed(3302);rng=np.random.default_rng(3302)
            fit_mask=rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit)
            mask=fit_mask&rec.split.eq('manual')
            if 'hardneg' in kind:mask=fit_mask&rec.split.isin(['manual','empty','oof'])
            if 'pool' in kind:mask=fit_mask&rec.split.isin(['manual','empty','oof','pool_geometry','pool_presence'])
            if 'rare' in kind:mask|=fit_mask&rec.split.eq('pool_rare_presence')
            train=rec.index[mask].to_numpy();evaluation=rec.index[rec.split.eq('test')|(rec.day.ne(fit)&rec.split.isin(['oof','manual','empty']))].to_numpy()
            if fit=='all':evaluation=rec.index[rec.split.eq('test')].to_numpy()
            assert not set(rec.loc[train].frame_id)&set(rec.loc[evaluation].frame_id)
            # Explicit positive/negative mixture prevents empty frames dominating.
            groups=[train[rec.loc[train].split.eq('manual').to_numpy()&rec.loc[train].class_name.isin(CL).to_numpy()],
                    train[(~(rec.loc[train].split.eq('manual')&rec.loc[train].class_name.isin(CL))&~rec.loc[train].split.str.startswith('pool_')).to_numpy()]]
            draws=[6,2]
            if 'pool' in kind:
                groups += [train[rec.loc[train].split.eq(s).to_numpy()] for s in ['pool_geometry','pool_presence']];draws=[6,2,1,1]
            if 'rare' in kind:groups.append(train[rec.loc[train].split.eq('pool_rare_presence').to_numpy()]);draws.append(1)
            probs=[]
            for group in groups:
                if 'balanced' in kind and rec.loc[group].split.str.startswith('pool_').all():
                    rr=rec.loc[group];stamp=pd.to_datetime(rr.frame_id.str[-15:],format='%Y%m%d_%H%M%S').astype('int64')//(600*10**9)
                    keys=rr.class_name+'_'+stamp.astype(str)+'_'+(((rr.x1+rr.x2)/2)//80).astype(int).astype(str)+'_'+(((rr.y1+rr.y2)/2)//80).astype(int).astype(str)
                    counts=keys.value_counts();w=np.array([1/counts[k] for k in keys])
                else:
                    counts=rec.loc[group].groupby('frame_id').size();w=np.array([1/counts[f] for f in rec.loc[group].frame_id])
                probs.append(w/w.sum())
            regional='regional' in kind;model=RegionalHead() if regional else SideHead();opt=torch.optim.AdamW(model.parameters(),lr=.00035,weight_decay=.02);start=time.time();running=[]
            for step in range(args.steps):
                ids=np.concatenate([rng.choice(group,n,p=prob) for group,n,prob in zip(groups,draws,probs)])
                targets=[data.target(int(i),rng) for i in ids];c,b,v,y,w=[torch.from_numpy(np.stack([t[k] for t in targets])) for k in range(5)]
                loss=loss_fn(model(data.features(ids),c),(b,v,y,w),regional=regional);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();running.append(float(loss.detach()))
                if step%200==0:print('FIT',kind,fit,step,round(np.mean(running[-100:]),4),round(time.time()-start,1),flush=True)
            name=f'side_{kind}_fit{fit if fit=="all" else fit[-2:]}';torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();rows=[];maps=[];mapids=[]
            # Other-class manual crops are not evaluated as a positive class.
            evaluation=evaluation[rec.loc[evaluation].class_name.isin(CL).to_numpy()]
            for off in range(0,len(evaluation),16):
                ids=evaluation[off:off+16];rr=rec.loc[ids]
                c=torch.from_numpy(np.stack([cond(normbox(r[B].to_numpy(float),r[CB].to_numpy(float)),CL.index(r.class_name)) for _,r in rr.iterrows()]))
                with torch.inference_mode():pred=model(data.features(ids),c)
                out,p=decode(pred,rr);rows+=out;maps.extend(p.astype(np.float16));mapids.extend(rr.id)
            pd.DataFrame(rows).to_csv(A/f'features/{name}.csv',index=False);np.savez_compressed(A/f'features/{name}_distributions.npz',p=np.stack(maps),ids=np.array(mapids))
            manifest=[m for m in manifest if m['name']!=name]
            manifest.append(dict(name=name,fit_day=fit,fit_frames=sorted(rec.loc[train].frame_id.unique()),train_crops=len(train),steps=args.steps,seed=3302,seconds=time.time()-start,loss=float(np.mean(running[-100:])),test_labels_used=False,quality_target='soft probability of IoU>=.6' if regional else 'IoU',pseudo_sampling='class + ten-minute + 80px spatial bucket balanced' if 'balanced' in kind else 'frame balanced',training_split_counts=rec.loc[train].split.value_counts().to_dict()))
            (A/'models/training_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',name,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
