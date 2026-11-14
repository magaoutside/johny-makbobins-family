"""Query-conditioned masks, box targets versus projection/color-pair targets.

Small BoxInst-inspired loss ablation, NOT a reproduction of BoxInst.
Frozen DINOv2 features; all supervision comes from the fitting day's manual boxes.
Random query jitter prevents simply copying the input box. Test annotations unused.
"""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parent.parent;A=ROOT/'rascar_v32';V31=ROOT/'rascar_v31'
sys.path.insert(0,str(ROOT/'rascar_v23'))
from common23 import h,B,CL,DAYS,iou
from features23 import image
import numpy as np,pandas as pd,torch,cv2
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3);torch.set_num_interop_threads(1);cv2.setNumThreads(1)

class ConditionalMask(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(392,64,1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,1,1))
    def forward(self,x,condition):
        return self.net(torch.cat([F.normalize(x,dim=1)*np.sqrt(384),condition],1))[:,0]

yy,xx=np.mgrid[:32,:32];xx=(xx+.5)/32;yy=(yy+.5)/32
def normalized_box(r):
    cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float)
    return (r[B].to_numpy(float)-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]
def condition(box,cls):
    wh=np.maximum(box[2:]-box[:2],.03)
    d=np.stack([(xx-box[0])/wh[0],(yy-box[1])/wh[1],(box[2]-xx)/wh[0],(box[3]-yy)/wh[1]])
    one=np.zeros((4,32,32));one[cls]=1
    return np.concatenate([np.clip(d,-4,4),one]).astype(np.float32)
def dice(p,y):return 1-(2*(p*y).sum(-1)+1e-5)/(p.square().sum(-1)+y.square().sum(-1)+1e-5)
def loss_fn(logits,target,affinity,kind,step):
    if kind=='conditional_box':return F.binary_cross_entropy_with_logits(logits,target,weight=1+3*target)
    p=logits.sigmoid();proj=(dice(p.amax(1),target.amax(1))+dice(p.amax(2),target.amax(2))).mean()
    outside=(F.softplus(logits)*(1-target)).sum()/(1-target).sum().clamp_min(1)
    pair=0.
    for k,(dy,dx) in enumerate([(0,2),(2,0),(2,2),(2,-2)]):
        q=torch.roll(p,(dy,dx),(1,2));agreement=p*q+(1-p)*(1-q)
        w=affinity[:,k];pair+=(-(agreement.clamp_min(1e-6).log())*w).sum()/w.sum().clamp_min(1)
    return proj+.5*outside+.2*min(step/200,1)*pair/4

def decode(p,r):
    cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(float);size=cb[2:]-cb[:2];out=[]
    for cut in [.35,.5,.65]:
        n,lab,stats,_=cv2.connectedComponentsWithStats((p>=cut).astype(np.uint8),8)
        for j in range(1,n):
            x,y,w,hh,area=stats[j]
            if area<4:continue
            box=np.r_[x,y,x+w,y+hh]/32*np.r_[size,size]+np.r_[cb[:2],cb[:2]]
            out.append(dict(class_name=r.class_name,threshold=cut,confidence=float(p[lab==j].mean()),**dict(zip(B,box))))
    return out

def main():
    rec=pd.read_csv(V31/'inputs/dense_records.csv');features=np.load(V31/'features/dense_grids.npy',mmap_mode='r')
    targets={};affinities={};norm={}
    for i,r in rec[rec.split.eq('manual')].iterrows():
        box=normalized_box(r);norm[i]=box
        y=((xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])).astype(np.float32) if r.class_name in CL else np.zeros((32,32),np.float32)
        cb=r[['crop_x1','crop_y1','crop_x2','crop_y2']].to_numpy(int);im=image(r.frame_id)[cb[1]:cb[3],cb[0]:cb[2]]
        lab=cv2.cvtColor(cv2.resize(im.astype(np.float32)/255,(32,32),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2LAB)
        pairs=[]
        for dy,dx in [(0,2),(2,0),(2,2),(2,-2)]:
            same=np.linalg.norm(lab-np.roll(lab,(dy,dx),(0,1)),axis=2)<10
            valid=np.ones((32,32),bool)
            if dy:valid[:dy]=False
            if dx>0:valid[:,:dx]=False
            if dx<0:valid[:,dx:]=False
            pairs.append((same&valid&(y>0)).astype(np.float32))
        targets[i]=y;affinities[i]=np.stack(pairs)
    logs=[]
    for fit in DAYS:
        train=rec[rec.split.eq('manual')&rec.day.eq(fit)].index.to_numpy()
        evaluation=rec[rec.split.eq('test')|((rec.split=='oof')&rec.day.ne(fit))|((rec.split=='manual')&rec.day.ne(fit)&rec.class_name.isin(CL))].index.to_numpy()
        assert not set(rec.loc[train,'frame_id'])&set(rec.loc[evaluation[rec.loc[evaluation,'split'].ne('test')],'frame_id'])
        counts=rec.loc[train].groupby('frame_id').size();weights=np.array([1/counts[f] for f in rec.loc[train].frame_id]);weights/=weights.sum()
        for kind in ['conditional_box','conditional_projection']:
            torch.manual_seed(3213);rng=np.random.default_rng(3213);model=ConditionalMask();opt=torch.optim.AdamW(model.parameters(),lr=.00035,weight_decay=.01);running=[];start=time.time()
            for step in range(1000):
                ids=rng.choice(train,8,replace=True,p=weights);conds=[]
                for i in ids:
                    r=rec.loc[i];b=norm[i];wh=b[2:]-b[:2];center=(b[:2]+b[2:])/2+rng.normal(0,.18,2)*wh;size=wh*np.exp(rng.normal(0,.3,2));query=np.r_[center-size/2,center+size/2]
                    cls=CL.index(r.class_name) if r.class_name in CL else int(rng.integers(0,4));conds.append(condition(query,cls))
                x=torch.from_numpy(np.array(features[ids],dtype=np.float32));c=torch.from_numpy(np.stack(conds));y=torch.from_numpy(np.stack([targets[i] for i in ids]));aff=torch.from_numpy(np.stack([affinities[i] for i in ids]))
                loss=loss_fn(model(x,c),y,aff,kind,step);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();running.append(float(loss.detach()))
                if step%200==0:print('FIT',kind,fit,step,round(np.mean(running[-100:]),4),round(time.time()-start,1),flush=True)
            name=f'{kind}_fit{fit[-2:]}';torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();out=[];maps=[];mapids=[]
            for off in range(0,len(evaluation),16):
                ids=evaluation[off:off+16];conds=np.stack([condition(normalized_box(rec.loc[i]),CL.index(rec.loc[i].class_name)) for i in ids])
                with torch.inference_mode():p=model(torch.from_numpy(np.array(features[ids],dtype=np.float32)),torch.from_numpy(conds)).sigmoid().numpy()
                for j,i in enumerate(ids):
                    r=rec.loc[i]
                    for b in decode(p[j],r):out.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,**b))
                    if r.split in ['oof','test']:maps.append(p[j].astype(np.float16));mapids.append(r.id)
            pd.DataFrame(out).to_csv(A/f'features/{name}_boxes.csv',index=False);np.savez_compressed(A/f'features/{name}_maps.npz',maps=np.stack(maps),ids=np.array(mapids))
            logs.append(dict(name=name,fit_day=fit,fit_frames=sorted(rec.loc[train].frame_id.unique().tolist()),steps=1000,last_loss=float(np.mean(running[-100:])),seconds=time.time()-start,seed=3213))
            (A/'models/conditional_training_manifest.json').write_text(json.dumps(logs,indent=2));print('DONE',name,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
