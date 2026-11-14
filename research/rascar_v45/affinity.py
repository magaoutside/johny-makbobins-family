"""Cross-day instance affinity on frozen DINO features; rectangles are weak masks.

Learn pixel identities from manual same-class instances; ignore overlapping GT
rectangles. The decoder compares independent BPJ anchor embeddings, never query
coordinates. Not true instance-mask supervision and not a new object detector.
"""
from common45 import *
from decompose import coverage,split_map
import torch,cv2,time
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3);cv2.setNumThreads(1)
(A/'models').mkdir(exist_ok=True)
class Affinity(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(384,64,1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,32,3,padding=1))
    def forward(self,x):return F.normalize(self.net(F.normalize(x,dim=1)*np.sqrt(384)),dim=1)

def manual_labels(r,h):
    cb=r[CB].to_numpy(float);wh=cb[2:]-cb[:2];maps=[]
    for cls in ['person','nude_hands']:
        bb=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(cls)][B].to_numpy(float)
        if not len(bb):continue
        box=(bb-np.tile(cb[:2],2))/np.tile(wh,2)*32
        cov=coverage(box);inside=cov>.95;count=inside.sum(0)
        labels=np.where(count==1,inside.argmax(0)+1,0)
        if (np.bincount(labels.ravel(),minlength=len(bb)+1)[1:]>=6).sum()>=2:maps.append(labels)
    return maps

def decode(prob,cb,boxes,points,emb,weight):
    size=cb[2:]-cb[:2];bb=(boxes-np.tile(cb[:2],2))/np.tile(size,2)*32;gp=(points-cb[:2])/size*32
    n,labels,stats,_=cv2.connectedComponentsWithStats((prob>=.5).astype(np.uint8),8);out=[]
    for j in range(1,n):
        x,y,w,hh,area=stats[j]
        if area<4:continue
        old=np.array([x,y,x+w,y+hh],float);mask=labels==j;padded=cv2.dilate(mask.astype(np.uint8),np.ones((3,3),np.uint8));ix=np.floor(gp).astype(int)
        ids=[k for k in range(len(gp)) if ((ix[k]>=0)&(ix[k]<32)).all() and padded[ix[k,1],ix[k,0]]]
        if len(ids)<2 or len(ids)>5:out.append((old,False,0,float(prob[mask].mean())));continue
        prototypes=[]
        for k in ids:
            ax,ay=ix[k];p=emb[:,max(ay-1,0):min(ay+2,32),max(ax-1,0):min(ax+2,32)].mean((1,2));prototypes.append(p/max(np.linalg.norm(p),1e-9))
        cos=np.einsum('nc,cyx->nyx',np.array(prototypes),emb);yy,xx=np.mgrid[:32,:32];xy=np.stack([xx+.5,yy+.5],-1)
        center=(bb[ids,:2]+bb[ids,2:])/2;scale=np.maximum(bb[ids,2:]-bb[ids,:2],2)
        spatial=(((xy[None]-center[:,None,None])/scale[:,None,None])**2).sum(-1)
        assignment=(spatial+weight*(1-cos)).argmin(0)
        for k in range(len(ids)):
            ys,xs=np.where(mask&(assignment==k))
            if len(xs)<4:continue
            out.append((np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1]),True,len(ids),float(prob[mask].mean())))
    return [(b/32*np.tile(size,2)+np.tile(cb[:2],2),change,n,conf) for b,change,n,conf in out]

def main():
    q,te,h=inputs();rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');feat=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');heads=load_jsonl('head');hands=load_jsonl('hand_local');lookup=rec.reset_index().set_index('id')['index'].to_dict()
    targets={i:manual_labels(r,h) for i,r in rec[rec.split.eq('manual')].iterrows()};manifest=[];output=[]
    device='mps' if torch.backends.mps.is_available() else 'cpu';print('DEVICE',device,flush=True)
    for fit in DAYS:
        train=[i for i in targets if rec.day.iloc[i]==fit and targets[i]]
        assert not set(rec.iloc[train].frame_id)&set(q[q.day.ne(fit)].frame_id)
        torch.manual_seed(4502);rng=np.random.default_rng(4502);model=Affinity().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.00025,weight_decay=.05);losses=[];start=time.time()
        frame_counts=rec.iloc[train].groupby('frame_id').size();sampling=np.array([1/frame_counts[rec.frame_id.iloc[i]] for i in train]);sampling/=sampling.sum()
        for step in range(400):
            idx=rng.choice(train,4,replace=True,p=sampling);z=model(torch.from_numpy(np.array(feat[idx],np.float32)).to(device));terms=[]
            for k,i in enumerate(idx):
                label=targets[i][rng.integers(len(targets[i]))];classes=[c for c in np.unique(label) if c>0 and (label==c).sum()>=6]
                pixels=[];groups=[]
                for c in classes:
                    pos=np.flatnonzero(label.ravel()==c);chosen=rng.choice(pos,min(24,len(pos)),replace=False);pixels.extend(chosen);groups.extend([c]*len(chosen))
                zz=z[k].flatten(1)[:,pixels].T;sim=zz@zz.T/.12;y=torch.tensor(groups,device=device);positive=y[:,None].eq(y[None]);diag=torch.eye(len(y),dtype=torch.bool,device=device);positive=positive&~diag
                # Supervised contrastive instance discrimination within one image/class.
                logprob=F.log_softmax(sim.masked_fill(diag,-1e4),dim=1);terms.append(-(logprob*positive).sum(1).div(positive.sum(1).clamp_min(1)).mean())
            loss=torch.stack(terms).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();losses.append(float(loss.detach().cpu()))
            if step%100==0:print('FIT',fit,step,round(np.mean(losses[-100:]),4),round(time.time()-start,1),flush=True)
        torch.save({k:v.cpu() for k,v in model.state_dict().items()},A/f'models/affinity_fit{fit[-2:]}.pt');model.eval();manifest.append(dict(fit_day=fit,steps=400,seed=4502,fit_frame_ids=sorted(rec.iloc[train].frame_id.unique()),crops=len(train),last_loss=float(np.mean(losses[-100:])),device=device))
        data=np.load(ROOT/f'rascar_v31/features/linear_mask_fit{fit[-2:]}_maps.npz');ml={s:i for i,s in enumerate(data['ids'])};rows=pd.concat([q[q.day.ne(fit)].assign(split='oof'),te.assign(split='test')],ignore_index=True)
        for _,r in rows.iterrows():
            if r.class_name not in ['person','nude_hands']:continue
            i=lookup[r.id];cb=rec.loc[i,CB].to_numpy(float);bb,pp,_=anchors(r.frame_id,r.class_name,heads,hands)
            if len(bb)<2:continue
            x=torch.from_numpy(np.array(feat[i:i+1],np.float32)).to(device)
            with torch.inference_mode():learned=model(x)[0].cpu().numpy();frozen=F.normalize(x,dim=1)[0].cpu().numpy()
            prob=data['maps'][ml[r.id],CL.index(r.class_name)].astype(float)
            for kind,emb in [('frozen',frozen),('learned',learned)]:
                for weight in [.5,1.]:
                    method=f'{kind}_affinity{weight:g}';refs=decode(prob,cb,bb,pp,emb,weight)
                    if not refs:continue
                    ov=iou(r[B],np.array([v[0] for v in refs]))[0];j=ov.argmax();box,changed,count,conf=refs[j]
                    output.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,split=r.split,fit_day=fit,class_name=r.class_name,method=method,overlap=ov[j],modified=changed,anchors=count,confidence=conf,**{'ref_'+b:v for b,v in zip(B,box)}))
    (A/'models/affinity_manifest.json').write_text(json.dumps(manifest,indent=2));pd.DataFrame(output).to_csv(A/'features/affinity.csv',index=False)
    original=pd.read_csv(A/'features/decomposition.csv');original=original[~original.method.str.contains('affinity')];pd.concat([original,pd.DataFrame(output)],ignore_index=True).to_csv(A/'features/decomposition.csv',index=False)
if __name__=='__main__':main()
