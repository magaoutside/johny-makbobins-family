"""Small set decoder on frozen DINOv2, inspired by DETR's matching objective.

Not a pretrained DETR or reproduction. Predicts up to 32 overlapping instances
jointly, with one-to-one training assignment and no instance-erasing NMS.
Only official manual boxes of the fitting day supervise the head.
"""
from common39 import *
import torch,time,argparse,math
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(3)
CB=['crop_'+b for b in B];SEEDS=[3901,3902,3903]

def xyxy(box):
    return torch.cat([box[...,:2]-box[...,2:]/2,box[...,:2]+box[...,2:]/2],-1)

def generalized_iou(a,b):
    lo=torch.maximum(a[:,None,:2],b[None,:,:2]);hi=torch.minimum(a[:,None,2:],b[None,:,2:]);intersection=(hi-lo).clamp_min(0).prod(-1)
    aa=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1);bb=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1);union=(aa[:,None]+bb[None]-intersection).clamp_min(1e-8)
    cover=(torch.maximum(a[:,None,2:],b[None,:,2:])-torch.minimum(a[:,None,:2],b[None,:,:2])).clamp_min(0).prod(-1).clamp_min(1e-8)
    return intersection/union-(cover-union)/cover

class SetHead(nn.Module):
    def __init__(self):
        super().__init__();dim=96
        self.project=nn.Sequential(nn.Conv2d(384,dim,1),nn.GroupNorm(8,dim),nn.GELU())
        self.queries=nn.Parameter(torch.randn(32,dim)*.02)
        self.layers=nn.ModuleList([nn.TransformerDecoderLayer(dim,4,192,dropout=0,batch_first=True,norm_first=True) for _ in range(2)])
        self.norm=nn.LayerNorm(dim);self.cls=nn.Linear(dim,5);self.box=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,4))
        yy,xx=torch.meshgrid((torch.arange(16)+.5)/16,(torch.arange(16)+.5)/16,indexing='ij')
        freq=2*math.pi*2**torch.arange(24).float().div(4)
        pe=torch.cat([(xx[...,None]*freq).sin(),(xx[...,None]*freq).cos(),(yy[...,None]*freq).sin(),(yy[...,None]*freq).cos()],-1)
        self.register_buffer('position',pe.reshape(1,256,dim))
    def forward(self,x):
        x=F.avg_pool2d(F.normalize(x,dim=1)*math.sqrt(384),2)
        memory=self.project(x).flatten(2).transpose(1,2)+self.position
        z=self.queries[None].expand(len(x),-1,-1);out=[]
        for layer in self.layers:
            z=layer(z,memory);v=self.norm(z);out.append((self.cls(v),self.box(v).sigmoid()))
        return out

class Data:
    def __init__(self):
        r=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').assign(bank=0)
        e=pd.read_csv(ROOT/'rascar_v33/inputs/empty_records.csv').assign(bank=1)
        self.rec=pd.concat([r,e],ignore_index=True)
        self.banks=[np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r'),np.load(ROOT/'rascar_v33/features/empty_grids.npy',mmap_mode='r')]
        self.targets={};self.metadata=[]
        for i,row in self.rec.iterrows():
            if row.split=='test':continue
            cb=row[CB].to_numpy(float);size=cb[2:]-cb[:2];truth=h[h.frame_id.eq(row.frame_id)&h.class_name.isin(CL)]
            boxes=truth[B].to_numpy(float);lo=np.maximum(boxes[:,:2],cb[:2]);hi=np.minimum(boxes[:,2:],cb[2:]);inside=(hi-lo).clip(0).prod(1)>4
            norm=(np.c_[lo,hi][inside]-np.r_[cb[:2],cb[:2]])/np.r_[size,size]
            full=boxes[inside];labels=np.array([CL.index(c) for c in truth.loc[inside,'class_name']])
            # Train visible crop parts, but inference abstains at an internal crop boundary.
            wh=norm[:,2:]-norm[:,:2];cc=(norm[:,2:]+norm[:,:2])/2
            assert len(norm)<=32
            self.targets[i]=(torch.tensor(labels,dtype=torch.long),torch.tensor(np.c_[cc,wh],dtype=torch.float32))
            self.metadata.append(dict(record_id=row.id,split=row.split,day=row.day,frame_id=row.frame_id,instances=len(norm),partial_instances=int(((full[:,:2]<cb[:2])|(full[:,2:]>cb[2:])).any(1).sum())))
    def features(self,ids):
        return torch.tensor(np.stack([self.banks[int(self.rec.iloc[i].bank)][int(self.rec.iloc[i].feature_index)].astype(np.float32) for i in ids]))

def set_loss(outputs,targets):
    losses=[]
    for logits,boxes in outputs:
        classes=torch.full(logits.shape[:2],4,dtype=torch.long);p=logits.softmax(-1);loc=[];over=[]
        for i,(labels,target) in enumerate(targets):
            if not len(labels):continue
            with torch.no_grad():
                cost=5*torch.cdist(boxes[i],target,p=1)-2*generalized_iou(xyxy(boxes[i]),xyxy(target))-p[i,:,labels]
                ii,jj=linear_sum_assignment(cost.detach().numpy())
            classes[i,ii]=labels[jj];loc.append(F.l1_loss(boxes[i,ii],target[jj],reduction='sum'))
            over.append((1-generalized_iou(xyxy(boxes[i,ii]),xyxy(target[jj])).diag()).sum())
        count=max(sum(len(x[0]) for x in targets),1)
        ce=F.cross_entropy(logits.transpose(1,2),classes,weight=torch.tensor([1.,1.,1.,1.,.1]))
        loss=ce+(5*sum(loc)+2*sum(over))/count if loc else ce
        losses.append(loss)
    return losses[-1]+.5*sum(losses[:-1])

def decode(logits,boxes,row):
    p=logits.softmax(-1).numpy();box=xyxy(boxes).numpy();cb=row[CB].to_numpy(float);size=cb[2:]-cb[:2];out=[]
    for slot,(prob,b) in enumerate(zip(p,box)):
        k=int(prob.argmax());confidence=float(prob[k])
        if k==4 or confidence<.1:continue
        b=np.clip(b,0,1);pixel=b*np.r_[size,size]+np.r_[cb[:2],cb[:2]]
        internal=np.array([cb[0]>0,cb[1]>0,cb[2]<1280,cb[3]<720])
        edge=np.r_[b[:2]<.02,b[2:]>.98]&internal
        if (pixel[2:]-pixel[:2]).min()<1:continue
        out.append(dict(slot=slot,class_name=CL[k],confidence=confidence,crop_edge=bool(edge.any()),**dict(zip(B,pixel))))
    return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=1000);parser.add_argument('--seeds',nargs='+',type=int,default=SEEDS);parser.add_argument('--run-name',default='set');parser.add_argument('--cosine',action='store_true');parser.add_argument('--fit-both',action='store_true');args=parser.parse_args();data=Data();rec=data.rec
    fits=['all'] if args.fit_both else DAYS
    pd.DataFrame(data.metadata).to_csv(A/'audit/set_target_manifest.csv',index=False)
    (A/f'inputs/{args.run_name}_protocol.json').write_text(json.dumps(dict(seeds=args.seeds,steps=args.steps,dim=96,decoder_layers=2,slots=32,cosine_schedule=args.cosine,train_splits=['manual','empty'],fit_days=fits,confidence=.5,blend=.25,early_stopping=False,test_annotations_used=False,source='https://github.com/facebookresearch/detr',weights='https://huggingface.co/facebook/dinov2-small'),indent=2))
    logs=[]
    for fit in fits:
        tr=rec.index[(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))&rec.split.isin(['manual','empty'])].to_numpy()
        evaluation=rec.split.eq('test') if fit=='all' else rec.split.eq('test')|(rec.day.ne(fit)&rec.split.isin(['manual','oof']))
        ev=rec.index[evaluation].to_numpy()
        assert not set(rec.loc[tr,'frame_id'])&set(rec.loc[ev,'frame_id'])
        groups=[tr[rec.loc[tr,'split'].eq('manual').to_numpy()],tr[rec.loc[tr,'split'].eq('empty').to_numpy()]]
        weights=[]
        for group in groups:
            count=rec.loc[group].groupby('frame_id').size();w=np.array([1/count[f] for f in rec.loc[group,'frame_id']]);weights.append(w/w.sum())
        for seed in args.seeds:
            name=f'{args.run_name}_fit{fit if fit=="all" else fit[-2:]}_seed{seed}';torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SetHead();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);losses=[];start=time.time()
            for step in range(args.steps):
                ids=np.r_[rng.choice(groups[0],6,p=weights[0]),rng.choice(groups[1],2,p=weights[1])]
                if args.cosine:
                    for param in opt.param_groups:param['lr']=.0003*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
                loss=set_loss(model(data.features(ids)),[data.targets[i] for i in ids]);assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                if step%200==0:print('TRAIN',name,step,round(np.mean(losses[-100:]),4),round(time.time()-start,1),flush=True)
            torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();out=[]
            for offset in range(0,len(ev),16):
                ids=ev[offset:offset+16]
                with torch.inference_mode():logits,boxes=model(data.features(ids))[-1]
                for j,i in enumerate(ids):
                    row=rec.loc[i]
                    for z in decode(logits[j],boxes[j],row):out.append(dict(record_id=row.id,frame_id=row.frame_id,split=row.split,fit_day=fit,seed=seed,**z))
            pd.DataFrame(out).to_csv(A/f'features/{name}_boxes.csv',index=False)
            logs.append(dict(name=name,fit_day=fit,seed=seed,steps=args.steps,fit_frames=sorted(rec.loc[tr,'frame_id'].unique()),manual_crops=len(groups[0]),empty_crops=len(groups[1]),last_loss=float(np.mean(losses[-100:])),seconds=time.time()-start))
            (A/f'models/{args.run_name}_manifest.json').write_text(json.dumps(logs,indent=2));print('DONE',name,len(out),round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
