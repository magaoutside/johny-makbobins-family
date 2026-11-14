"""Mask-conditioned ResNet18 box-quality head: clean vs pseudo-augmented.
Fixed optimizer budget, fitting-day-only annotations, frozen early backbone.
"""
from pathlib import Path
import os,sys,json,random,hashlib,gc
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,pandas as pd,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision.models import resnet18,ResNet18_Weights
from PIL import Image,ImageDraw
from functools import lru_cache
A=Path('/kaggle/working/rascar_v20');D=Path('/kaggle/working/competition_data')
fit,device=sys.argv[1:3];device=int(device);held='2026-10-26' if fit.endswith('25') else '2026-10-25'
CL=['person','nude_hands','cup','tray'];B=['x1','y1','x2','y2'];torch.set_num_threads(2)
paths={p.stem:p for p in D.rglob('*.jpg')};out=A/'quality'/fit;out.mkdir(parents=True,exist_ok=True)
manual=pd.read_csv(A/'inputs/manual_boxes.csv');manual=manual[manual.class_name.isin(CL)]
queries=pd.read_csv(A/'inputs/baseline_oof.csv');test=pd.read_csv(A/'inputs/test_queries.csv')
days=['2026-10-25','2026-10-26'] if fit=='all' else [fit]
pseudo=pd.concat([pd.read_csv(A/'pool'/day/'approved_pseudo_boxes.csv') for day in days],ignore_index=True)
machine=pd.read_csv(A/'inputs/machine_predictions.csv')
def iou(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4);inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(1)[:,None]+(b[:,2:]-b[:,:2]).prod(1)[None,:]-inter,1e-9)
@lru_cache(256)
def imread(fid):return Image.open(paths[fid]).convert('RGB').resize((1280,720),Image.Resampling.BICUBIC)

def records_from(gt,source):
    out=[];pseudo_source=source=='pseudo'
    for idx,r in enumerate(gt.itertuples()):
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;variants=[('exact',b)]
        for scale in [.6,.8,1.2,1.5]:variants.append((f'scale_{scale}',np.r_[cc-wh*scale/2,cc+wh*scale/2]))
        for k,delta in enumerate([[-.45,0],[.45,0],[0,-.45],[0,.45]]):variants.append((f'shift_{k}',b+np.tile(wh*delta,2)))
        if r.class_name=='person':
            for frac in [.35,.6]:
                bb=b.copy();bb[3]-=wh[1]*frac;variants.append((f'short_{frac}',bb))
        if r.class_name=='nude_hands':
            for k in range(4):
                bb=b.copy();bb[k]+=(-1 if k<2 else 1)*wh[k%2];variants.append((f'extension_{k}',bb))
        semantic_only=pseudo_source and r.class_name=='person'
        if semantic_only:variants=[('semantic_only',b)]
        peers=gt[gt.frame_id.eq(r.frame_id)&gt.class_name.eq(r.class_name)][B].to_numpy()
        for j,bb in enumerate(peers):
            if np.allclose(bb,b):continue
            if not semantic_only and np.linalg.norm((bb[:2]+bb[2:])/2-cc)<1.4*max(wh.max(),(bb[2:]-bb[:2]).max()):variants.append((f'merged_{j}',np.r_[np.minimum(b[:2],bb[:2]),np.maximum(b[2:],bb[2:])]))
        unknown=machine[machine.frame_id.eq(r.frame_id)&machine.class_name.eq(r.class_name)] if pseudo_source else None
        if pseudo_source:unknown=unknown[~unknown.detection_id.isin(gt.id)]
        for kind,bb in variants:
            bb=np.clip(bb,0,[1280,720,1280,720]);area=np.prod(bb[2:]-bb[:2])
            if min(bb[2:]-bb[:2])<6:continue
            # Never train a negative across an unverified same-class proposal.
            if pseudo_source and len(unknown) and iou(bb,unknown[B]).max()>.1:continue
            target=float(iou(bb,peers).max());pc=(peers[:,:2]+peers[:,2:])/2;multiple=int((((pc>bb[:2])&(pc<bb[2:])).all(1)).sum()>1)
            out.append(dict(id=f'{source}_{idx}_{kind}',frame_id=r.frame_id,day=r.day,class_name=r.class_name,source=source,kind=kind,target_iou=target,error=int(target<.6),multiple=multiple,presence=float(target>.05),geometry_safe=float(not semantic_only),weight=1. if kind in ['exact','semantic_only'] else .4,**dict(zip(B,bb))))
    return pd.DataFrame(out)
clean=records_from(manual[manual.day.isin(days)],'manual');extra=records_from(pseudo,'pseudo') if len(pseudo) else clean.iloc[:0].copy()
real=queries[queries.day.isin(days)].copy();real['source']='manual';real['kind']='real_query';real['weight']=2.
real['target_iou']=[float(iou(r[B].to_numpy(float),manual[manual.frame_id.eq(r.frame_id)&manual.class_name.eq(r.class_name)][B]).max()) if len(manual[manual.frame_id.eq(r.frame_id)&manual.class_name.eq(r.class_name)]) else 0. for _,r in real.iterrows()];real['multiple']=0
real['presence']=(real.target_iou>.05).astype(float);real['geometry_safe']=1.
clean=pd.concat([clean,real],ignore_index=True);clean.to_csv(out/'clean_records.csv',index=False);extra.to_csv(out/'pseudo_records.csv',index=False)

class Crops(Dataset):
    def __init__(self,rows,augment=False):self.rows=rows.reset_index(drop=True);self.augment=augment
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i];b=r[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;side=max(96,float(wh.max()*2))
        if self.augment:side*=np.random.uniform(.9,1.15)
        crop=np.r_[cc-side/2,cc+side/2];im=imread(r.frame_id).crop(tuple(crop)).resize((224,224),Image.Resampling.BILINEAR)
        mask=Image.new('L',(224,224),0);bb=(b-crop[[0,1,0,1]])/side*224;ImageDraw.Draw(mask).rectangle(tuple(bb),fill=255)
        rgb=np.asarray(im,dtype=np.float32)/255;m=np.asarray(mask,dtype=np.float32)/255
        if self.augment:
            if np.random.rand()<.5:rgb=rgb[:,::-1].copy();m=m[:,::-1].copy()
            rgb=np.clip(rgb*np.random.uniform(.85,1.15),0,1)
        rgb=(rgb-[.485,.456,.406])/[.229,.224,.225];x=np.concatenate([rgb,m[:,:,None]],axis=2).transpose(2,0,1).astype(np.float32)
        target=np.array([r.get('error',0),r.get('target_iou',0),r.get('multiple',0),r.get('presence',0),r.get('geometry_safe',1)],dtype=np.float32)
        return torch.from_numpy(x),CL.index(r.class_name),torch.from_numpy(np.nan_to_num(target)),float(r.get('weight',1))

class Quality(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        old=self.backbone.conv1;new=nn.Conv2d(4,64,7,2,3,bias=False)
        with torch.no_grad():new.weight[:,:3].copy_(old.weight);new.weight[:,3].zero_()
        self.backbone.conv1=new;self.backbone.fc=nn.Linear(512,16)
        for name,p in self.backbone.named_parameters():p.requires_grad_(name.startswith(('conv1','layer4','fc')))
    def forward(self,x):return self.backbone(x).reshape(-1,4,4)

def seed():random.seed(201);np.random.seed(201);torch.manual_seed(201);torch.cuda.manual_seed_all(201)
device_name=f'cuda:{device}'
for variant in ['clean','expanded']:
    seed();train=clean.copy() if variant=='clean' else pd.concat([clean,extra],ignore_index=True)
    if variant=='expanded' and len(extra)==0:print('NO_PSEUDO_EXPANSION',fit,flush=True)
    sampler_weight=train.weight.to_numpy(float)
    if variant=='expanded' and len(extra):
        p=train.source.eq('pseudo').to_numpy();sampler_weight[p]*=.3/max(sampler_weight[p].sum(),1);sampler_weight[~p]*=.7/sampler_weight[~p].sum()
    loader=DataLoader(Crops(train,True),batch_size=48,sampler=WeightedRandomSampler(sampler_weight,48*400,replacement=True),num_workers=2,pin_memory=True)
    model=Quality().to(device_name);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1.5e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');losses=[]
    for step,(x,c,y,w) in enumerate(loader):
        model.train()
        # Keep pretrained normalization fixed on this small dataset.
        for module in model.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        x=x.to(device_name);c=c.to(device_name);y=y.to(device_name);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            p=model(x)[torch.arange(len(x),device=device_name),c]
            geom=nn.functional.binary_cross_entropy_with_logits(p[:,0],y[:,0],reduction='none')+.5*nn.functional.smooth_l1_loss(p[:,1].sigmoid(),y[:,1],reduction='none')+.2*nn.functional.binary_cross_entropy_with_logits(p[:,2],y[:,2],reduction='none')
            loss=(geom*y[:,4]).sum()/y[:,4].sum().clamp(min=1)+.3*nn.functional.binary_cross_entropy_with_logits(p[:,3],y[:,3])
        scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.item()))
        if step%50==0:print('QUALITY',fit,variant,step,round(np.mean(losses[-50:]),4),flush=True)
    torch.save({'state_dict':model.state_dict(),'classes':CL,'fit_days':days,'variant':variant,'steps':400,'seed':201},out/f'{variant}.pt')
    pd.DataFrame(dict(step=np.arange(len(losses)),loss=losses)).to_csv(out/f'{variant}_loss.csv',index=False)
    model.eval()
    for split,rr in [('held',queries[queries.day.eq(held)] if fit!='all' else queries.iloc[:0]),('test',test)]:
        if len(rr)==0:continue
        pp=[]
        with torch.inference_mode():
            for x,c,_,_ in DataLoader(Crops(rr),batch_size=64,num_workers=2):
                x=x.to(device_name);c=c.to(device_name)
                with torch.autocast('cuda',dtype=torch.float16):p=model(x)[torch.arange(len(x),device=device_name),c].sigmoid()
                pp.append(p.float().cpu().numpy())
        pp=np.concatenate(pp);rr[['id']].assign(quality_error=pp[:,0],quality_iou=pp[:,1],multiple=pp[:,2],presence=pp[:,3]).to_csv(out/f'{variant}_{split}.csv',index=False)
    del model;gc.collect();torch.cuda.empty_cache()
(out/'manifest.json').write_text(json.dumps(dict(fit=fit,clean_records=len(clean),pseudo_records=len(extra),pseudo_frames=pseudo.frame_id.nunique(),steps=400,batch=48,architecture='ImageNet ResNet18; 4-channel RGB+query-mask; separate error/IoU/multiple heads by class',manual_pseudo_sampling=[.7,.3],held_day_in_training=False,test_labels_used=False),indent=2))
print('QUALITY_DONE',fit,flush=True)
