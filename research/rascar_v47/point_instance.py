"""Point-conditioned single-instance box occupancy on frozen DINOv2-S.

Independent BPJ points prompt an instance, rather than splitting a union mask.
Query bounds never supervise/influence the individual reference refinement;
they only set the fixed V31 context crop and select the final reference by IoU.
This is a custom small head, not a reproduction of SAM/CondInst.
"""
from pathlib import Path
import sys,json,time,argparse
import numpy as np,pandas as pd,torch,cv2
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v45'))
from common45 import inputs,baseline,iou,metrics,B,CL,DAYS,CB,load_jsonl,anchors,v22base,compose
for p in ['inputs','features','models','logs','results/oof','results/submissions','audit']:(A/p).mkdir(parents=True,exist_ok=True)
CLASS=['person','nude_hands']
class PointHead(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Sequential(nn.Conv2d(384,32,1),nn.GroupNorm(8,32),nn.GELU())
        self.head=nn.Sequential(nn.Conv2d(32*2+3+2,32,1),nn.GELU(),nn.Conv2d(32,16,3,padding=1),nn.GELU(),nn.Conv2d(16,1,1))
    def forward(self,x,point,classes):
        z=self.project(F.normalize(x,dim=1)*np.sqrt(384));n,_,h,w=z.shape
        token=F.grid_sample(z,point[:,None,None]*2-1,align_corners=False,padding_mode='border').expand(-1,-1,h,w)
        yy,xx=torch.meshgrid((torch.arange(h,device=x.device)+.5)/h,(torch.arange(w,device=x.device)+.5)/w,indexing='ij')
        rel=torch.stack([xx,yy])[None]-point[:,:,None,None];dist=rel.square().sum(1,keepdim=True)
        cls=F.one_hot(classes,2).float()[:,:,None,None].expand(-1,-1,h,w)
        return self.head(torch.cat([z,token,rel,dist,cls],1))[:,0]
def norm(b,cb):return (b-np.tile(cb[:2],2))/np.tile(cb[2:]-cb[:2],2)
def occupancy(box):
    xy=np.arange(32)/32;wx=np.maximum(0,np.minimum(xy+1/32,box[2])-np.maximum(xy,box[0]))*32;wy=np.maximum(0,np.minimum(xy+1/32,box[3])-np.maximum(xy,box[1]))*32
    return wy[:,None]*wx[None,:]
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=4701);p.add_argument('--steps',type=int,default=600);args=p.parse_args();torch.set_num_threads(3);cv2.setNumThreads(1)
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');feat=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');q,te,h=inputs();head=load_jsonl('head');hands=load_jsonl('hand_local')
    bank={};examples=[]
    for fid in rec.frame_id.unique():
        for cls in CLASS:bank[(fid,cls)]=anchors(fid,cls,head,hands)
    # Match prompt points to full GT on annotated fitting frames only.
    for i,r in rec[rec.split.eq('manual')].iterrows():
        cb=r[CB].to_numpy(float)
        for cls in CLASS:
            gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(cls)][B].to_numpy();bb,points,scores=bank[(r.frame_id,cls)]
            for b in gt:
                nb=norm(b,cb);cc=(b[:2]+b[2:])/2
                if not ((nb[:2]>=0)&(nb[2:]<=1)).all():continue
                inside=((points>=b[:2])&(points<=b[2:])).all(1)
                if inside.any():
                    candidates=np.flatnonzero(inside);j=candidates[np.argmax(iou(b,bb[candidates])[0])];point=points[j];source='bpj'
                else:
                    point=cc.copy()
                    if cls=='person':point[1]=b[1]+.15*(b[3]-b[1])
                    source='synthetic_gt_point'
                point=(point-cb[:2])/(cb[2:]-cb[:2]);examples.append(dict(record_index=i,frame_id=r.frame_id,day=r.day,cls=CLASS.index(cls),point=point.tolist(),box=nb.tolist(),positive=True,prompt_source=source))
            # Hard prompt negatives: independent false detections well outside GT.
            for j,point in enumerate(points):
                pt=(point-cb[:2])/(cb[2:]-cb[:2])
                if not ((pt>.02)&(pt<.98)).all():continue
                inside=((point>=gt[:,:2])&(point<=gt[:,2:])).all(1).any() if len(gt) else False
                overlap=float(iou(bb[j],gt).max()) if len(gt) else 0.
                if not inside and overlap<.1:examples.append(dict(record_index=i,frame_id=r.frame_id,day=r.day,cls=CLASS.index(cls),point=pt.tolist(),box=[0,0,0,0],positive=False,prompt_source='bpj_false'))
    ledger=pd.DataFrame(examples);ledger.to_json(A/'inputs/prompt_ledger.json',orient='records',indent=2)
    targets=np.stack([occupancy(np.array(e['box'])) if e['positive'] else np.zeros((32,32)) for e in examples]).astype(np.float32);output=[];manifest=[]
    for fit in DAYS:
        tr=ledger.index[ledger.day.eq(fit)].to_numpy();held=rec[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit))];held=held[held.class_name.isin(CLASS)]
        assert not set(ledger.loc[tr].frame_id)&set(held.frame_id)
        # Equalize total contribution of frames; preserve observed foreground/negative mix.
        n=ledger.loc[tr].groupby('frame_id').size();w=np.array([1/n[f] for f in ledger.loc[tr].frame_id]);w/=w.sum()
        torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=PointHead();opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.02);start=time.time();losses=[]
        for step in range(args.steps):
            ix=rng.choice(tr,8,replace=True,p=w);rows=ledger.loc[ix];x=torch.from_numpy(np.array(feat[rows.record_index.to_numpy()],np.float32));point=np.stack(rows.point)
            # Small prompt jitter stays within its GT for positive examples.
            jitter=rng.normal(0,.008,point.shape);point+=jitter
            for j,e in enumerate(rows.itertuples()):
                if e.positive:point[j]=np.clip(point[j],np.array(e.box[:2])+.001,np.array(e.box[2:])-.001)
            point=np.clip(point,0,1);pred=model(x,torch.tensor(point,dtype=torch.float32),torch.tensor(rows.cls.to_numpy()));y=torch.from_numpy(targets[ix]);prob=pred.sigmoid()
            bce=F.binary_cross_entropy_with_logits(pred,y,weight=1+3*y);dice=1-(2*(prob*y).sum((1,2))+1)/(prob.sum((1,2))+y.sum((1,2))+1);loss=bce+.5*dice.mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),3);opt.step();losses.append(float(loss.detach()))
            if step%150==0:print('TRAIN',fit,args.seed,step,round(np.mean(losses[-100:]),5),round(time.time()-start,1),flush=True)
        model.eval();name=f'point_s{args.seed}_fit{fit[-2:]}';torch.save(model.state_dict(),A/f'models/{name}.pt')
        for i,r in held.iterrows():
            cb=r[CB].to_numpy(float);size=cb[2:]-cb[:2];bb,pp,conf=bank[(r.frame_id,r.class_name)];pt=(pp-cb[:2])/size;valid=((pt>.01)&(pt<.99)).all(1);pt=pt[valid];conf=conf[valid]
            if len(pt)>5:keep=np.argsort(conf)[-5:];pt=pt[keep];conf=conf[keep]
            refs=[]
            if len(pt):
                x=torch.from_numpy(np.repeat(np.array(feat[i:i+1],np.float32),len(pt),0));cl=torch.full((len(pt),),CLASS.index(r.class_name),dtype=torch.long)
                with torch.inference_mode():maps=model(x,torch.tensor(pt,dtype=torch.float32),cl).sigmoid().numpy()
                for point,prob,c in zip(pt,maps,conf):
                    _,labels,stats,_=cv2.connectedComponentsWithStats((prob>=.5).astype(np.uint8),8)
                    # Prompt chooses an instance; do not choose a component by query IoU.
                    px,py=np.clip((point*32).astype(int),0,31);near=labels[max(0,py-1):min(32,py+2),max(0,px-1):min(32,px+2)];labels_near=np.unique(near);choices=[]
                    for j in labels_near:
                        if j==0 or stats[j,4]<4:continue
                        choices.append((float(prob[labels==j].mean()),j))
                    if choices:
                        quality,j=max(choices);xx,yy,ww,hh,_=stats[j];box=np.array([xx,yy,xx+ww,yy+hh])/32*np.tile(size,2)+np.tile(cb[:2],2);refs.append((box,quality,float(c)))
            if refs:
                ov=iou(r[B],[z[0] for z in refs])[0];j=ov.argmax();b,conf_model,conf_anchor=refs[j];overlap=float(ov[j])
            else:b=np.full(4,np.nan);conf_model=conf_anchor=0.;overlap=np.nan
            output.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,fit_day=fit,split=r.split,source='query' if r.split=='oof' else 'test',class_name=r.class_name,overlap=overlap,reference_count=len(refs),prompt_count=len(pt),confidence=conf_model,anchor_confidence=conf_anchor,**dict(zip(['ref_'+b for b in B],b))))
        manifest.append(dict(name=name,fit_day=fit,seed=args.seed,fit_frames=sorted(ledger.loc[tr].frame_id.unique()),examples=len(tr),positive=int(ledger.loc[tr].positive.sum()),steps=args.steps,seconds=time.time()-start,last_loss=float(np.mean(losses[-100:]))));print('DONE',name,flush=True)
    pd.DataFrame(output).to_csv(A/f'features/point_s{args.seed}_evidence.csv',index=False);(A/f'models/point_s{args.seed}_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
