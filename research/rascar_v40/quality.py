"""ROI-aligned quality verifier; localization confidence separate from presence.

Inspired by IoU-Net, not its implementation or pretrained weights. Supervision:
maximum same-class IoU to official fit-day GT of jittered proposals, including
background. Cross-day synthetic controls and actual queries are never fitted.
"""
from common40 import *
from set_detector import Data,CB,torch,nn,F,math,SEEDS
import argparse,time

def sample_roi(features,boxes,size):
    # align_corners=False: normalized 0 and 1 are feature-map outer boundaries.
    t=(torch.arange(size,device=features.device,dtype=features.dtype)+.5)/size
    yy,xx=torch.meshgrid(t,t,indexing='ij');unit=torch.stack([xx,yy],-1)
    grid=boxes[:,None,None,:2]+unit[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
    return F.grid_sample(features,grid*2-1,mode='bilinear',padding_mode='border',align_corners=False)

class QualityHead(nn.Module):
    def __init__(self):
        super().__init__();self.project=nn.Sequential(nn.Conv2d(384,32,1),nn.GroupNorm(8,32),nn.GELU())
        self.mlp=nn.Sequential(nn.Linear(32*(49+81)+4+4,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,64),nn.GELU(),nn.Linear(64,2))
    def forward(self,features,boxes,classes,source_indices=None):
        z=self.project(F.normalize(features,dim=1)*math.sqrt(384))
        if source_indices is not None:z=z[source_indices]
        cc=(boxes[:,:2]+boxes[:,2:])/2;wh=(boxes[:,2:]-boxes[:,:2])*1.5
        context=torch.cat([cc-wh/2,cc+wh/2],1).clamp(0,1)
        roi=sample_roi(z,boxes,7).flatten(1);ctx=sample_roi(z,context,9).flatten(1)
        return self.mlp(torch.cat([roi,ctx,boxes,F.one_hot(classes,4).float()],1))

def normalize(box,cb):return (box-np.r_[cb[:2],cb[:2]])/np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]

class QualityData(Data):
    def __init__(self):
        super().__init__();self.truth={f:{c:g[g.class_name.eq(c)][B].to_numpy(float) for c in CL} for f,g in h.groupby('frame_id')}
    def proposal(self,i,rng):
        r=self.rec.loc[i];assert r.split!='test';cb=r[CB].to_numpy(float)
        cls=r.class_name if r.class_name in CL and rng.random()<.8 else CL[int(rng.integers(4))]
        gt=self.truth[r.frame_id][cls];norm=normalize(gt,cb) if len(gt) else np.empty((0,4));complete=((norm>=0)&(norm<=1)).all(1)
        if complete.any() and rng.random()<.85:
            target=norm[rng.choice(np.flatnonzero(complete))];wh=target[2:]-target[:2];cc=(target[:2]+target[2:])/2
            if rng.random()<.1:proposal=target.copy()
            else:
                cc+=rng.normal(0,.24,2)*wh;size=wh*np.exp(rng.normal(0,.38,2));proposal=np.r_[cc-size/2,cc+size/2]
                if rng.random()<.4:
                    side=int(rng.integers(4));proposal[side]+=rng.uniform(-.35,.35)*wh[side%2]
        else:
            cc=rng.uniform(.15,.85,2);wh=rng.uniform(.12,.65,2);proposal=np.r_[cc-wh/2,cc+wh/2]
        proposal=np.clip(proposal,0,1);proposal[2:]=np.maximum(proposal[2:],proposal[:2]+.02);proposal=np.clip(proposal,0,1)
        # Entire GT remains the target. A clipped crop does not redefine its box.
        pixel=proposal*np.r_[cb[2:]-cb[:2],cb[2:]-cb[:2]]+np.r_[cb[:2],cb[:2]]
        overlap=float(iou(pixel,gt).max()) if len(gt) else 0.
        return proposal.astype(np.float32),CL.index(cls),np.array([float(overlap>=.6),overlap],np.float32)

def infer(model,data,jobs):
    out=[]
    for off in range(0,len(jobs),32):
        part=jobs.iloc[off:off+32];boxes=torch.tensor(part[['norm_'+b for b in B]].to_numpy(),dtype=torch.float32);classes=torch.tensor([CL.index(c) for c in part.class_name])
        with torch.inference_mode():p=model(data.features(part.source_index.to_numpy(int)),boxes,classes).sigmoid().numpy()
        out.extend(dict(**r._asdict(),p_good=float(p[j,0]),predicted_iou=float(p[j,1])) for j,r in enumerate(part.itertuples(index=False)))
    return pd.DataFrame(out)

def jobs_for_queries(data,fit):
    rec=data.rec;selected=rec[(rec.split.eq('test')|((rec.day.ne(fit)&rec.split.eq('oof')) if fit!='all' else False))&rec.class_name.isin(CL)];jobs=[]
    source=ROOT/'rascar_v39/features'/('set_long_full_identity_matching_evidence.csv' if fit=='all' else 'set_long_identity_matching_evidence.csv')
    refs=pd.read_csv(source);refs=refs[refs.fit_day.eq(fit)]
    for i,r in selected.iterrows():
        cb=r[CB].to_numpy(float);values=[('query',-1,r[B].to_numpy(float))]
        for _,v in refs[refs.id.eq(r.id)&refs.split.eq(r.split)].iterrows():values.append(('v39_reference',int(v.seed),v[['ref_'+b for b in B]].to_numpy(float)))
        for kind,proposal_seed,box in values:
            norm=normalize(box,cb)
            if not np.isfinite(norm).all() or norm.min()<0 or norm.max()>1 or (norm[2:]-norm[:2]).min()<=0:continue
            jobs.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,split=r.split,class_name=r.class_name,kind=kind,proposal_seed=proposal_seed,source_index=i,**dict(zip(['norm_'+b for b in B],norm))))
    return pd.DataFrame(jobs)

def synthetic_jobs(data,fit):
    rng=np.random.default_rng(40000);jobs=[]
    # A fixed panel on the opposite day; labels are diagnostics, not fit targets.
    for i,r in data.rec[data.rec.day.ne(fit)&data.rec.split.isin(['manual','empty'])].iterrows():
        for k in range(6):
            box,cls,y=data.proposal(i,rng);jobs.append(dict(id=f'{r.id}_j{k}',origin_id=r.origin_id if isinstance(r.origin_id,str) else r.id,source_index=i,frame_id=r.frame_id,day=r.day,class_name=CL[cls],good=float(y[0]),true_iou=float(y[1]),**dict(zip(['norm_'+b for b in B],box))))
    return pd.DataFrame(jobs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1500);p.add_argument('--seeds',nargs='+',type=int,default=SEEDS);p.add_argument('--fit-both',action='store_true');args=p.parse_args();data=QualityData();rec=data.rec
    fits=['all'] if args.fit_both else DAYS;run='quality_full' if args.fit_both else 'quality';manifest=[]
    (A/f'inputs/{run}_protocol.json').write_text(json.dumps(dict(steps=args.steps,seeds=args.seeds,fit_days=fits,loss='BCE IoU>=.6 + 4 smoothL1 IoU',roi_sizes=[7,9],train_splits=['manual','empty'],train_test_targets=False,source='https://arxiv.org/abs/1807.11590'),indent=2))
    for fit in fits:
        fitmask=rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit)
        train=rec.index[fitmask&rec.split.isin(['manual','empty'])].to_numpy();groups=[train[rec.loc[train,'split'].eq(s).to_numpy()] for s in ['manual','empty']];prob=[]
        for g in groups:
            part=rec.loc[g];keys=part.frame_id+'_'+part.class_name;count=keys.value_counts();w=np.array([1/count[k] for k in keys]);prob.append(w/w.sum())
        jobs=jobs_for_queries(data,fit);assert not set(rec.loc[train,'frame_id'])&set(jobs.frame_id)
        panel=synthetic_jobs(data,fit) if fit!='all' else None
        if panel is not None:panel.to_csv(A/f'inputs/quality_panel_fit{fit[-2:]}.csv',index=False)
        for seed in args.seeds:
            name=f'{run}_fit{fit if fit=="all" else fit[-2:]}_seed{seed}';torch.manual_seed(seed+40000);rng=np.random.default_rng(seed+40000);model=QualityHead();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.02);start=time.time();losses=[]
            for step in range(args.steps):
                ids=np.r_[rng.choice(groups[0],6,p=prob[0]),rng.choice(groups[1],2,p=prob[1])]
                targets=[data.proposal(int(i),rng) for i in ids for _ in range(3)]
                b,c,y=[torch.tensor(np.array([z[k] for z in targets])) for k in range(3)];indices=torch.arange(len(ids)).repeat_interleave(3)
                output=model(data.features(ids),b,c,indices);loss=F.binary_cross_entropy_with_logits(output[:,0],y[:,0])+4*F.smooth_l1_loss(output[:,1].sigmoid(),y[:,1],beta=.1);assert torch.isfinite(loss)
                for g in opt.param_groups:g['lr']=.0003*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                if step%200==0:print('TRAIN',name,step,round(np.mean(losses[-100:]),4),round(time.time()-start,1),flush=True)
            torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();infer(model,data,jobs).assign(fit_day=fit,seed=seed).to_csv(A/f'features/{name}_predictions.csv',index=False)
            if panel is not None:infer(model,data,panel).assign(fit_day=fit,seed=seed).to_csv(A/f'features/{name}_panel.csv',index=False)
            manifest.append(dict(name=name,fit_day=fit,seed=seed,steps=args.steps,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),last_loss=float(np.mean(losses[-100:])),seconds=time.time()-start))
            (A/f'models/{run}_manifest.json').write_text(json.dumps(manifest,indent=2));print('DONE',name,round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
