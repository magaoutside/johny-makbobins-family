"""Train image-augmented occupancy heads; replace the old dense stage in V52."""
from common55 import *
import torch,cv2,argparse
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as AP

class Head(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Conv2d(384,4,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(384))

class ResidualHead(nn.Module):
    def __init__(self,fit):
        super().__init__();self.base=Head()
        self.base.load_state_dict(torch.load(ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt',weights_only=True))
        for p in self.base.parameters():p.requires_grad=False
        self.residual=nn.Sequential(nn.Conv2d(384,32,1),nn.GELU(),nn.Dropout2d(.1),nn.Conv2d(32,4,3,padding=1))
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)
    def forward(self,x):
        z=F.normalize(x,dim=1)*np.sqrt(384)
        return self.base.net(z)+self.residual(z)

SPECS={
 'control450':dict(steps=450,augmentation=None,aug_probability=0.,loss='bce',hard=False),
 'control900':dict(steps=900,augmentation=None,aug_probability=0.,loss='bce',hard=False),
 'flip450':dict(steps=450,augmentation=['flip'],aug_probability=.5,loss='bce',hard=False),
 'photo450':dict(steps=450,augmentation=['photo'],aug_probability=.5,loss='bce',hard=False),
 'aug450':dict(steps=450,augmentation=['flip','photo','crop_photo'],aug_probability=.5,loss='bce',hard=False),
 'aug900':dict(steps=900,augmentation=['flip','photo','crop_photo'],aug_probability=.5,loss='bce',hard=False),
 'aug_boundary900':dict(steps=900,augmentation=['flip','photo','crop_photo'],aug_probability=.5,loss='boundary',hard=False),
 'aug_hard900':dict(steps=900,augmentation=['flip','photo','crop_photo'],aug_probability=.5,loss='bce',hard=True),
 'residual450':dict(steps=450,augmentation=None,aug_probability=0.,loss='bce',hard=False,architecture='residual'),
 'residual_aug450':dict(steps=450,augmentation=['flip','photo','crop_photo'],aug_probability=.5,loss='bce',hard=False,architecture='residual'),
}

def fit_predict(name,seed,fit,rec,aug,orig_x,orig_y,aug_x,aug_y):
    spec=SPECS[name];train=rec.index[rec.split.eq('manual')&rec.day.eq(fit)].to_numpy()
    ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit))].to_numpy()
    assert set(rec.loc[train,'frame_id']).isdisjoint(rec.loc[ev,'frame_id'])
    frame_n=rec.loc[train].groupby('frame_id').size();prob=np.array([1/frame_n[f] for f in rec.loc[train,'frame_id']]);prob/=prob.sum()
    augment={i:g.index.to_numpy() for i,g in aug[aug.family.isin(spec['augmentation'] or [])].groupby('parent_index')}
    torch.manual_seed(seed);rng=np.random.default_rng(seed);arng=np.random.default_rng(seed+15000)
    residual=spec.get('architecture')=='residual'
    model=ResidualHead(fit) if residual else Head()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002 if residual else .0007,weight_decay=.01);start=time.time();losses=[];seen=set();augmented_draws=0
    for step in range(spec['steps']):
        # Separate RNGs preserve the original crop sampling sequence in controls.
        indices=rng.choice(train,8,replace=True,p=prob);xs=[];ys=[]
        for i in indices:
            if augment and arng.random()<spec['aug_probability']:
                j=arng.choice(augment[i]);assert aug.loc[j,'day']==fit
                xs.append(aug_x[j]);ys.append(aug_y[j]);seen.add(int(j));augmented_draws+=1
            else:xs.append(orig_x[i]);ys.append(orig_y[i])
        x=torch.from_numpy(np.array(xs,np.float32));y=torch.from_numpy(np.array(ys,np.float32));z=model(x)
        weight=1+3*y
        if spec['loss']=='boundary':
            # Rectangular GT uncertainty is not a true object segmentation target.
            outer=F.max_pool2d(y,3,stride=1,padding=1);inner=-F.max_pool2d(-y,3,stride=1,padding=1)
            weight=weight*(1+2*(outer-inner))
        element=F.binary_cross_entropy_with_logits(z,y,weight=weight,reduction='none');per_example=element.mean((1,2,3))
        loss=per_example.mean()
        if residual:
            with torch.no_grad():base_logits=model.base(x)
            loss=loss+.02*(z-base_logits).square().mean()
        if spec['hard'] and step>=150:
            # Online hard-example mining within the TRAIN batch only.
            loss=.5*loss+.5*per_example.topk(4).values.mean()
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
    modelname=f'{name}_s{seed}_fit{fit[-2:]}';torch.save(model.state_dict(),A/f'models/{modelname}.pt');model.eval();maps=[]
    for off in range(0,len(ev),16):
        ix=ev[off:off+16]
        with torch.inference_mode():maps.append(model(torch.from_numpy(np.array(orig_x[ix],np.float32))).sigmoid().numpy())
    np.savez_compressed(A/f'features/{modelname}_maps.npz',ids=rec.loc[ev,'id'].to_numpy(dtype=str),maps=np.concatenate(maps))
    m=dict(name=modelname,recipe=name,seed=seed,fit_day=fit,**spec,original_crops=len(train),train_frames=sorted(rec.loc[train,'frame_id'].unique()),
           validation_frames=sorted(rec.loc[ev[rec.loc[ev,'split'].eq('oof')],'frame_id'].unique()),augmented_draws=augmented_draws,distinct_augmented_crops=len(seen),
           final_loss=float(np.mean(losses[-100:])),seconds=time.time()-start)
    (A/f'models/{modelname}.json').write_text(json.dumps(m,indent=2));print('TRAIN',modelname,round(m['seconds'],1),m['final_loss'],flush=True)

def overlaps(prob,rows):
    values=[]
    for p,(_,r) in zip(prob,rows.iterrows()):
        crop=r[CB].to_numpy(float);k=CL.index(r.class_name)
        _,lab,stats,_=cv2.connectedComponentsWithStats((p[k]>=.5).astype(np.uint8),8)
        boxes=[]
        for x,y,w,h,area in stats[1:]:
            if area>=4:boxes.append(np.array([x,y,x+w,y+h])/32*np.tile(crop[2:]-crop[:2],2)+np.tile(crop[:2],2))
        values.append(float(iou(r[B],boxes).max()) if boxes else np.nan)
    return np.array(values)

def predecessors(q,t):
    oldq=pd.read_csv(ROOT/'rascar_v22/results/oof/final_reviewed.csv').set_index('id').loc[q.id]
    inherited=[]
    for filename in ['release_main.csv','release_main_no_inherited_contradiction.csv']:
        inherited.append(pd.read_csv(ROOT/'rascar_v18/results/oof'/filename).set_index('id').loc[q.id,'score'].to_numpy())
    pre_o=np.clip(oldq.score.to_numpy()-(inherited[0]-inherited[1]),0,1)
    pre_t=pd.read_csv(ROOT/'rascar_v22/submission_v22_candidate.csv').set_index('id').loc[t.id,'score'].to_numpy()
    return pre_o,pre_t

def score_dense(oo,tt,q,t):
    pre_o,pre_t=predecessors(q,t)
    target_o=np.clip((.85-oo)/.5,0,1);target_t=np.clip((.85-tt)/.5,0,1)
    s=pre_o.copy();valid=np.isfinite(oo);s[valid]+=.25*(target_o[valid]-pre_o[valid])
    delta=target_t-pre_t[None,:];same=((delta>=0).all(0)|(delta<=0).all(0))&np.isfinite(target_t).all(0)
    safe=np.where(np.isfinite(target_t),target_t,pre_t[None,:]);conservative=np.where((delta>=0).all(0),safe.min(0),safe.max(0))
    z=pre_t.copy();z[same]+=.25*(conservative[same]-pre_t[same])
    return s,z

def evaluate(names,seeds,tag):
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv').set_index('id');q,t,h=inputs();bo,bt=incumbent(q,t);v31o,v31t=baseline(q,t);results=[]
    for name in names:
        evidence=[]
        for fit in DAYS:
            maps=[];ids=None
            for seed in seeds:
                x=np.load(A/f'features/{name}_s{seed}_fit{fit[-2:]}_maps.npz')
                if ids is None:ids=x['ids']
                assert np.array_equal(ids,x['ids']);maps.append(x['maps'])
            p=np.mean(maps,axis=0);r=rec.loc[ids]
            d=r[['split','day','frame_id','class_name']].reset_index();d['fit_day']=fit;d['overlap']=overlaps(p,r);evidence.append(d)
        e=pd.concat(evidence);modeltag=f'{name}_{tag}';e.to_csv(A/f'features/{modeltag}_evidence.csv',index=False)
        oo=e[e.split.eq('oof')].set_index('id').loc[q.id,'overlap'].to_numpy()
        tt=np.stack([g.set_index('id').loc[t.id,'overlap'].to_numpy() for _,g in e[e.split.eq('test')].groupby('fit_day')])
        so,st=score_dense(oo,tt,q,t)
        if name=='control450' and seeds==[3101]:
            replay=dict(oof_max_error=float(abs(so-v31o).max()),test_max_error=float(abs(st-v31t).max()))
            (A/'audit/v31_control_replay.json').write_text(json.dumps(replay,indent=2));assert max(replay.values())<1e-10,replay
        for scope in ['all','person','nude_hands']:
            om=q.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(q),bool)
            tm=t.class_name.eq(scope).to_numpy() if scope!='all' else np.ones(len(t),bool)
            for fraction in [.5,1.]:
                s,z=bo.copy(),bt.copy()
                # Preserve the fixed V52 person anchor, replacing its dense predecessor.
                attenuation_o=np.where(q.class_name.eq('person'),1/1.2,1)
                attenuation_t=np.where(t.class_name.eq('person'),1/1.2,1)
                s[om]+=fraction*(so-v31o)[om]*attenuation_o[om]
                z[tm]+=fraction*(st-v31t)[tm]*attenuation_t[tm]
                label=f'{modeltag}_{scope}_replace{int(fraction*100)}'
                assert np.isfinite(s).all() and np.isfinite(z).all() and ((s>=0)&(s<=1)).all() and ((z>=0)&(z<=1)).all()
                q.assign(score=s).to_csv(A/f'results/oof/{label}.csv',index=False);t[['id']].assign(score=z).to_csv(A/f'results/submissions/{label}.csv',index=False)
                results.append(dict(name=label,recipe=name,seeds=','.join(map(str,seeds)),scope=scope,fraction=fraction,
                                    **metrics(q,s),delta=AP(q.error,s)-AP(q.error,bo),changed_test=int((abs(z-bt)>1e-9).sum()),max_test_change=float(abs(z-bt).max())))
        print('EVAL',modeltag,'best',max(r['ap'] for r in results[-6:]),flush=True)
    d=pd.DataFrame(results).sort_values('ap',ascending=False);d.to_csv(A/f'results/experiments_{tag}.csv',index=False)
    print(d.head(15).to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=list(SPECS));p.add_argument('--seeds',nargs='+',type=int,default=[3101]);p.add_argument('--tag',default='initial');p.add_argument('--evaluate-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(3);cv2.setNumThreads(1)
    if not args.evaluate_only:
        complete=json.loads((A/'features/extraction_complete.json').read_text());assert complete['record_sha256']==hashlib.sha256((A/'inputs/augmented_crops.csv').read_bytes()).hexdigest()
        rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');aug=pd.read_csv(A/'inputs/augmented_crops.csv')
        orig_x=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');orig_y=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
        aug_x=np.load(A/'features/augmented_grids.npy',mmap_mode='r');aug_y=np.load(A/'features/augmented_targets.npy',mmap_mode='r')
        for name in args.names:
            for seed in args.seeds:
                for fit in DAYS:fit_predict(name,seed,fit,rec,aug,orig_x,orig_y,aug_x,aug_y)
            evaluate([name],args.seeds,args.tag+'_'+name)
    evaluate(args.names,args.seeds,args.tag)

if __name__=='__main__':main()
