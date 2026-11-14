"""Hand-only pixel-logit suppression using reviewed TRAIN negatives.

Inside a reviewed negative query is supervised as no-hand. Surrounding pixels
are ignored, since context may contain true hands. All other classes preserve
the refit weights exactly. No manually assigned test targets or per-ID rules.
"""
from common37 import *
from refit import Head, SEEDS, CB
import torch, cv2, time
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
torch.set_num_threads(3);cv2.setNumThreads(1)

def negatives():
    nr=pd.read_csv(A/'inputs/near_hand_review.csv')
    lab=json.loads((A/'inputs/near_hand_review.json').read_text())
    nr=nr[nr.review_index.isin(lab['approved_negative_indices'])].copy()
    nr['review_source']='near_train';pool=pd.read_csv(A/'inputs/promoted_pool_presence.csv')
    pool=pool[pool.semantic_target.eq(0)].copy()
    rec=pd.concat([nr,pool],ignore_index=True)
    assert rec.id.is_unique and not set(rec.frame_id)&set(te.frame_id)
    for i,r in rec.iterrows():
        b=r[B].to_numpy(float);wh=np.maximum(b[2:]-b[:2],16);cc=(b[:2]+b[2:])/2
        cb=np.r_[np.maximum(0,np.floor(cc-1.2*wh)),np.minimum([1280,720],np.ceil(cc+1.2*wh))]
        rec.loc[i,CB]=cb
    rec.to_csv(A/'inputs/occupancy_negatives.csv',index=False)
    return rec

def grids(rec):
    path=A/'features/negative_grids.npy'
    if path.exists():
        x=np.load(path);assert len(x)==len(rec);return x
    model=AutoModel.from_pretrained(str(ROOT/'rascar_v31/model_cache/dinov2-small'),local_files_only=True).eval();out=[]
    mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
    for off in range(0,len(rec),4):
        batch=[]
        for _,r in rec.iloc[off:off+4].iterrows():
            c=r[CB].to_numpy(int);im=cv2.resize(cv2.imread(str(ROOT/'rascar_data_full'/f'{r.frame_id}.jpg')),(1280,720))
            crop=im[c[1]:c[3],c[0]:c[2],::-1];batch.append(cv2.resize(crop,(448,448),interpolation=cv2.INTER_CUBIC).transpose(2,0,1).copy())
        with torch.inference_mode():x=model(pixel_values=(torch.from_numpy(np.stack(batch)).float()/255-mean)/std).last_hidden_state[:,1:].transpose(1,2).reshape(-1,384,32,32).numpy()
        out.append(x.astype(np.float16))
    x=np.concatenate(out);np.save(path,x);return x

def masks(rec):
    yy,xx=np.mgrid[:32,:32];xy=np.stack([(xx+.5)/32,(yy+.5)/32],-1);out=[]
    for _,r in rec.iterrows():
        c=r[CB].to_numpy(float);b=(r[B].to_numpy(float)-np.tile(c[:2],2))/np.tile(c[2:]-c[:2],2)
        out.append(((xy>=b[:2])&(xy<=b[2:])).all(-1))
    return np.stack(out)

def main():
    protocol=dict(supervision='Only hand-channel logits inside reviewed train-negative boxes; ignore surrounding context.',
        negatives='12 reviewed near-GT plus V36/V37 reviewed absent hand pool crops; fit-day only.',
        steps_daily=150,steps_full=300,seeds=SEEDS,lr=.0002,negative_loss_weight=.25,
        correction='Only raise error when all models have 90th percentile of hand probability < .15 inside query; weight .25.',
        reference='Start each seed from V37 positive-box occupancy head; positive crops retain their original supervision.',
        test_labels_used=False)
    (A/'inputs/negative_occupancy_protocol.json').write_text(json.dumps(protocol,indent=2))
    nr=negatives();nx=grids(nr);nm=masks(nr)
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');x=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');y=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r');rm=masks(rec)
    predictions=[];manifest=[];hand=CL.index('nude_hands')
    for fit in DAYS+['all']:
        train=rec.index[rec.split.eq('manual')&(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))].to_numpy()
        neg=nr.index[nr.day.isin(DAYS) if fit=='all' else nr.day.eq(fit)].to_numpy()
        ev=rec.index[rec.class_name.eq('nude_hands')&(rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit)&(fit!='all')))].to_numpy()
        assert not (set(rec.loc[train,'frame_id'])|set(nr.loc[neg,'frame_id']))&set(rec.loc[ev,'frame_id'])
        counts=rec.loc[train].groupby('frame_id').size();wt=np.array([1/counts[f] for f in rec.loc[train,'frame_id']]);wt/=wt.sum()
        nc=nr.loc[neg].groupby('frame_id').size();nw=np.array([1/nc[f] for f in nr.loc[neg,'frame_id']]);nw/=nw.sum()
        for seed in SEEDS:
            start=time.time();torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Head()
            original=torch.load(A/f'models/linear_mask_fit{fit[-2:]}_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(original)
            opt=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=0);steps=300 if fit=='all' else 150
            for step in range(steps):
                ix=rng.choice(train,8,p=wt);nnn=rng.choice(neg,8,p=nw)
                pred=model(torch.from_numpy(np.asarray(x[ix],np.float32)))[:,hand]
                target=torch.from_numpy(np.asarray(y[ix,hand],np.float32));pos=F.binary_cross_entropy_with_logits(pred,target,weight=1+3*target)
                nlog=model(torch.from_numpy(np.asarray(nx[nnn],np.float32)))[:,hand];mask=torch.from_numpy(nm[nnn].astype(np.float32))
                negative=(F.softplus(nlog)*mask).sum()/mask.sum().clamp_min(1)
                loss=pos+.25*negative;assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            # No gradient in unrelated output channels, and zero weight decay.
            keep=[k for k in range(4) if k!=hand]
            assert torch.equal(model.net.weight.detach()[keep],original['net.weight'][keep])
            torch.save(model.state_dict(),A/f'models/negative_occupancy_fit{fit[-2:]}_seed{seed}.pt');model.eval()
            for off in range(0,len(ev),16):
                ids=ev[off:off+16]
                with torch.inference_mode():p=model(torch.from_numpy(np.asarray(x[ids],np.float32)))[:,hand].sigmoid().numpy()
                for i,prob in zip(ids,p):predictions.append(dict(id=rec.loc[i,'id'],split=rec.loc[i,'split'],fit_day=fit,seed=seed,p90=float(np.quantile(prob[rm[i]],.9))))
            manifest.append(dict(fit_day=fit,seed=seed,fit_frames=sorted(set(rec.loc[train,'frame_id'])|set(nr.loc[neg,'frame_id'])),negative_crops=len(neg),steps=steps))
            print('NEGATIVE_OCCUPANCY',fit,seed,round(time.time()-start,1),flush=True)
    pred=pd.DataFrame(predictions);pred.to_csv(A/'features/negative_occupancy_predictions.csv',index=False);(A/'models/negative_occupancy_manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[]
    for mode in ['daily_consensus','full_fit']:
        scores={}
        for split,rows in ROWS.items():
            s=BASE[split].copy()
            for i,r in rows[rows.class_name.eq('nude_hands')].iterrows():
                d=pred[pred.id.eq(r.id)&pred.split.eq(split)]
                if split=='test':d=d[d.fit_day.eq('all') if mode=='full_fit' else d.fit_day.ne('all')]
                p=d.p90.max()
                if p<.15:s[i]+=.25*max(1-p-s[i],0)
            scores[split]=s
        results.append(save(f'negative_occupancy_{mode}',scores,'Fine-tune hand channel with reviewed TRAIN-only absent-hand regions; spatial negatives, 3 seeds; fixed p90 .15 gate.'))
    pd.DataFrame(results).to_csv(A/'results/negative_occupancy_experiments.csv',index=False)

if __name__=='__main__':main()
