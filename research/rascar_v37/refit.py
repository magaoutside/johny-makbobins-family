"""Refit V31's simple occupancy head on all manual training data.

Cross-day metrics always use held-day predictions. Three fixed seeds have
the same agreement rule on OOF and test; final test heads see both train
days. No test or pool labels are fitted. Compare with a daily-head ensemble
under exactly the same OOF protocol, without claiming full-fit OOF exists.
"""
from common37 import *
import torch,time,cv2
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3);cv2.setNumThreads(1)
SEEDS=[3101,3102,3103]
CB=['crop_'+b for b in B]

class Head(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Conv2d(384,4,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(384))

def decode(pred,r):
    k=CL.index(r.class_name);prob=pred[k];n,labels,stats,_=cv2.connectedComponentsWithStats((prob>=.5).astype(np.uint8),8)
    cb=r[CB].to_numpy(float);sz=cb[2:]-cb[:2];boxes=[]
    for j in range(1,n):
        x,y,w,hh,ar=stats[j]
        if ar<4:continue
        bb=np.r_[x,y,x+w,y+hh]/32*np.r_[sz,sz]+np.r_[cb[:2],cb[:2]];boxes.append(bb)
    out=dict(id=r.id,overlap=np.nan,references=len(boxes))
    if boxes:
        ov=iou(r[B],boxes).ravel();j=ov.argmax();out.update(overlap=float(ov[j]),**dict(zip(['ref_'+b for b in B],boxes[j])))
    return out

def main():
    (A/'inputs/refit_protocol.json').write_text(json.dumps(dict(seeds=SEEDS,steps_daily=450,steps_all=900,threshold=.5,weight=.25,
        supervision='V31 manual box occupancy only; not segmentation truth',
        comparison='OOF three opposite-day seeds; test three both-day seeds. Daily consensus control shares OOF; full-fit has no separately observed OOF.',
        target='clip((.85-IoU)/.5,0,1)',test_labels_used=False),indent=2))
    rec=pd.read_csv(ROOT/'rascar_v31/inputs/dense_records.csv');feat=np.load(ROOT/'rascar_v31/features/dense_grids.npy',mmap_mode='r');y=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
    manifest=[];allpred=[]
    for fit in DAYS+['all']:
        train=rec.index[rec.split.eq('manual')&(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))].to_numpy()
        ev=rec.index[rec.split.eq('test')| (rec.split.eq('oof')&rec.day.ne(fit)&(fit!='all'))].to_numpy()
        assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
        counts=rec.loc[train].groupby('frame_id').size();weights=np.array([1/counts[f] for f in rec.loc[train,'frame_id']]);weights/=weights.sum()
        for seed in SEEDS:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Head();start=time.time();running=[];steps=900 if fit=='all' else 450
            if seed==3101 and fit!='all':
                model.load_state_dict(torch.load(ROOT/f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt',map_location='cpu',weights_only=True))
            else:
                opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
                for step in range(steps):
                    ids=rng.choice(train,8,p=weights);x=torch.from_numpy(np.asarray(feat[ids],np.float32));target=torch.from_numpy(np.asarray(y[ids],np.float32))
                    loss=F.binary_cross_entropy_with_logits(model(x),target,weight=1+3*target)
                    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();running.append(float(loss.detach()))
            torch.save(model.state_dict(),A/f'models/linear_mask_fit{fit[-2:]}_seed{seed}.pt');model.eval()
            rows=[]
            for off in range(0,len(ev),16):
                ids=ev[off:off+16]
                with torch.inference_mode():prob=model(torch.from_numpy(np.asarray(feat[ids],np.float32))).sigmoid().numpy()
                for j,i in enumerate(ids):
                    r=rec.loc[i];z=decode(prob[j],r);z.update(seed=seed,fit_day=fit,split=r.split);rows.append(z)
            allpred+=rows;manifest.append(dict(seed=seed,fit_day=fit,steps=steps,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),manual_crops=len(train),last_loss=np.mean(running[-100:]) if running else None,seconds=time.time()-start))
            print('REFIT',fit,seed,round(time.time()-start,1),flush=True)
    pd.DataFrame(allpred).to_csv(A/'features/refit_predictions.csv',index=False);(A/'models/refit_manifest.json').write_text(json.dumps(manifest,indent=2))
    evaluate()

def evaluate():
    pred=pd.read_csv(A/'features/refit_predictions.csv')
    # Recover the pre-V31 V22 OOF with its historical OOF-only adjustment removed.
    def aligned(p,rows):return pd.read_csv(ROOT/p).set_index('id').loc[rows.id].score.to_numpy()
    inherited=aligned('rascar_v18/results/oof/release_main.csv',q)-aligned('rascar_v18/results/oof/release_main_no_inherited_contradiction.csv',q)
    floor={'oof':np.clip(aligned('rascar_v22/results/oof/final_reviewed.csv',q)-inherited,0,1),'test':aligned('rascar_v22/submission_v22_candidate.csv',te)}
    results=[];evidence=[]
    for mode in ['daily_consensus','full_fit']:
        for aggregate in ['conservative','median']:
            scores={}
            for split,rows in ROWS.items():
                score=floor[split].copy()
                for i,r in rows.iterrows():
                    z=pred[pred.id.eq(r.id)&pred.split.eq(split)]
                    if split=='test':z=z[z.fit_day.eq('all') if mode=='full_fit' else z.fit_day.ne('all')]
                    target=np.clip((.85-z.overlap.to_numpy())/.5,0,1);delta=target-floor[split][i]
                    valid=np.isfinite(target).all() and len(target)>0 and (np.all(delta>=0) or np.all(delta<=0))
                    if valid:
                        t=np.median(target) if aggregate=='median' else (target.min() if np.all(delta>=0) else target.max())
                        score[i]+=.25*(t-floor[split][i])
                    evidence.append(dict(mode=mode,aggregate=aggregate,split=split,id=r.id,models=len(z),allowed=valid,new_score=score[i],v31_score=BASE[split][i]))
                scores[split]=score
            for cls,label in [(CL,'all'),(['cup','tray'],'tableware'),(['person'],'person'),(['nude_hands'],'hands')]:
                isolated={s:np.where(ROWS[s].class_name.isin(cls),scores[s],BASE[s]) for s in ROWS}
                results.append(save(f'refit_{mode}_{aggregate}_{label}',isolated,'V31 architecture; three seeds, same direction consensus. Full-fit test uses both train days; OOF uses opposite day. Threshold/weight unchanged.'))
    pd.DataFrame(results).to_csv(A/'results/refit_experiments.csv',index=False);pd.DataFrame(evidence).to_csv(A/'audit/refit_gates.csv',index=False)

if __name__=='__main__':main()
