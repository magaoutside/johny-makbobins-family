"""Compare local early layers with four-layer DINO linear occupancy heads."""
from common38 import *
from refit import decode
import torch,time,cv2,argparse
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(3);cv2.setNumThreads(1)
SEEDS=[3101,3102,3103]

class Head(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.layers=4 if kind=='all4' else 2
        self.net=nn.Conv2d(384*self.layers,4,1)
    def forward(self,x):
        if self.kind=='early2':x=x[:,:768]
        n,_,hh,ww=x.shape
        x=F.normalize(x.reshape(n,self.layers,384,hh,ww),dim=2).reshape(n,-1,hh,ww)*np.sqrt(384)
        return self.net(x)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kinds',nargs='+',default=['all4','early2']);args=ap.parse_args()
    rec=pd.read_csv(A/'inputs/multilevel_records.csv');x=np.load(A/'features/multilevel_grids.npy',mmap_mode='r');y=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r')
    assert (A/'features/multilevel_complete.json').exists()
    (A/'inputs/multilevel_protocol.json').write_text(json.dumps(dict(kinds=args.kinds,seeds=SEEDS,steps_daily=450,steps_all=900,
       learning_rate=.0007,threshold=.5,blend=.25,training='Exact V31 official manual crops, same targets; no new pool or test annotations',
       source='https://github.com/facebookresearch/dinov2',reason='Test whether earlier transformer layers add local geometry discarded by last-layer features.'),indent=2))
    rows=[];manifest=[]
    for kind in args.kinds:
        for fit in DAYS+['all']:
            train=rec.index[rec.split.eq('manual')&(rec.day.isin(DAYS) if fit=='all' else rec.day.eq(fit))].to_numpy()
            ev=rec.index[rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit)&(fit!='all'))].to_numpy()
            assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
            counts=rec.loc[train].groupby('frame_id').size();weights=np.array([1/counts[f] for f in rec.loc[train,'frame_id']]);weights/=weights.sum()
            for seed in SEEDS:
                start=time.time();torch.manual_seed(seed);rng=np.random.default_rng(seed);model=Head(kind);opt=torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.01)
                steps=900 if fit=='all' else 450;losses=[]
                for step in range(steps):
                    ids=rng.choice(train,8,p=weights);target=torch.from_numpy(np.asarray(y[ids],np.float32))
                    loss=F.binary_cross_entropy_with_logits(model(torch.from_numpy(np.asarray(x[ids],np.float32))),target,weight=1+3*target)
                    assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
                name=f'multilevel_{kind}_fit{fit[-2:]}_s{seed}';torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval()
                for off in range(0,len(ev),16):
                    ids=ev[off:off+16]
                    with torch.inference_mode():prob=model(torch.from_numpy(np.asarray(x[ids],np.float32))).sigmoid().numpy()
                    for i,p in zip(ids,prob):
                        r=rec.loc[i];z=decode(p,r);z.update(split=r.split,fit_day=fit,seed=seed,kind=kind);rows.append(z)
                manifest.append(dict(name=name,fit_day=fit,kind=kind,seed=seed,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),train_crops=len(train),steps=steps,last_loss=np.mean(losses[-100:]),seconds=time.time()-start))
                print('TRAIN',kind,fit,seed,round(time.time()-start,1),flush=True)
                (A/'models/multilevel_manifest.json').write_text(json.dumps(manifest,indent=2));pd.DataFrame(rows).to_csv(A/'features/multilevel_predictions.csv',index=False)

if __name__=='__main__':main()
