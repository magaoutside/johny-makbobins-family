"""Paired manual-only and silver-pool continuation of the tray channel."""
from pathlib import Path
import sys,json,time,itertools,argparse
import numpy as np,pandas as pd,torch,cv2
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];A=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rascar_v46'))
from common46 import inputs,baseline,iou,metrics,B,CL,DAYS,CB,records
from train import decode
for p in ['results/oof','results/submissions','audit']:(A/p).mkdir(parents=True,exist_ok=True)
class Tray(nn.Module):
    def __init__(self,dim):super().__init__();self.dim=dim;self.net=nn.Conv2d(dim,1,1)
    def forward(self,x):return self.net(F.normalize(x,dim=1)*np.sqrt(self.dim))[:,0]
def occupancy(b,cb):
    b=(b-np.tile(cb[:2],2))/np.tile(cb[2:]-cb[:2],2);xy=np.arange(32)/32;wx=np.maximum(0,np.minimum(xy+1/32,b[2])-np.maximum(xy,b[0]))*32;wy=np.maximum(0,np.minimum(xy+1/32,b[3])-np.maximum(xy,b[1]))*32;return wy[:,None]*wx[None,:]
def bounds(query,box,tol):
    values=[[v-tol,v+tol,np.clip(q,v-tol,v+tol)] for q,v in zip(query,box)];boxes=np.array(list(itertools.product(*values)));boxes=boxes[(boxes[:,2:]>boxes[:,:2]).all(1)];ov=iou(query,boxes)[0];return float(ov.min()),float(ov.max())
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=4901);p.add_argument('--keys',nargs='+',choices=['small','siglip2'],default=['small','siglip2']);args=p.parse_args();suffix='' if args.seed==4901 else f'_s{args.seed}'
    assert (A/'features/pool_complete.json').exists()
    torch.set_num_threads(3);cv2.setNumThreads(1);rec=records();pool=pd.read_csv(A/'inputs/pool_records.csv');truth=pd.read_csv(A/'pool_review/boxes.csv').set_index('frame_id');targets=np.load(ROOT/'rascar_v31/features/dense_targets.npy',mmap_mode='r');py=[];valid=[]
    for _,r in pool.iterrows():
        b=truth.loc[r.frame_id,B].to_numpy(float);tol=float(r.edge_uncertainty_px);cb=r[CB].to_numpy(float);py.append(occupancy(b,cb));inner=occupancy(b+np.array([tol,tol,-tol,-tol]),cb);outer=occupancy(b+np.array([-tol,-tol,tol,tol]),cb);valid.append((inner>=.999)|(outer<=.001))
    py=np.array(py,np.float32);valid=np.array(valid,np.float32);ledger=[];allout=[];poolout=[]
    for key in args.keys:
        orig=ROOT/('rascar_v31/features/dense_grids.npy' if key=='small' else 'rascar_v46/features/siglip2_grids.npy');feat=np.load(orig,mmap_mode='r');pf=np.load(A/f'features/{key}_pool.npy',mmap_mode='r')
        for fit in DAYS:
            tr=rec.index[rec.split.eq('manual')&rec.day.eq(fit)].to_numpy();pr=pool.index[pool.day.eq(fit)&pool.role.eq('development')].to_numpy();ev=rec[rec.class_name.eq('tray')&(rec.split.eq('test')|(rec.split.eq('oof')&rec.day.ne(fit)))];ph=pool[pool.day.ne(fit)]
            assert len(pr)>0;assert not set(pool.loc[pr].frame_id)&set(pool[pool.role.eq('reserved')].frame_id);assert not set(rec.loc[tr].frame_id)&set(ev.frame_id);assert not set(pool.loc[pr].frame_id)&set(ev.frame_id);assert not set(pool.loc[pr].day)&set(ev[ev.split.eq('oof')].day)
            source=ROOT/(f'rascar_v31/models/linear_mask_fit{fit[-2:]}.pt' if key=='small' else f'rascar_v46/models/siglip2_s3101_fit{fit[-2:]}.pt');state=torch.load(source,weights_only=True);count=rec.loc[tr].groupby('frame_id').size();w=np.array([1/count[f] for f in rec.loc[tr].frame_id]);w/=w.sum();rng=np.random.default_rng(args.seed);manual_draws=rng.choice(tr,(350,8),replace=True,p=w);pool_draws=rng.choice(pr,(350,2),replace=True)
            for kind in ['original','control','silver']:
                model=Tray(feat.shape[1]);model.load_state_dict({'net.weight':state['net.weight'][3:4].clone(),'net.bias':state['net.bias'][3:4].clone()});start=time.time()
                if kind!='original':
                    opt=torch.optim.AdamW(model.parameters(),lr=.00035,weight_decay=.01)
                    for step in range(350):
                        ix=manual_draws[step];x=np.array(feat[ix],np.float32);y=np.array(targets[ix,3],np.float32);vv=np.ones_like(y)
                        if kind=='silver':pi=pool_draws[step];x[-2:]=pf[pi];y[-2:]=py[pi];vv[-2:]=valid[pi]
                        y=torch.tensor(y);vv=torch.tensor(vv);z=model(torch.tensor(x));loss=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*(1+3*y)*vv).sum()/vv.sum();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                    torch.save(model.state_dict(),A/f'models/{key}_{kind}_fit{fit[-2:]}{suffix}.pt')
                model.eval();ledger.append(dict(key=key,kind=kind,fit_day=fit,seed=args.seed,manual_frames=sorted(rec.loc[tr].frame_id.unique()),pool_frames=sorted(pool.loc[pr].frame_id.unique()) if kind=='silver' else [],steps=350 if kind!='original' else 0,seconds=time.time()-start))
                for split,rows,features in [('query_test',ev,feat),('pool',ph,pf)]:
                    for i,r in rows.iterrows():
                        with torch.inference_mode():prob=model(torch.tensor(np.array(features[i:i+1],np.float32)))[0].sigmoid().numpy()
                        probs=np.zeros((4,32,32),np.float32);probs[3]=prob;d=decode(probs,r)
                        if split=='query_test':allout.append(dict(id=r.id,frame_id=r.frame_id,day=r.day,fit_day=fit,split=r.split,source='query' if r.split=='oof' else 'test',class_name='tray',key=key,kind=kind,**d))
                        else:
                            b=truth.loc[r.frame_id,B].to_numpy(float);ref=np.array([d['ref_'+v] for v in B]);lo,hi=bounds(r[B].to_numpy(float),b,r.edge_uncertainty_px)
                            poolout.append(dict(review_index=r.review_index,frame_id=r.frame_id,day=r.day,role=r.role,key=key,kind=kind,fit_day=fit,query_iou_low=lo,query_iou_high=hi,robust_label=0 if lo>=.6 else (1 if hi<.6 else np.nan),reference_gt_iou=float(iou(ref,b)[0,0]) if np.isfinite(ref).all() else 0.,**d))
                print('DONE',key,kind,fit,round(time.time()-start,1),flush=True)
    pd.DataFrame(allout).to_csv(A/f'features/evidence{suffix}.csv',index=False);pd.DataFrame(poolout).to_csv(A/f'audit/pool_geometry{suffix}.csv',index=False);(A/f'models/manifest{suffix}.json').write_text(json.dumps(ledger,indent=2))
if __name__=='__main__':main()
