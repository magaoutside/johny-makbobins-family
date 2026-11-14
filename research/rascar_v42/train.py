"""Single binary error score, not predicted IoU or reconstructed boxes."""
from network import *
import argparse,time,gc,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['tight','dual','dual_pool'],required=True);parser.add_argument('--steps',type=int,default=320);parser.add_argument('--seed',type=int,default=4201);args=parser.parse_args()
device='mps';assert torch.backends.mps.is_available();rec=pd.read_csv(A/'inputs/records.csv');feat=np.load(A/'features/early_maps.npy',mmap_mode='r');frozen=np.load(A/'features/frozen_vectors.npy');meta=torch.tensor(np.load(A/'inputs/metadata.npy'));q,te,h=inputs();base=backbone();outputs=[];manifests=[]
for fit in DAYS:
 mode='tight' if args.variant=='tight' else 'dual';sources=['query','synthetic']+(['pool'] if args.variant=='dual_pool' else [])
 train=rec.day.eq(fit)&rec.source.isin(sources);ev=rec.source.eq('test')|(rec.source.eq('query')&rec.day.ne(fit));ix=np.flatnonzero(train);ex=np.flatnonzero(ev)
 assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id']);assert rec.loc[train,'target'].notna().all()
 prob=np.zeros(len(rec));budgets={'query':.55,'synthetic':.3,'pool':.15} if 'pool' in sources else {'query':.6,'synthetic':.4}
 for source,budget in budgets.items():
  m=train&rec.source.eq(source);sub=rec[m];counts=sub.groupby('frame_id').id.transform('size').to_numpy();w=1/counts;prob[m]=budget*w/w.sum()
 prob=prob[ix]/prob[ix].sum();torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=Classifier(base,mode).to(device)
 opt=torch.optim.AdamW([{'params':list(model.last.parameters())+list(model.norm.parameters()),'lr':1e-5},{'params':model.head.parameters(),'lr':3e-4}],weight_decay=.03)
 start=time.time();losses=[];model.train()
 for step in range(args.steps):
  ii=rng.choice(ix,20,replace=True,p=prob);x=torch.tensor(np.array(feat[ii],dtype=np.float32),device=device);m=meta[ii].to(device);y=torch.tensor(rec.iloc[ii].target.to_numpy(),device=device,dtype=torch.float32);w=torch.tensor(rec.iloc[ii].weight.to_numpy(),device=device,dtype=torch.float32)
  # Low-amplitude feature noise regularizes without changing bbox labels.
  x=x+torch.randn_like(x)*.015
  logits=model(x,m);bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).mean()
  # Rank only real queries, same semantic class. No synthetic/weak pair targets.
  real=torch.tensor(rec.iloc[ii].source.eq('query').to_numpy(),device=device);cls=m[:,:4].argmax(1);pairs=real[:,None]&real[None,:]&y[:,None].eq(1)&y[None,:].eq(0)&cls[:,None].eq(cls[None,:])
  rank=F.softplus(-(logits[:,None]-logits[None,:]))[pairs].mean() if pairs.any() else logits.sum()*0
  loss=bce+.1*rank;assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  if step%40==0:print('TRAIN',args.variant,fit,step,round(np.mean(losses[-40:]),5),round(time.time()-start,1),flush=True)
 model.eval();scores=[]
 with torch.inference_mode():
  for off in range(0,len(ex),20):
   ii=ex[off:off+20];x=torch.tensor(np.array(feat[ii],dtype=np.float32),device=device);scores.extend(model(x,meta[ii].to(device)).sigmoid().cpu().tolist())
 d=rec.loc[ev,['id','frame_id','class_name','day','source']].copy();d['score']=scores;d['variant']=args.variant;d['fit_day']=fit;d['seed']=args.seed;outputs.append(d)
 torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},A/f'models/{args.variant}_fit{fit[-2:]}_s{args.seed}.pt')
 manifests.append(dict(variant=args.variant,fit_day=fit,seed=args.seed,steps=args.steps,counts=rec.loc[train].source.value_counts().to_dict(),train_frames=sorted(rec.loc[train,'frame_id'].unique()),test_used=False,held_day_used=False,target='binary error',seconds=time.time()-start,trainable_parameters=sum(p.numel() for p in model.parameters())))
 print('DONE',args.variant,fit,metrics(q[q.day.ne(fit)],d[d.source.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True)
 del model,opt;gc.collect();torch.mps.empty_cache()
pd.concat(outputs).to_csv(A/f'results/{args.variant}_predictions_s{args.seed}.csv',index=False);(A/f'models/{args.variant}_manifest_s{args.seed}.json').write_text(json.dumps(manifests,indent=2))
