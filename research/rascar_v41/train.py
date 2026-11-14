"""Cross-day frozen-head control and actual last-block fine-tuning.
No epoch/seed/threshold is selected using test or held-out day labels.
"""
from vision import *
import argparse,time,gc,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('key',choices=['clip','vit_b','vit_l']);p.add_argument('--steps',type=int,default=160);p.add_argument('--seed',type=int,default=4101);args=p.parse_args();key=args.key
device='mps' if torch.backends.mps.is_available() else 'cpu';assert device=='mps'
rec=pd.read_csv(A/'inputs/records.csv');feat=np.load(A/f'features/{key}_penultimate.npy',mmap_mode='r');frozen=np.load(A/f'features/{key}_frozen.npy')
mask,extra=aux(rec,int(np.sqrt(feat.shape[1]-1)));q,te,h=inputs();outputs=[];manifests=[]
labels=torch.tensor(rec[['error','quality','presence']].fillna(0).to_numpy(),dtype=torch.float32)
available=torch.tensor(rec[['error','quality','presence']].notna().to_numpy(),dtype=torch.float32)
base_model=load(key)
for fit in DAYS:
 train=rec.day.eq(fit)&rec.split.isin(['query','synthetic','pool_presence']);ix=np.flatnonzero(train)
 ev=rec.split.isin(['test','pool_holdout'])|(rec.split.eq('query')&rec.day.ne(fit));ex=np.flatnonzero(ev)
 assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
 clean=train&rec.error.notna();source=rec.loc[clean,'split'];w=rec.loc[clean,'weight'].to_numpy();w/=rec.loc[clean].groupby('frame_id').id.transform('size').to_numpy();w*=len(w)/w.sum()
 lr=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=600,random_state=args.seed))
 lr.fit(frozen[clean],rec.loc[clean,'error'],logisticregression__sample_weight=w);joblib.dump(lr,A/f'models/{key}_frozen_fit{fit[-2:]}.joblib')
 d=rec.loc[ev,['id','frame_id','day','class_name','split']].copy();d['score']=lr.predict_proba(frozen[ev])[:,1];d['variant']='frozen';d['key']=key;d['fit_day']=fit;d['seed']=args.seed;outputs.append(d)
 torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed);model=QualityModel(base_model,key).to(device)
 opt=torch.optim.AdamW([{'params':list(model.block.parameters())+list(model.norm.parameters()),'lr':1e-5},{'params':model.head.parameters(),'lr':5e-4}],weight_decay=.02)
 # Sample group budgets, then frames equally within each source group.
 probs=np.zeros(len(rec))
 for split,budget in [('query',.35),('synthetic',.5),('pool_presence',.15)]:
  m=train&rec.split.eq(split);ww=1/rec.loc[m].groupby('frame_id').id.transform('size').to_numpy();probs[m]=budget*ww/ww.sum()
 probs=probs[ix]/probs[ix].sum();start=time.time();losses=[]
 for step in range(args.steps):
  model.train();ii=rng.choice(ix,24,replace=True,p=probs);x=torch.from_numpy(np.array(feat[ii],dtype=np.float32)).to(device)
  y=labels[ii].to(device);valid=available[ii].to(device);pred=model(x,mask[ii].to(device),extra[ii].to(device))
  loss_error=F.binary_cross_entropy_with_logits(pred[:,0],y[:,0],reduction='none')
  loss_iou=F.smooth_l1_loss(pred[:,1].sigmoid(),y[:,1],reduction='none',beta=.1)
  loss_presence=F.binary_cross_entropy_with_logits(pred[:,2],y[:,2],reduction='none')
  loss=((torch.stack([loss_error,.5*loss_iou,.3*loss_presence],1)*valid).sum(1)*torch.tensor(rec.loc[ii,'weight'].to_numpy(),device=device,dtype=torch.float32)).mean()
  assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach()))
  if step%40==0:print('TRAIN',key,fit,step,round(np.mean(losses[-40:]),4),round(time.time()-start,1),flush=True)
 model.eval();preds=[]
 with torch.inference_mode():
  for off in range(0,len(ex),24):
   ii=ex[off:off+24];x=torch.from_numpy(np.array(feat[ii],dtype=np.float32)).to(device)
   preds.append(model(x,mask[ii].to(device),extra[ii].to(device)).sigmoid().cpu().numpy())
 z=np.concatenate(preds);d=rec.loc[ev,['id','frame_id','day','class_name','split']].copy();d['score']=z[:,0];d['predicted_iou']=z[:,1];d['presence']=z[:,2];d['variant']='finetuned';d['key']=key;d['fit_day']=fit;d['seed']=args.seed;outputs.append(d)
 state={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(state,A/f'models/{key}_lastblock_fit{fit[-2:]}_s{args.seed}.pt')
 manifests.append(dict(key=key,fit_day=fit,seed=args.seed,steps=args.steps,train_frames=sorted(rec.loc[train,'frame_id'].unique()),train_records=len(ix),parameters_trained=sum(p.numel() for p in model.parameters()),held_day_in_fit=False,test_in_fit=False,pool_geometry_loss=False,last_loss=float(np.mean(losses[-20:])),seconds=time.time()-start))
 print('DONE',key,fit,metrics(q[q.day.ne(fit)],d[d.split.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True)
 del model,opt;gc.collect();torch.mps.empty_cache()
pd.concat(outputs).to_csv(A/f'features/{key}_predictions_s{args.seed}.csv',index=False)
(A/f'models/{key}_manifest_s{args.seed}.json').write_text(json.dumps(manifests,indent=2))
