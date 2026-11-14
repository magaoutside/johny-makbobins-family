"""Regularized class-conditioned CNN with training-day-only early stopping.
The evaluation day is never used for step selection or fitting preprocessing.
"""
from network import *
from sklearn.metrics import log_loss
import time,gc
device='mps';assert torch.backends.mps.is_available();rec=pd.read_csv(A/'inputs/records.csv');feat=np.load(A/'features/early_maps.npy',mmap_mode='r');meta=np.load(A/'inputs/metadata.npy');q,te,h=inputs();base=backbone();outputs=[];manifest=[]
class Conditional(nn.Module):
 def __init__(self):
  super().__init__();self.last=copy.deepcopy(base.features[6:]);self.norm=copy.deepcopy(base.classifier[0]);self.reduce=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,48),nn.GELU(),nn.Dropout(.3));self.geometry=nn.Sequential(nn.Linear(18,32),nn.GELU());self.head=nn.Sequential(nn.Linear(48*7+32,96),nn.GELU(),nn.Dropout(.4),nn.Linear(96,4))
 def forward(self,x,m):
  n=len(x);z=self.last(x.reshape(n*2,384,14,14)).reshape(n,2,768,7,7);v=vectors(z,m,'dual',self.norm)[:,:768*7].reshape(n,7,768);v=self.reduce(v).reshape(n,-1);logits=self.head(torch.cat([v,self.geometry(m)],1));return logits.gather(1,m[:,:4].argmax(1)[:,None]).squeeze(1)
def predict(net,ix):
 net.eval();out=[]
 with torch.inference_mode():
  for off in range(0,len(ix),20):
   ii=ix[off:off+20];out.extend(net(torch.tensor(np.array(feat[ii],dtype=np.float32),device=device),torch.tensor(meta[ii],device=device)).sigmoid().cpu().tolist())
 return np.array(out)
def fit_model(train,steps,check=None):
 torch.manual_seed(4203);rng=np.random.default_rng(4203);net=Conditional().to(device)
 opt=torch.optim.AdamW([{'params':list(net.last.parameters())+list(net.norm.parameters()),'lr':3e-6},{'params':list(net.reduce.parameters())+list(net.geometry.parameters())+list(net.head.parameters()),'lr':3e-4}],weight_decay=.1)
 ix=np.flatnonzero(train);prob=np.zeros(len(rec))
 for source,budget in [('query',.65),('synthetic',.25),('pool',.1)]:
  m=train&rec.source.eq(source)
  if m.any():
   w=1/rec.loc[m].groupby('frame_id').id.transform('size').to_numpy();prob[m]=budget*w/w.sum()
 prob=prob[ix]/prob[ix].sum();history=[];start=time.time()
 for step in range(1,steps+1):
  net.train();ii=rng.choice(ix,20,replace=True,p=prob);x=torch.tensor(np.array(feat[ii],dtype=np.float32),device=device);m=torch.tensor(meta[ii],device=device);y=torch.tensor(rec.iloc[ii].target.to_numpy(),device=device,dtype=torch.float32);w=torch.tensor(rec.iloc[ii].weight.to_numpy(),device=device,dtype=torch.float32)
  logits=net(x,m);loss=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  if step%40==0:
   row=dict(step=step,train_loss=float(loss.detach()))
   if check is not None:
    s=predict(net,check);y=rec.iloc[check].target.to_numpy();row.update(ap=float(average_precision_score(y,s)),log_loss=float(log_loss(y,s,labels=[0,1])))
   history.append(row);print('CONDITIONAL',step,row,round(time.time()-start,1),flush=True)
 return net,history
for fit in DAYS:
 train=rec.day.eq(fit)&rec.source.isin(['query','synthetic','pool']);ev=rec.source.eq('test')|(rec.source.eq('query')&rec.day.ne(fit));frames=sorted(rec.loc[train&rec.source.eq('query'),'frame_id'].unique());rng=np.random.default_rng(4203)
 hold=sorted(rng.choice(frames,max(3,round(len(frames)*.25)),replace=False));check=np.flatnonzero(rec.source.eq('query')&rec.frame_id.isin(hold))
 stamps=pd.to_datetime(rec.frame_id.str.split('_').str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9
 held_stamps=pd.to_datetime(pd.Series(hold).str.split('_').str[-2:].str.join(''),format='%Y%m%d%H%M%S').astype('int64').to_numpy()/1e9
 near=(abs(stamps[:,None]-held_stamps[None,:])<=120).any(1);inner=train&~near
 assert not set(rec.loc[inner,'frame_id'])&set(rec.iloc[check].frame_id)
 print('INNER',fit,'held frames',len(hold),'queries',len(check),flush=True);net,history=fit_model(inner,240,check)
 selected=sorted(history,key=lambda x:(-x['ap'],x['log_loss'],x['step']))[0]['step'];del net;gc.collect();torch.mps.empty_cache()
 print('SELECTED',fit,selected,flush=True);net,_=fit_model(train,selected);score=predict(net,np.flatnonzero(ev));d=rec.loc[ev,['id','frame_id','class_name','day','source']].copy();d['score']=score;d['fit_day']=fit;d['variant']='conditional_inner_selected';d['seed']=4203;outputs.append(d)
 torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},A/f'models/conditional_fit{fit[-2:]}.pt');manifest.append(dict(fit_day=fit,selected_steps=selected,inner_validation_frames=hold,inner_history=history,train_frames=sorted(rec.loc[train,'frame_id'].unique()),held_day_used=False,test_used=False,inner_temporal_exclusion_seconds=120))
 print('OUTER',fit,metrics(q[q.day.ne(fit)],d[d.source.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True);del net;gc.collect();torch.mps.empty_cache()
pd.concat(outputs).to_csv(A/'results/conditional_predictions_s4203.csv',index=False);(A/'models/conditional_manifest.json').write_text(json.dumps(manifest,indent=2))
