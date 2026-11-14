"""Direct classifier with same-object positive/negative pairs.
Pairing suppresses shortcuts based only on clothing, object identity or class.
"""
from network import *
import time,gc
device='mps';assert torch.backends.mps.is_available();rec=pd.read_csv(A/'inputs/records.csv');feat=np.load(A/'features/early_maps.npy',mmap_mode='r');meta=np.load(A/'inputs/metadata.npy');q,te,h=inputs();base=backbone();outputs=[];manifest=[]
class Conditional(nn.Module):
 def __init__(self):
  super().__init__();self.last=copy.deepcopy(base.features[6:]);self.norm=copy.deepcopy(base.classifier[0]);self.reduce=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,48),nn.GELU(),nn.Dropout(.3));self.geometry=nn.Sequential(nn.Linear(18,32),nn.GELU());self.head=nn.Sequential(nn.Linear(48*7+32,96),nn.GELU(),nn.Dropout(.4),nn.Linear(96,4))
 def forward(self,x,m):
  n=len(x);z=self.last(x.reshape(n*2,384,14,14)).reshape(n,2,768,7,7);v=vectors(z,m,'dual',self.norm)[:,:768*7].reshape(n,7,768);v=self.reduce(v).reshape(n,-1);return self.head(torch.cat([v,self.geometry(m)],1)).gather(1,m[:,:4].argmax(1)[:,None]).squeeze(1)
for fit in DAYS:
 train=rec.day.eq(fit)&rec.source.isin(['query','synthetic','pool']);ev=rec.source.eq('test')|(rec.source.eq('query')&rec.day.ne(fit));ex=np.flatnonzero(ev);assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
 real=np.flatnonzero(train&rec.source.eq('query'));pseudo=np.flatnonzero(train&rec.source.eq('pool'));pairs=[]
 for _,g in rec[train&rec.source.eq('synthetic')&rec.origin_id.str.startswith('human_')].groupby(['origin_id','class_name']):
  good=g.index[g.target.eq(0)].to_numpy();bad=g.index[g.target.eq(1)].to_numpy()
  if len(good) and len(bad):pairs.append((good,bad))
 torch.manual_seed(4204);rng=np.random.default_rng(4204);net=Conditional().to(device);opt=torch.optim.AdamW([{'params':list(net.last.parameters())+list(net.norm.parameters()),'lr':3e-6},{'params':list(net.reduce.parameters())+list(net.geometry.parameters())+list(net.head.parameters()),'lr':3e-4}],weight_decay=.1);start=time.time()
 real_w=1/rec.iloc[real].groupby('frame_id').id.transform('size').to_numpy();real_w/=real_w.sum()
 for step in range(320):
  # 8 official queries + 4 weak pool queries + 4 paired comparisons (8 crops).
  ii=list(rng.choice(real,8,replace=True,p=real_w))+list(rng.choice(pseudo,4,replace=True));pp=[]
  for j in rng.choice(len(pairs),4,replace=True):
   good,bad=pairs[j];pp.append((len(ii),len(ii)+1));ii.extend([int(rng.choice(good)),int(rng.choice(bad))])
  ii=np.array(ii);net.train();x=torch.tensor(np.array(feat[ii],dtype=np.float32),device=device);m=torch.tensor(meta[ii],device=device);y=torch.tensor(rec.iloc[ii].target.to_numpy(),device=device,dtype=torch.float32);w=torch.tensor(rec.iloc[ii].weight.to_numpy(),device=device,dtype=torch.float32)
  logits=net(x,m);bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).mean();ranking=torch.stack([F.softplus(-(logits[b]-logits[g])) for g,b in pp]).mean();loss=bce+.5*ranking;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
  if step%40==0:print('PAIRS',fit,step,float(bce.detach()),float(ranking.detach()),round(time.time()-start,1),flush=True)
 net.eval();scores=[]
 with torch.inference_mode():
  for off in range(0,len(ex),20):
   ii=ex[off:off+20];scores.extend(net(torch.tensor(np.array(feat[ii],dtype=np.float32),device=device),torch.tensor(meta[ii],device=device)).sigmoid().cpu().tolist())
 d=rec.loc[ev,['id','frame_id','class_name','day','source']].copy();d['score']=scores;d['variant']='paired_classifier';d['seed']=4204;d['fit_day']=fit;outputs.append(d);torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},A/f'models/paired_fit{fit[-2:]}.pt');manifest.append(dict(fit_day=fit,steps=320,train_frames=sorted(rec.loc[train,'frame_id'].unique()),pair_groups=len(pairs),test_used=False,held_day_used=False))
 print('PAIR OUTER',fit,metrics(q[q.day.ne(fit)],d[d.source.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True);del net,opt;gc.collect();torch.mps.empty_cache()
pd.concat(outputs).to_csv(A/'results/paired_predictions_s4204.csv',index=False);(A/'models/paired_manifest.json').write_text(json.dumps(manifest,indent=2))
