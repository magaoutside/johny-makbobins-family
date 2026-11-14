"""Check fit-vs-held query AP using saved checkpoints, not training-loss claims."""
from network import *
import gc
device='mps';assert torch.backends.mps.is_available();rec=pd.read_csv(A/'inputs/records.csv');feat=np.load(A/'features/early_maps.npy',mmap_mode='r');meta=torch.tensor(np.load(A/'inputs/metadata.npy'));base=backbone();rows=[];ex=np.flatnonzero(rec.source.eq('query'))
for variant in ['tight','dual','dual_pool']:
 for fit in DAYS:
  net=Classifier(base,'tight' if variant=='tight' else 'dual').to(device);net.load_state_dict(torch.load(A/f'models/{variant}_fit{fit[-2:]}_s4201.pt',map_location=device,weights_only=True));net.eval();scores=[]
  with torch.inference_mode():
   for off in range(0,len(ex),24):
    ii=ex[off:off+24];scores.extend(net(torch.tensor(np.array(feat[ii],dtype=np.float32),device=device),meta[ii].to(device)).sigmoid().cpu().tolist())
  d=rec.iloc[ex].copy();d['score']=scores
  for scope,m in [('fit',d.day.eq(fit)),('held',d.day.ne(fit))]:rows.append(dict(variant=variant,fit_day=fit,scope=scope,n=int(m.sum()),ap=float(average_precision_score(d.loc[m,'target'],d.loc[m,'score']))))
  del net;gc.collect();torch.mps.empty_cache()
pd.DataFrame(rows).to_csv(A/'audit/fit_vs_held.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))

