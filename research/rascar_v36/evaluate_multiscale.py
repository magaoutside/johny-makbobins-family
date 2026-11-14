"""Same-head, cross-context-scale instance consensus. No query-specific rules."""
from evaluate_votes import *
from itertools import combinations

def evidence(rows,split,fit,decoder,threshold,raw=None):
 if raw is None:raw=pd.read_csv(A/f'features/multiscale_fit{fit[-2:]}_boxes.csv')
 parts=[]
 for scale in [1.8,2.4,3.2]:
  d=collect(raw[raw.view_scale.eq(scale)],rows,decoder,.3 if decoder!='linear' else .5,identity=True).set_index('id').loc[rows.id]
  parts.append(d)
 out=[]
 for i,(_,r) in enumerate(rows.iterrows()):
  vals=[d.iloc[i] for d in parts];candidates=[]
  for ids in [(0,1,2),(0,1),(0,2),(1,2)]:
   vs=[vals[j] for j in ids]
   if not all(v.valid and v.confidence>=.3 for v in vs):continue
   refs=np.array([v[['ref_'+b for b in B]].to_numpy(float) for v in vs]);pair=iou(refs,refs);agreement=pair[np.triu_indices(len(ids),1)].min()
   confidence=float(np.mean([v.confidence for v in vs]))
   if agreement<.7 or confidence<threshold:continue
   ref=np.median(refs,axis=0)
   candidates.append((len(ids),agreement*confidence,ref,confidence,agreement,ids))
  entry=dict(id=r.id,valid=False,overlap=np.nan,confidence=0.,views=0,agreement=0.,chosen_views='',**{'ref_'+b:np.nan for b in B})
  if candidates:
   n,_,ref,confidence,agreement,ids=max(candidates,key=lambda x:x[:2])
   entry.update(valid=True,overlap=float(iou(r[B],ref)[0,0]),confidence=confidence,views=n,agreement=float(agreement),chosen_views=';'.join(str([1.8,2.4,3.2][j]) for j in ids),**{'ref_'+b:float(ref[k]) for k,b in enumerate(B)})
  out.append(entry)
 return pd.DataFrame(out)

def main():
 results=[]
 for decoder,thresholds in [('vote',[.4,.5]),('peak',[.4,.5]),('linear',[.65,.75])]:
  for threshold in thresholds:
   fs={}
   for split,rows in ROWS.items():
    parts=[]
    for fit in DAYS:
     rr=rows[rows.day.ne(fit)] if split=='oof' else rows
     d=evidence(rr,split,fit,decoder,threshold);d['fit_day']=fit;parts.append(d)
     d.to_csv(A/f'features/multiscale_{decoder}_c{threshold}_{split}_fit{fit[-2:]}.csv',index=False)
    if split=='oof':
     d=pd.concat(parts).set_index('id').loc[rows.id];target=np.clip((.85-d.overlap.to_numpy())/.5,0,1);valid=d.valid.to_numpy(bool)
    else:
     parts=[d.set_index('id').loc[rows.id] for d in parts];raw=np.stack([np.clip((.85-d.overlap.to_numpy())/.5,0,1) for d in parts]);delta=raw-BASE[split]
     valid=np.logical_and.reduce([d.valid.to_numpy(bool) for d in parts])&(np.all(delta>=0,0)|np.all(delta<=0,0))
     refs=[d[['ref_'+b for b in B]].to_numpy(float) for d in parts]
     valid&=np.array([iou(a,b)[0,0]>=.6 for a,b in zip(*refs)])
     target=np.where(np.all(delta>=0,0),np.min(raw,0),np.max(raw,0))
    fs[split]=(valid,np.nan_to_num(target-BASE[split],nan=0.))
   for weight in [.1,.25]:
    scores={}
    for split,rows in ROWS.items():
     valid,delta=fs[split];scores[split]=np.clip(BASE[split]+weight*delta*valid*rows.class_name.eq('person').to_numpy(),0,1)
    results.append(save(f'multiscale_{decoder}_c{threshold}_person_w{weight}',scores,'Two/three visual context scales agree (IoU>=.7), same independent head, both daily models agree on test.'))
 pd.DataFrame(results).sort_values('pooled_ap',ascending=False).to_csv(A/'results/multiscale_experiments.csv',index=False)

if __name__=='__main__':main()
