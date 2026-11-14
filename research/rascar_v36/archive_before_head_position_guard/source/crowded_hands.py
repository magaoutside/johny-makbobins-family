"""Test gentle proposal suppression for overlapping hands, across all three seeds.
Official manual example revealed two real GT hand boxes with IoU .57;
ordinary .45 NMS removes one even with perfect regression targets.
Retaining both proposals does not by itself prove they are different instances.
Existing duplicate/semantic rescue vetoes therefore remain active.
"""
from dense_votes import *
from evaluate_votes import collect,guard

def main():
 rec=pd.read_csv(V31/'inputs/dense_records.csv');feat=np.load(V31/'features/dense_grids.npy',mmap_mode='r');results=[];ensemble={s:[] for s in ROWS}
 for seed in [3601,3602,3603]:
  all_evidence={s:[] for s in ROWS}
  for fit in DAYS:
   name=f'dense_fit{fit[-2:]}' if seed==3601 else f'dense_seed{seed}_fit{fit[-2:]}'
   model=VoteHead();model.load_state_dict(torch.load(A/f'models/{name}.pt',weights_only=True));model.eval();out=[]
   ev=rec[rec.class_name.eq('nude_hands')&rec.split.isin(['oof','test'])&(rec.split.eq('test')|rec.day.ne(fit))]
   for off in range(0,len(ev),16):
    batch=ev.iloc[off:off+16];x=torch.from_numpy(np.array(feat[batch.feature_index],np.float32))
    with torch.inference_mode():pred=model(x)
    for j,(_,r) in enumerate(batch.iterrows()):
     for b in decode(tuple(v[j] for v in pred),r,suppression_iou=.85):
      if b['class_name']=='nude_hands' and b['decoder']=='vote':out.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,**b))
   raw=pd.DataFrame(out);raw.to_csv(A/f'features/crowded_seed{seed}_fit{fit[-2:]}_boxes.csv',index=False)
   for split,rows in ROWS.items():
    rr=rows[rows.day.ne(fit)] if split=='oof' else rows;d=collect(raw,rr,'vote',.55).set_index('id').loc[rr.id];d['fit_day']=fit;d.to_csv(A/f'features/crowded_seed{seed}_{split}_fit{fit[-2:]}.csv');all_evidence[split].append(d)
  scores={}
  for split,rows in ROWS.items():
   parts=all_evidence[split]
   if split=='oof':
    d=pd.concat(parts).loc[rows.id];target=np.clip((.85-d.overlap.to_numpy())/.5,0,1);valid=d.valid.to_numpy(bool)
   else:
    raw=np.stack([np.clip((.85-d.overlap.to_numpy())/.5,0,1) for d in parts]);delta=raw-BASE[split]
    valid=np.logical_and.reduce([d.valid.to_numpy(bool) for d in parts])&(np.all(delta>=0,0)|np.all(delta<=0,0));refs=[d[['ref_'+b for b in B]].to_numpy(float) for d in parts];valid&=np.array([iou(a,b)[0,0]>=.6 for a,b in zip(*refs)])
    target=np.where(np.all(delta>=0,0),np.min(raw,0),np.max(raw,0))
   delta=np.nan_to_num(target-BASE[split],nan=0.);valid&=(delta>=0)|guard(rows,split)
   scores[split]=np.clip(BASE[split]+.1*delta*valid*rows.class_name.eq('nude_hands').to_numpy(),0,1);ensemble[split].append(scores[split])
  results.append(save(f'crowded_hands_seed{seed}',scores,'Hand proposals retained until IoU .85; confidence .55 and weight .1 fixed; prior duplicate and semantic vetoes retained.'))
 scores={s:np.mean(v,axis=0) for s,v in ensemble.items()};results.append(save('crowded_hands_three_seeds',scores,'Equal average of all three crowded-hand seeds, no winner selection.'))
 pd.DataFrame(results).to_csv(A/'results/crowded_experiments.csv',index=False)

if __name__=='__main__':main()
