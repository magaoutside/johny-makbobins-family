"""Repeat the fixed person policy with two prespecified random seeds.
No threshold/weight/step tuning per seed. Keep individual results and a
three-seed average, including the original seed. All OOF models fit the other day.
"""
from dense_votes import *
from evaluate_multiscale import evidence
from evaluate_votes import collect

def train_predict(seed,fit):
 base=pd.read_csv(V31/'inputs/dense_records.csv');base['bank']=0;base['view_scale']=2.4
 empty=pd.read_csv(ROOT/'rascar_v33/inputs/empty_records.csv');empty['bank']=1
 multi=pd.read_csv(A/'inputs/multiscale_records.csv');multi['bank']=2
 rec=pd.concat([base,empty,multi],ignore_index=True)
 banks=[np.load(p,mmap_mode='r') for p in [V31/'features/dense_grids.npy',ROOT/'rascar_v33/features/empty_grids.npy',A/'features/multiscale_grids.npy']]
 def features(ids):return torch.from_numpy(np.stack([np.array(banks[int(rec.iloc[i].bank)][int(rec.iloc[i].feature_index)],np.float32) for i in ids]))
 train=rec[rec.split.isin(['manual','empty'])&rec.day.eq(fit)].index.to_numpy()
 ev=rec[rec.class_name.eq('person')&rec.split.isin(['oof','test'])&(rec.split.eq('test')|rec.day.ne(fit))].index.to_numpy()
 assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
 tt={i:targets(rec.loc[i],'dense') for i in train};groups=[train[rec.loc[train].split.eq(s)] for s in ['manual','empty']];probs=[]
 for group in groups:
  counts=rec.loc[group].groupby('frame_id').size();pr=np.array([1/counts[f] for f in rec.loc[group,'frame_id']]);probs.append(pr/pr.sum())
 torch.manual_seed(seed);rng=np.random.default_rng(seed);model=VoteHead();opt=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.02);start=time.time()
 for step in range(700):
  ids=np.concatenate([rng.choice(g,n,replace=True,p=p) for g,n,p in zip(groups,[6,2],probs)])
  target=tuple(torch.from_numpy(np.stack([tt[i][k] for i in ids])) for k in range(4))
  loss=loss_fn(model(features(ids)),target,'dense');assert torch.isfinite(loss)
  opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
 name=f'dense_seed{seed}_fit{fit[-2:]}';torch.save(model.state_dict(),A/f'models/{name}.pt');model.eval();out=[]
 for off in range(0,len(ev),16):
  ids=ev[off:off+16]
  with torch.inference_mode():pred=model(features(ids))
  for j,i in enumerate(ids):
   r=rec.loc[i]
   for b in decode(tuple(v[j] for v in pred),r):
    if b['class_name']=='person' and b['decoder']=='vote':out.append(dict(record_id=r.id,record_split=r.split,frame_id=r.frame_id,fit_day=fit,view_scale=r.view_scale,**b))
 raw=pd.DataFrame(out);raw.to_csv(A/f'features/{name}_boxes.csv',index=False)
 manifest=dict(seed=seed,fit_day=fit,steps=700,fit_frames=sorted(rec.loc[train,'frame_id'].unique()),train_crops=len(train),policy_fixed_before_replicates=True,test_labels_used=False,seconds=time.time()-start)
 (A/f'models/{name}.json').write_text(json.dumps(manifest,indent=2));print('REPLICATE_DONE',seed,fit,len(out),round(time.time()-start,1),flush=True)
 return raw

def predict(seed,raws):
 scores={}
 for split,rows in ROWS.items():
  parts=[]
  for fit in DAYS:
   rr=rows[rows.day.ne(fit)] if split=='oof' else rows;raw=raws[fit]
   d=evidence(rr,split,fit,'vote',.4,raw).set_index('id').loc[rr.id]
   strong=collect(raw[raw.view_scale.eq(2.4)],rr,'vote',.55,identity=True).set_index('id').loc[rr.id]
   refs=[z[['ref_'+b for b in B]].to_numpy(float) for z in [d,strong]]
   d['strong']=d.valid.to_numpy(bool)&strong.valid.to_numpy(bool)&np.array([iou(a,b)[0,0]>=.7 for a,b in zip(*refs)])
   d['fit_day']=fit;d.to_csv(A/f'features/replicate_seed{seed}_{split}_fit{fit[-2:]}.csv');parts.append(d)
  if split=='oof':
   d=pd.concat(parts).loc[rows.id];target=np.clip((.85-d.overlap.to_numpy())/.5,0,1);valid=d.valid.to_numpy(bool);strong=d.strong.to_numpy(bool)
  else:
   parts=[d.loc[rows.id] for d in parts];raw=np.stack([np.clip((.85-d.overlap.to_numpy())/.5,0,1) for d in parts]);delta=raw-BASE[split]
   valid=np.logical_and.reduce([d.valid.to_numpy(bool) for d in parts])&(np.all(delta>=0,0)|np.all(delta<=0,0))
   refs=[d[['ref_'+b for b in B]].to_numpy(float) for d in parts];valid&=np.array([iou(a,b)[0,0]>=.6 for a,b in zip(*refs)])
   strong=np.logical_and.reduce([d.strong.to_numpy(bool) for d in parts]);target=np.where(np.all(delta>=0,0),np.min(raw,0),np.max(raw,0))
  scores[split]=np.clip(BASE[split]+np.where(strong,.25,.1)*valid*rows.class_name.eq('person').to_numpy()*np.nan_to_num(target-BASE[split],nan=0.),0,1)
 return scores

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--reuse-boxes',action='store_true');args=ap.parse_args()
 results=[];ensemble={s:[] for s in ROWS}
 for split,rows in ROWS.items():
  path=A/('results/oof/multiscale_adaptive_person.csv' if split=='oof' else 'results/submissions/submission_multiscale_adaptive_person.csv');ensemble[split].append(align(path,rows).score.to_numpy())
 for seed in [3602,3603]:
  raws={fit:(pd.read_csv(A/f'features/dense_seed{seed}_fit{fit[-2:]}_boxes.csv') if args.reuse_boxes else train_predict(seed,fit)) for fit in DAYS};scores=predict(seed,raws)
  results.append(save(f'multiscale_adaptive_seed{seed}',scores,'Frozen V36 hyperparameters; independent initialization and training resampling; no seed selection.'))
  for s in ROWS:ensemble[s].append(scores[s])
 scores={s:np.mean(a,axis=0) for s,a in ensemble.items()}
 results.append(save('multiscale_adaptive_three_seeds',scores,'Equal average of all three prespecified seeds; no winner selection.'))
 pd.DataFrame(results).to_csv(A/'results/replicate_experiments.csv',index=False)

if __name__=='__main__':main()
