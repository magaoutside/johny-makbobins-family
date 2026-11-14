"""Frame-cluster stability, exact AP, pair-order changes and submission integrity.
These are development diagnostics after repeated OOF use, not fresh holdout proof.
"""
from common36 import *

def main():
 tables=[pd.read_csv(p) for p in sorted((A/'results').glob('*experiments.csv')) if p.name!='all_experiments.csv']
 ex=pd.concat(tables,ignore_index=True);assert ex.experiment.is_unique
 ex.sort_values('pooled_ap',ascending=False).to_csv(A/'results/all_experiments.csv',index=False)
 checked=[]
 for _,r in ex.iterrows():
  test=pd.read_csv(A/f'results/submissions/submission_{r.experiment}.csv');oo=align(A/f'results/oof/{r.experiment}.csv',q)
  assert list(test.columns)==['id','score'] and len(test)==362 and test.id.is_unique and set(test.id)==set(te.id)
  assert np.isfinite(test.score).all() and test.score.between(0,1).all()
  assert abs(AP(q.error,oo.score)-r.pooled_ap)<1e-12
  checked.append(r.experiment)
 names=['multiscale_adaptive_person','multiscale_vote_c0.4_person_w0.1','dense_identity_vote_c0.55_person_w0.25','dense_identity_peak_c0.35_person_w0.25']
 if (A/'results/replicate_experiments.csv').exists():names+=pd.read_csv(A/'results/replicate_experiments.csv').experiment.tolist()
 if (A/'results/consensus_experiments.csv').exists():names+=pd.read_csv(A/'results/consensus_experiments.csv').experiment.tolist()
 if (A/'results/crowded_experiments.csv').exists():names+=pd.read_csv(A/'results/crowded_experiments.csv').experiment.tolist()
 for table in ['siglip_experiments.csv','pool_augmented_experiments.csv','strict_seed_experiments.csv','blend_experiments.csv','pose_guard_experiments.csv']:
  if (A/'results'/table).exists():names+=pd.read_csv(A/'results'/table).experiment.tolist()
 y=q.error.to_numpy();base=BASE['oof'];rng=np.random.default_rng(3611);frames=q.frame_id.unique();ids={f:np.flatnonzero(q.frame_id.eq(f)) for f in frames};boot=[]
 for _ in range(1500):
  bb=[]
  for day in DAYS:
   ff=q[q.day.eq(day)].frame_id.unique();bb.extend(ids[f] for f in rng.choice(ff,len(ff),replace=True))
  boot.append(np.concatenate(bb))
 summary=[];loo=[];inversions=[];test_ranks=[];p=np.flatnonzero(y==1);n=np.flatnonzero(y==0);old=np.sign(base[p,None]-base[n]);within=q.day.to_numpy()[p,None]==q.day.to_numpy()[n]
 for name in names:
  s=align(A/f'results/oof/{name}.csv',q).score.to_numpy();bt=[AP(y[ix],s[ix])-AP(y[ix],base[ix]) for ix in boot];ll=[]
  for f in frames:
   mask=~q.frame_id.eq(f).to_numpy();delta=AP(y[mask],s[mask])-AP(y[mask],base[mask]);ll.append(delta)
   loo.append(dict(experiment=name,omitted_frame=f,delta_ap=delta))
  new=np.sign(s[p,None]-s[n])
  for a,b in zip(*np.where(new!=old)):
   inversions.append(dict(experiment=name,error_id=q.iloc[p[a]].id,correct_id=q.iloc[n[b]].id,error_frame=q.iloc[p[a]].frame_id,correct_frame=q.iloc[n[b]].frame_id,old_margin=base[p[a]]-base[n[b]],new_margin=s[p[a]]-s[n[b]],improved=bool(new[a,b]>old[a,b]),within_day=bool(within[a,b])))
  tst=align(A/f'results/submissions/submission_{name}.csv',te).score.to_numpy();br=pd.Series(BASE['test']).rank(ascending=False,method='min').to_numpy();nr=pd.Series(tst).rank(ascending=False,method='min').to_numpy()
  test_ranks.append(te.assign(experiment=name,v31_score=BASE['test'],new_score=tst,v31_rank=br,new_rank=nr,rank_delta=nr-br))
  upper=np.triu(np.ones((len(te),len(te)),bool),1);rp=((np.sign(tst[:,None]-tst)!=np.sign(BASE['test'][:,None]-BASE['test']))&upper).sum()
  summary.append(dict(experiment=name,**metric(q,s),delta_ap=AP(y,s)-AP(y,base),bootstrap_low=np.quantile(bt,.025),bootstrap_high=np.quantile(bt,.975),bootstrap_fraction_positive=float(np.mean(np.array(bt)>0)),loo_delta_min=min(ll),loo_delta_max=max(ll),improved_pairs=int((new>old).sum()),worsened_pairs=int((new<old).sum()),within_day_improved=int(((new>old)&within).sum()),within_day_worsened=int(((new<old)&within).sum()),changed_test=int((abs(tst-BASE['test'])>1e-10).sum()),test_pair_reorders=int(rp),caveat='Post-selection development OOF; bootstrap does not account for algorithm/threshold selection.'))
 pd.DataFrame(summary).to_csv(A/'results/stability.csv',index=False);pd.DataFrame(loo).to_csv(A/'audit/leave_one_frame_out.csv',index=False);pd.DataFrame(inversions).to_csv(A/'audit/pair_order_changes.csv',index=False);pd.concat(test_ranks).to_csv(A/'audit/test_rank_changes.csv',index=False)
 for manifest in [json.loads((A/'models/training_manifest.json').read_text())]+[[json.loads(p.read_text()) for p in (A/'models').glob('dense_seed*.json')]]:
  for model in manifest:
   assert all(f[-15:-7]==model['fit_day'].replace('-','') for f in model['fit_frames'])
   assert not set(model['fit_frames'])&set(te.frame_id)
 frozen=hashlib.sha256(PATHS['test'].read_bytes()).hexdigest();assert frozen=='8720eb732639532bab49eff75b956a6bd4db21b96ede309ed901889777f4ab2c'
 (A/'results/checks.json').write_text(json.dumps(dict(validated_experiments=len(checked),v31_sha256=frozen,v31_unchanged=True,train_day_separation=True,submission_rows=362,test_labels_used=False,leaderboard_uploaded=False),indent=2))
 print(pd.DataFrame(summary).drop(columns='caveat').to_string(index=False),flush=True)

if __name__=='__main__':main()
