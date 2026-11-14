"""Abstain when any of the three independently trained replicas disagrees.
Both direction and the proposed person rectangle must agree across seeds.
The smallest proposed score change sets the conservative correction.
"""
from common36 import *

def main():
 names=['multiscale_adaptive_person','multiscale_adaptive_seed3602','multiscale_adaptive_seed3603'];scores={};audit=[]
 for split,rows in ROWS.items():
  proposals=[]
  for name in names:
   path=A/(f'results/oof/{name}.csv' if split=='oof' else f'results/submissions/submission_{name}.csv');proposals.append(align(path,rows).score.to_numpy()-BASE[split])
  ds=np.stack(proposals);same=(np.all(ds>1e-10,0)|np.all(ds< -1e-10,0));good_by_fit=[]
  for fit in DAYS:
   rr=rows[rows.day.ne(fit)] if split=='oof' else rows
   paths=[A/f'features/multiscale_vote_c0.4_{split}_fit{fit[-2:]}.csv']+[A/f'features/replicate_seed{seed}_{split}_fit{fit[-2:]}.csv' for seed in [3602,3603]]
   refs=np.stack([align(p,rr)[['ref_'+b for b in B]].to_numpy(float) for p in paths]);agree=[]
   for i in range(len(rr)):
    ov=iou(refs[:,i],refs[:,i]);agree.append(bool(np.isfinite(ov).all() and ov[np.triu_indices(3,1)].min()>=.7))
   good_by_fit.append(pd.Series(agree,index=rr.id))
  good=pd.concat(good_by_fit).loc[rows.id].to_numpy() if split=='oof' else np.logical_and.reduce([z.loc[rows.id].to_numpy() for z in good_by_fit])
  delta=np.sign(ds[0])*np.min(abs(ds),axis=0)*same*good;scores[split]=np.clip(BASE[split]+delta,0,1)
  audit.append(rows[['id','class_name']].assign(split=split,same_direction=same,reference_agreement=good,delta=delta))
 pd.concat(audit).to_csv(A/'audit/seed_consensus_gates.csv',index=False)
 out=save('multiscale_three_seed_consensus',scores,'Three fixed seeds agree in direction and rectangle IoU>=.7; all previous identity and multi-view gates; minimum correction magnitude.')
 pd.DataFrame([out]).to_csv(A/'results/consensus_experiments.csv',index=False)

if __name__=='__main__':main()
