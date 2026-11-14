"""Confidence changes correction size only after multi-view identity agreement.
No new eligible rows beyond the fixed multiscale consensus policy.
"""
from common36 import *

def strong_mask(rows,split):
 parts=[]
 for fit in DAYS:
  rr=rows[rows.day.ne(fit)] if split=='oof' else rows
  strong=align(A/f'features/dense_identity_vote_c0.55_{split}_fit{fit[-2:]}.csv',rr)
  multi=align(A/f'features/multiscale_vote_c0.4_{split}_fit{fit[-2:]}.csv',rr)
  refs=[d[['ref_'+b for b in B]].to_numpy(float) for d in [strong,multi]]
  good=strong.valid.to_numpy(bool)&multi.valid.to_numpy(bool)
  good&=np.array([iou(a,b)[0,0]>=.7 for a,b in zip(*refs)])
  parts.append(pd.Series(good,index=rr.id))
 if split=='oof':return pd.concat(parts).loc[rows.id].to_numpy()
 return np.logical_and.reduce([p.loc[rows.id].to_numpy() for p in parts])

def main():
 scores={};eligibility=[]
 for split,rows in ROWS.items():
  name='multiscale_vote_c0.4_person_w0.1';path=A/f'results/oof/{name}.csv' if split=='oof' else A/f'results/submissions/submission_{name}.csv'
  original=align(path,rows).score.to_numpy();strong=strong_mask(rows,split)&rows.class_name.eq('person').to_numpy()
  delta=original-BASE[split];scores[split]=np.clip(BASE[split]+delta*np.where(strong,2.5,1.),0,1)
  eligibility.append(rows[['id','class_name']].assign(split=split,strong=strong,changed=abs(delta)>1e-10))
 pd.concat(eligibility).to_csv(A/'audit/strong_multiscale_gate.csv',index=False)
 results=[save('multiscale_adaptive_person',scores,'Weight .1 for same-head multi-context consensus; .25 only if high-confidence base-view voting confirms the same reference (IoU>=.7). No additional eligible rows.')]
 pd.DataFrame(results).to_csv(A/'results/composed_experiments.csv',index=False)

if __name__=='__main__':main()
