"""Replicate the narrow high-confidence identity policy without multiscale search."""
from evaluate_votes import *

def main():
 results=[];ensemble={s:[] for s in ROWS}
 for split,rows in ROWS.items():
  path=A/(f'results/oof/dense_identity_vote_c0.55_person_w0.25.csv' if split=='oof' else f'results/submissions/submission_dense_identity_vote_c0.55_person_w0.25.csv');ensemble[split].append(align(path,rows).score.to_numpy())
 for seed in [3602,3603]:
  fs,_=combine(f'dense_seed{seed}_identity','vote',.55);scores={}
  for split,rows in ROWS.items():
   valid,delta=fs[split];scores[split]=np.clip(BASE[split]+.25*valid*delta*rows.class_name.eq('person').to_numpy(),0,1);ensemble[split].append(scores[split])
  results.append(save(f'strict_person_seed{seed}',scores,'Fixed single-view confidence .55, same-head and upper-half gate, weight .25; independent seed without per-seed tuning.'))
 scores={s:np.mean(v,axis=0) for s,v in ensemble.items()};results.append(save('strict_person_three_seeds',scores,'Equal average of all three fixed single-view strict person policies, including original seed.'))
 pd.DataFrame(results).to_csv(A/'results/strict_seed_experiments.csv',index=False)

if __name__=='__main__':main()
