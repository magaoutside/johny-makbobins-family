"""Compose disjoint-class experts; no new blend-weight optimization."""
from common36 import *

def main():
 results=[]
 for tag,person in [('person_multiscale_pool','multiscale_adaptive_person'),('person_strict_pool','dense_identity_vote_c0.55_person_w0.25'),('person_seedmean_pool','multiscale_adaptive_three_seeds'),('person_strict_seedmean_pool','strict_person_three_seeds')]:
  scores={}
  for split,rows in ROWS.items():
   source=lambda n:A/(f'results/oof/{n}.csv' if split=='oof' else f'results/submissions/submission_{n}.csv')
   p=align(source(person),rows).score.to_numpy();hands=align(source('dino_reviewed_pool_p0.15_w0.25'),rows).score.to_numpy()
   assert np.allclose(p[~rows.class_name.eq('person')],BASE[split][~rows.class_name.eq('person')],atol=1e-14)
   assert np.allclose(hands[~rows.class_name.eq('nude_hands')],BASE[split][~rows.class_name.eq('nude_hands')],atol=1e-14)
   scores[split]=p+hands-BASE[split]
  results.append(save(tag,scores,'Disjoint classes: '+person+' for person; DINO reviewed-pool hand veto; cup/tray frozen. Research only; Public unmeasured.'))
 pd.DataFrame(results).to_csv(A/'results/blend_experiments.csv',index=False)

if __name__=='__main__':main()
