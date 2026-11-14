"""Reproduce V22 ranking from preserved model outputs. No fitting or network."""
from pathlib import Path
import numpy as np,pandas as pd
A=Path(__file__).resolve().parent
def truth(x):return x.astype(str).str.lower().isin(['true','1','1.0']).to_numpy()
def rank(x,test=False,control=False):
 b=x.baseline.to_numpy();hand=x.structural_hand.to_numpy()
 cup=x.new_cup.to_numpy() if control else np.minimum(x.old_cup,x.new_cup).to_numpy()
 s=b+(cup-b)+(hand-b)
 veto=np.minimum(x.quality_logreg,x.quality_svm).to_numpy()>=.75
 gate=x.class_name.eq('nude_hands').to_numpy()&truth(x.zoom_assigned)&x.zoom_instances.eq(1).to_numpy()&x.zoom_overlap.ge(.65).to_numpy()&x.zoom_confidence.ge(.5).to_numpy()&x.zoom_coherence.ge(.75).to_numpy()&x.semantic_hand.ge(.75).to_numpy()&~truth(x.old_duplicate_veto)&~truth(x.zoom_duplicate_loser)&~veto&~(hand>b+1e-9)&x.independent_hand_centers.lt(2).to_numpy()
 if test:gate &= x.zoom_folds.ge(2).to_numpy()
 target=np.clip((.85-x.zoom_overlap.to_numpy())/.5,0,1);s[gate]-=.5*np.maximum(s[gate]-target[gate],0)
 return x[['id']].assign(score=s)
if __name__=='__main__':
 for split in ['oof','test']:
  x=pd.read_csv(A/f'replay_inputs/{split}.csv')
  for control,name in [(False,'final_reviewed'),(True,'validation_reviewed')]:
   got=rank(x,split=='test',control)
   path=A/f'results/{"submissions/submission_" if split=="test" else "oof/"}{name}.csv'
   expected=pd.read_csv(path).set_index('id').loc[got.id,'score'].to_numpy()
   error=float(np.max(abs(got.score.to_numpy()-expected)));assert error<1e-12
   assert len(got)==(362 if split=='test' else 198) and got.id.is_unique and got.score.between(0,1).all()
   got.to_csv(A/f'replayed_{name}_{split}.csv',index=False);print(split,name,'max_abs_error',error)
