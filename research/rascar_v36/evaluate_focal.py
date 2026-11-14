"""Sparse-center focal-loss controls, fixed before the later person gate."""
from evaluate_votes import *

def main():
 results=[];controls=[]
 for decoder in ['vote','peak']:
  for confidence in [.35,.55]:
   fs,cc=combine('center_focal',decoder,confidence);controls+=cc
   for family,classes in [('person',['person']),('hands',['nude_hands']),('tableware',['cup','tray']),('all',CL)]:
    scores={}
    for split,rows in ROWS.items():
     valid,delta=fs[split];scores[split]=np.clip(BASE[split]+.25*delta*valid*rows.class_name.isin(classes).to_numpy(),0,1)
    results.append(save(f'center_focal_{decoder}_c{confidence}_{family}_w0.25',scores,'Sparse class-specific control with focal loss to correct extreme center/background imbalance; same daily folds.'))
 pd.DataFrame(results).sort_values('pooled_ap',ascending=False).to_csv(A/'results/focal_experiments.csv',index=False);pd.DataFrame(controls).to_csv(A/'results/focal_localization.csv',index=False)

if __name__=='__main__':main()
