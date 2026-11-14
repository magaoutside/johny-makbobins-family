"""A reference must match the same independently detected head as the query.
Abstain on zero/multiple heads; no test IDs or hand-entered corrections.
"""
from evaluate_votes import *

def main():
 results=[];controls=[]
 for decoder in ['vote','peak']:
  for confidence in [.35,.55]:
   fs,cc=combine('dense_identity',decoder,confidence);controls+=cc
   for weight in [.1,.25]:
    scores={}
    for split,rows in ROWS.items():
     valid,delta=fs[split];active=valid&rows.class_name.eq('person').to_numpy()
     scores[split]=np.clip(BASE[split]+weight*delta*active,0,1)
    results.append(save(f'dense_identity_{decoder}_c{confidence}_person_w{weight}',scores,'Same independently detected head in query and reference; abstain for zero/multiple heads; class-specific dense vote head.'))
 pd.DataFrame(results).sort_values('pooled_ap',ascending=False).to_csv(A/'results/identity_experiments.csv',index=False)
 pd.DataFrame(controls).to_csv(A/'results/identity_localization.csv',index=False)

if __name__=='__main__':main()
