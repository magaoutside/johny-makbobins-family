from common22 import *
import sys
sys.path.insert(0,str(ROOT/'rascar_v21'))
from guard_quality import duplicate_veto
d=pd.read_csv(A/'gpu_results/spatial/predictions.csv')
for representation in ['inside','spatial']:
 for algorithm in ['logreg','svm']:
  dd=d[d.representation.eq(representation)&d.algorithm.eq(algorithm)]
  pp={split:dd[dd.split.eq('held' if split=='oof' else split)].groupby('id').probability.agg(['mean','max']).loc[rows.id] for split,rows in [('oof',q),('test',te)]}
  for cls in CL:
   for rule in ['blend','guarded']:
    for weight in [.15,.3]:
     outputs=[]
     for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
      pred=pp[split]['mean'].to_numpy();m=rows.class_name.eq(cls).to_numpy()
      if rule=='guarded':
       sem=align(ROOT/f'rascar_v16/semantic_both_{split}.csv',rows,'p_'+cls)
       structure=align(ROOT/f'rascar_v21/object_structure_{split}.csv',rows)
       m &= (sem>=.8)&(pp[split]['max'].to_numpy()<.5)&~duplicate_veto(rows,split=='test')&structure.distinct_instances.le(1).to_numpy()&~boolean(structure.ambiguous_duplicate)
       if cls=='person':
        f=align(ROOT/f'rascar_v20/replay_inputs/{split}.csv',rows)
        m &= f.mode_front.ge(.6).to_numpy()&f.visible_outside.lt(.01).to_numpy()&f.iou_q75.ge(.55).to_numpy()&f.width_ratio.le(1.5).to_numpy()
       target=np.minimum(base,pred)
      else:target=pred
      s=base.copy();s[m]=(1-weight)*base[m]+weight*target[m];outputs.append(s)
     register(f'dinov2_{representation}_{algorithm}_{cls}_{rule}_w{weight}',*outputs,dict(class_name=cls,representation=representation,algorithm=algorithm,rule=rule,weight=weight))
