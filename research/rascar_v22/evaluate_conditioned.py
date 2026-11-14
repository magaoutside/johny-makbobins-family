from common22 import *
from sklearn.metrics import average_precision_score
d=pd.read_csv(A/'gpu_results/conditioned_person/predictions.csv')
for family in d.family.unique():
 dd=d[d.family.eq(family)];outputs={}
 for split,rows in [('oof',q),('test',te)]:
  p=dd[dd.split.eq('held' if split=='oof' else 'test')].groupby('id').probability.agg(['mean','max','min']).reindex(rows.id).fillna(0)
  if split=='oof':
   f=pd.concat([pd.read_csv(A/f'person_boundary_features_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[rows.id]
   m=rows.class_name.eq('person');print(family,'person AP',average_precision_score(rows[m].error,p['mean'][m.to_numpy()]))
  else:
   tt=[pd.read_csv(A/f'person_boundary_features_test_{fit}.csv').set_index('id').loc[rows.id] for fit in DAYS];f=(tt[0]+tt[1])/2
  outputs[split]=(p,f)
 for rule in ['blend','guarded','penalty']:
  for weight in [.15,.3]:
   scores=[]
   for split,rows,base in [('oof',q,q.score.to_numpy()),('test',te,current)]:
    p,f=outputs[split];pred=p['mean'].to_numpy();m=rows.class_name.eq('person').to_numpy()&f.eligible.eq(1).to_numpy()&f.edge.eq(0).to_numpy()
    if rule!='blend':
     penalty=(pred>base)&(p['min'].to_numpy()>.65)&f.support.ge(3).to_numpy()&f.template_valid.lt(.4).to_numpy()&f.template_height_ratio.lt(.85).to_numpy()
     rescue=(pred<base)&(p['max'].to_numpy()<.35)&f.upper_outside.lt(.04).to_numpy()&f.template_iou_max.ge(.65).to_numpy()
     m &= penalty if rule=='penalty' else penalty|rescue
    s=base.copy();s[m]=(1-weight)*base[m]+weight*pred[m];scores.append(s)
   register(f'conditioned_{family}_{rule}_w{weight}',*scores,dict(family=family,rule=rule,weight=weight,features='DINOv2 pooled patch PCA48 + pose + fit-day LOFO seat geometry'))
