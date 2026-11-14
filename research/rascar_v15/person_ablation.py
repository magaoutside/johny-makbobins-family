from common import *
from scipy.special import expit
register('baseline_v8',v8,v8t)
register('baseline_v14',q.score,current)
cb=pd.read_csv(E/'stage1/oof/catboost_d2_blend1.csv').set_index('id').loc[q.id,'score'].to_numpy()
cbt=pd.read_csv(E/'stage1/candidates/submission_catboost_d2_blend1.csv').set_index('id').loc[te.id,'score'].to_numpy()
f=pd.read_csv(E/'stage1/raw_frozen_train_features.csv').set_index('id').loc[q.id];ft=pd.read_csv(E/'stage1/raw_frozen_test_features.csv').set_index('id').loc[te.id]
for weight in [0,.25,.5,.75]:
 s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('person');mt=te.class_name.eq('person')
 s[m]=(1-weight)*v8[m]+weight*cb[m];t[mt]=(1-weight)*v8t[mt]+weight*cbt[mt]
 register(f'person_cb_{weight:g}',s,t,{'person_cb_weight':weight})
for gate in ['agreement','no_downward']:
 s=q.score.to_numpy().copy();t=current.copy()
 for rows,feat,base,cbs,out in [(q,f,v8,cb,s),(te,ft,v8t,cbt,t)]:
  m=rows.class_name.eq('person').to_numpy()
  w=.5*expit((feat.best_iou.to_numpy()-.5)/.1)*np.exp(-abs(feat.sam_wevis_difference.to_numpy())*3) if gate=='agreement' else np.full(len(rows),.5)
  z=(1-w)*base+w*cbs
  if gate=='no_downward':z=np.maximum(z,base)
  out[m]=z[m]
 register('person_gate_'+gate,s,t,{'gate':gate})
