"""Joint image/pose/seat boundary experts, trained within each day."""
from pathlib import Path
import numpy as np,pandas as pd,joblib,json
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from lightgbm import LGBMClassifier
A=Path('/kaggle/working/rascar_v22');O=A/'conditioned_person';O.mkdir(exist_ok=True)
d=pd.read_csv(A/'inputs/spatial_records.csv');f=np.load(A/'spatial/features.npy').astype('float32').reshape(len(d),-1)/np.sqrt(6)
outputs=[]
for fit,held in [('2026-10-25','2026-10-26'),('2026-10-26','2026-10-25')]:
 tr=d[d.split.eq('train')&d.day.eq(fit)&d.class_name.eq('person')];ev=d[d.class_name.eq('person')&(d.split.eq('test')|(d.split.eq('held')&d.day.eq(held)))]
 pca=PCA(n_components=48,svd_solver='randomized',random_state=2212);z=pca.fit_transform(f[tr.index]);ze=pca.transform(f[ev.index])
 geo=pd.read_csv(A/f'person_boundary_features_train_{fit}.csv').set_index('id').loc[tr.id]
 geoe=pd.concat([pd.read_csv(A/f'person_boundary_features_held_{fit}.csv'),pd.read_csv(A/f'person_boundary_features_test_{fit}.csv')]).set_index('id').loc[ev.id]
 cols=geo.columns.tolist();x=np.c_[geo.to_numpy(),z];xe=np.c_[geoe[cols].to_numpy(),ze]
 for family in ['cb4','cb6','lgb7']:
  if family.startswith('cb'):m=CatBoostClassifier(iterations=400,depth=int(family[-1]),learning_rate=.025,l2_leaf_reg=30,random_seed=2212,thread_count=2,verbose=False,allow_writing_files=False)
  else:m=LGBMClassifier(n_estimators=250,num_leaves=7,max_depth=3,learning_rate=.025,reg_lambda=30,min_child_samples=35,verbosity=-1,n_jobs=2,random_state=2212)
  m.fit(x,tr.error.astype(int),sample_weight=tr.weight)
  pred=m.predict_proba(xe)[:,1];oo=ev[['id','frame_id','day','class_name','split','error']].copy();oo['fit']=fit;oo['family']=family;oo['probability']=pred;outputs.append(oo)
  joblib.dump(dict(model=m,pca=pca,columns=cols,training_indices=tr.index.to_numpy()),O/f'{fit}_{family}.joblib',compress=3)
  pd.concat(outputs).to_csv(O/'predictions.csv',index=False)
  print('CONDITIONED_PERSON',fit,family,flush=True)
print('CONDITIONED_DONE',flush=True)
