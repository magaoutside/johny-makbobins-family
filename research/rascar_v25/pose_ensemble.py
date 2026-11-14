from common25 import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
import joblib
import pose_lib
calibration=json.loads((A/'results/pose_calibration_padwide.json').read_text())
models=['rtmw_padwide','wilor_detector_padwide','rtmwx_bgr_padwide','rtmwx_rgb_padwide']
cols=['best_iou','reference_conf','area_ratio','query_covered','ref_covered','hand_count','center_distance']
fo=pd.DataFrame(index=q.id);ft=pd.DataFrame(index=te.id)
for m in models:
 o=pd.concat([pd.read_csv(A/f'features/{m}_held_{fit}.csv') for fit in DAYS]).set_index('id').loc[q.id]
 t=sum([align(A/f'features/{m}_test_{fit}.csv',te) for fit in DAYS])/2
 for c in cols:fo[m+'_'+c]=o[c];ft[m+'_'+c]=t[c]
for f in [fo,ft]:
 z=f[[m+'_best_iou' for m in models]];f['iou_max']=z.max(axis=1);f['iou_min']=z.min(axis=1);f['iou_spread']=z.max(axis=1)-z.min(axis=1)
fo.to_csv(A/'features/pose_joint_oof.csv');ft.to_csv(A/'features/pose_joint_test.csv')
classifiers={'logreg':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500)),
 'catboost':lambda:CatBoostClassifier(iterations=150,depth=2,learning_rate=.03,l2_leaf_reg=12,thread_count=2,verbose=False,allow_writing_files=False,random_seed=25),
 'extratrees':lambda:ExtraTreesClassifier(n_estimators=300,max_depth=4,min_samples_leaf=3,n_jobs=2,random_state=25)}
for name,factory in classifiers.items():
 op=np.zeros(len(q));tests=[]
 for fit,held in zip(DAYS,DAYS[::-1]):
  tr=q.day.eq(fit).to_numpy()&q.class_name.eq('nude_hands').to_numpy();va=q.day.eq(held).to_numpy()
  ff=pd.DataFrame(index=q.id[tr]);tt=pd.DataFrame(index=te.id)
  for m in models:
   pad=next(z['padding'] for z in calibration if z['fit']==fit and z['model']==m)
   xx=pose_lib.feature_rows(q[tr],m,pad);yy=align(A/f'features/{m}_test_{fit}.csv',te)
   for c in cols:ff[m+'_'+c]=xx[c];tt[m+'_'+c]=yy[c]
  for f in [ff,tt]:
   z=f[[m+'_best_iou' for m in models]];f['iou_max']=z.max(axis=1);f['iou_min']=z.min(axis=1);f['iou_spread']=z.max(axis=1)-z.min(axis=1)
  ff.to_csv(A/f'features/pose_joint_train_{fit}.csv');tt.to_csv(A/f'features/pose_joint_test_{fit}.csv')
  model=factory();model.fit(ff.to_numpy(),q.error.to_numpy()[tr]);op[va]=model.predict_proba(fo.to_numpy()[va])[:,1];tests.append(model.predict_proba(tt.to_numpy())[:,1]);joblib.dump(model,A/f'models/pose_joint_{name}_{fit}.joblib')
 tp=np.mean(tests,axis=0)
 for mode in ['both','rescue','confirmed_rescue']:
  for weight in [.2,.5]:
   out=[]
   for split,rows,base,f,p in [('oof',q,base_oof,fo,op),('test',te,current,ft,tp)]:
    g=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')
    if mode=='confirmed_rescue':g &= f.iou_max.ge(.6).to_numpy()&f.iou_min.ge(.3).to_numpy()
    if mode!='both':p=np.minimum(p,base)
    out.append(np.clip(base+weight*g*(p-base),0,1))
   register(f'pose_joint_{name}_{mode}_w{weight}',*out,dict(type='new_pose_evidence_ensemble',model=name,mode=mode,weight=weight,feature_count=len(fo.columns),train='opposite-day real hand query labels, no baseline score feature, no test labels'))
