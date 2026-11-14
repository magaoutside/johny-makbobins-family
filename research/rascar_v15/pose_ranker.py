from common import *
import joblib
from catboost import CatBoostClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
raw=pd.read_csv(E/'stage1/raw_frozen_train_features.csv').set_index('id').loc[q.id].reset_index(drop=True);rawt=pd.read_csv(E/'stage1/raw_frozen_test_features.csv').set_index('id').loc[te.id].reset_index(drop=True)
pf=pd.read_csv(A/'pose_oof_features.csv').set_index('id').loc[q.id].reset_index(drop=True);pft=pd.read_csv(A/'pose_test_features.csv').set_index('id').loc[te.id].reset_index(drop=True)
X=pd.concat([raw,pf],axis=1);XT=pd.concat([rawt,pft],axis=1)
for key,ctor in [('catboost',lambda:CatBoostClassifier(iterations=250,depth=2,learning_rate=.03,l2_leaf_reg=10,random_seed=17,verbose=False,allow_writing_files=False,thread_count=2)),('logreg',lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=2000)))]:
 pred=np.zeros(len(q));tp=[]
 for fit,held in zip(DAYS,DAYS[::-1]):
  fi=q.day.eq(fit);ho=q.day.eq(held);model=ctor();model.fit(X[fi],q.loc[fi,'error']);pred[ho]=model.predict_proba(X[ho])[:,1];tp.append(model.predict_proba(XT)[:,1]);joblib.dump(model,A/f'pose_ranker_{key}_{fit}.joblib')
 pt=np.mean(tp,axis=0);pd.DataFrame({'id':q.id,'score':pred}).to_csv(A/f'pose_ranker_{key}_oof.csv',index=False);pd.DataFrame({'id':te.id,'score':pt}).to_csv(A/f'pose_ranker_{key}_test.csv',index=False)
 for w in [.25,.5]:
  s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('person');mt=te.class_name.eq('person');s[m]=(1-w)*s[m]+w*pred[m];t[mt]=(1-w)*t[mt]+w*pt[mt];register(f'pose_ranker_{key}_{w:g}',s,t,dict(weight=w,features='raw frozen geometry and frozen pose; no fitted semantic feature leakage',depth=2 if key=='catboost' else None))
