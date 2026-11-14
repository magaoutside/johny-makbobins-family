"""Direct score controls: identical labels/splits; frozen RGB or geometry only."""
from base42 import *
import argparse,time,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier
rec=pd.read_csv(A/'inputs/records.csv');visual=np.load(A/'features/frozen_vectors.npy');meta=np.load(A/'inputs/metadata.npy');q,te,h=inputs();outputs=[];manifests=[]
for fit in DAYS:
 train=rec.day.eq(fit)&rec.source.isin(['query','synthetic','pool']);ev=rec.source.eq('test')|(rec.source.eq('query')&rec.day.ne(fit));assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
 w=np.zeros(len(rec))
 for source,budget in [('query',.55),('synthetic',.3),('pool',.15)]:
  m=train&rec.source.eq(source);counts=rec.loc[m].groupby('frame_id').id.transform('size').to_numpy();ww=rec.loc[m,'weight'].to_numpy()/counts;w[m]=budget*ww/ww.sum()
 w=w[train]*train.sum();y=rec.loc[train,'target'].to_numpy();start=time.time()
 for name,X in [('metadata_logistic',meta),('frozen_logistic',visual)]:
  model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=500,random_state=4201))
  model.fit(X[train],y,logisticregression__sample_weight=w);score=model.predict_proba(X[ev])[:,1]
  d=rec.loc[ev,['id','frame_id','class_name','day','source']].copy();d['score']=score;d['variant']=name;d['fit_day']=fit;d['seed']=4201;outputs.append(d)
  joblib.dump(model,A/f'models/{name}_fit{fit[-2:]}.joblib');print(name,fit,metrics(q[q.day.ne(fit)],d[d.source.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True)
 # PCA is fitted only to fit-day rows. No baseline/reference-model scores.
 scale=StandardScaler();x=scale.fit_transform(visual[train]);pca=PCA(n_components=64,random_state=4201,svd_solver='randomized');xx=pca.fit_transform(x);xe=pca.transform(scale.transform(visual[ev]));xx=np.c_[xx,meta[train]];xe=np.c_[xe,meta[ev]]
 model=CatBoostClassifier(iterations=400,depth=4,learning_rate=.04,l2_leaf_reg=20,loss_function='Logloss',random_seed=4201,thread_count=3,verbose=False,allow_writing_files=False)
 model.fit(xx,y,sample_weight=w);d=rec.loc[ev,['id','frame_id','class_name','day','source']].copy();d['score']=model.predict_proba(xe)[:,1];d['variant']='frozen_catboost';d['fit_day']=fit;d['seed']=4201;outputs.append(d)
 joblib.dump(dict(scale=scale,pca=pca,model=model),A/f'models/frozen_catboost_fit{fit[-2:]}.joblib');manifests.append(dict(fit_day=fit,train_frames=sorted(rec.loc[train,'frame_id'].unique()),fit_only_pca=True,seconds=time.time()-start))
 print('frozen_catboost',fit,metrics(q[q.day.ne(fit)],d[d.source.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True)
pd.concat(outputs).to_csv(A/'results/frozen_predictions_s4201.csv',index=False);(A/'models/frozen_manifest.json').write_text(json.dumps(manifests,indent=2))

