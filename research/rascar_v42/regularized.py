"""Class-specific direct classifiers; no synthetic labels, frozen RGB backbone.
Controls target synthetic-distribution shift and small-data overfitting.
"""
from base42 import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import joblib
rec=pd.read_csv(A/'inputs/records.csv');visual=np.load(A/'features/frozen_vectors.npy');meta=np.load(A/'inputs/metadata.npy');q,te,h=inputs();outputs=[];manifest=[]
for fit in DAYS:
 ev=rec.source.eq('test')|(rec.source.eq('query')&rec.day.ne(fit));ex=np.flatnonzero(ev)
 for kind in ['query_visual','query_pool_visual','query_visual_confidence','query_metadata_confidence']:
  train=rec.day.eq(fit)&rec.source.eq('query')
  if kind=='query_pool_visual':train|=rec.day.eq(fit)&rec.source.eq('pool')
  assert not set(rec.loc[train,'frame_id'])&set(rec.loc[ev,'frame_id'])
  X=visual.copy() if 'visual' in kind else meta.copy()
  if 'confidence' in kind:X=np.c_[X,rec.confidence.fillna(.8)]
  score=np.empty(len(rec));models={}
  for c in CL:
   tr=train&rec.class_name.eq(c);ee=ev&rec.class_name.eq(c);y=rec.loc[tr,'target'].to_numpy();w=rec.loc[tr,'weight'].to_numpy()
   if len(np.unique(y))<2:
    # Laplace-smoothed training-day prevalence, no held-day labels.
    p=(np.sum(y*w)+1)/(sum(w)+2);score[ee]=p;models[c]=dict(constant=float(p));continue
   # Low-dimensional subspace fitted exclusively to this class and fit day.
   ncomp=min(12,int(tr.sum())-2,X.shape[1]);model=make_pipeline(StandardScaler(),PCA(n_components=ncomp,svd_solver='full'),LogisticRegression(C=.03,max_iter=500,random_state=4202))
   model.fit(X[tr],y,logisticregression__sample_weight=w);score[ee]=model.predict_proba(X[ee])[:,1];models[c]=model
  d=rec.loc[ev,['id','frame_id','class_name','day','source']].copy();d['score']=score[ev];d['fit_day']=fit;d['variant']=kind;d['seed']=4202;outputs.append(d)
  joblib.dump(models,A/f'models/{kind}_fit{fit[-2:]}.joblib');manifest.append(dict(variant=kind,fit_day=fit,train_frames=sorted(rec.loc[train,'frame_id'].unique()),synthetic_used=False,frozen_backbone=True))
  print(kind,fit,metrics(q[q.day.ne(fit)],d[d.source.eq('query')].set_index('id').loc[q.loc[q.day.ne(fit),'id'],'score']),flush=True)
pd.concat(outputs).to_csv(A/'results/regularized_predictions_s4202.csv',index=False);(A/'models/regularized_manifest.json').write_text(json.dumps(manifest,indent=2))
