"""Kaggle pretrained models evaluated with padding fitted only on opposite day."""
from common25 import *
from features23 import nms
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import joblib
G=A/'gpu/results';models=['rtmw','wilor_detector'];pool=pd.read_csv(V22/'inputs/reviewed_pool_hands.csv')
rr={m:pd.read_csv(G/m/'references.csv').fillna({'query_id':''}) for m in models}
manual=h[h.class_name.eq('nude_hands')].copy();manual['id']=['manual_'+str(i) for i in manual.index]
def refs_for(r,model,padding):
 d=rr[model];d=d[d.frame_id.eq(r.frame_id)&(d.query_id.eq('')|d.query_id.eq(r.id))].copy()
 if not len(d):return d
 b=d[B].to_numpy();wh=b[:,2:]-b[:,:2];b+=np.c_[-wh*padding,wh*padding];d[B]=np.clip(b,0,[1280,720,1280,720])
 return d
cal=[]
def calibrate(fit,model):
 g=manual[manual.day.eq(fit)]
 scores=[]
 for pad in [0.,.1,.2,.35]:
  vv=[]
  for r in g.itertuples():
   d=refs_for(r,model,pad);d=d[d.confidence.ge(.3)]
   vv.append(float(iou([getattr(r,c) for c in B],d[B]).max()) if len(d) else 0)
  scores.append((float(np.mean(vv)),pad))
 best=max(scores)
 cal.append(dict(fit=fit,model=model,padding=best[1],fit_mean_best_iou=best[0],grid=scores))
 return best[1]
def feature_rows(rows,model,padding):
 out=[]
 for r in rows.itertuples():
  f=dict(id=r.id,best_iou=0.,reference_conf=0.,area_ratio=1.,query_covered=0.,ref_covered=0.,hand_count=0.,votes=0.,center_distance=2.,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
  if r.class_name!='nude_hands':out.append(f);continue
  b=np.array([getattr(r,c) for c in B]);d=refs_for(r,model,padding);d=d[d.confidence.ge(.2)]
  if not len(d):out.append(f);continue
  rb=d[B].to_numpy();ov=iou(b,rb)[0];rank=ov*np.sqrt(d.confidence.to_numpy().clip(0,1));j=int(rank.argmax());best=rb[j];wh=b[2:]-b[:2]
  inter=np.maximum(0,np.minimum(b[2:],best[2:])-np.maximum(b[:2],best[:2])).prod();ar=wh.prod();ar2=(best[2:]-best[:2]).prod()
  independent=nms(d[d.confidence.ge(.4)],cut=.3);cent=(independent[['x1','y1']].to_numpy()+independent[['x2','y2']].to_numpy())/2
  f.update(best_iou=float(ov[j]),reference_conf=float(d.iloc[j].confidence),area_ratio=float(ar/ar2),query_covered=float(inter/ar),ref_covered=float(inter/ar2),hand_count=float((((cent>=b[:2])&(cent<=b[2:])).all(1)).sum()),votes=float((ov>.6).sum()),center_distance=float(np.linalg.norm(((b[:2]+b[2:])-(best[:2]+best[2:]))/(2*wh))),**dict(zip(['ref_'+x for x in B],best)))
  out.append(f)
 return pd.DataFrame(out).set_index('id')
cols=['best_iou','reference_conf','area_ratio','query_covered','ref_covered','hand_count','votes','center_distance']
for model in models:
 of=[];tf=[];preds={m:[] for m in ['logreg','catboost']};testpreds={m:[] for m in preds};detection=[]
 for fit,held in zip(DAYS,DAYS[::-1]):
  pad=calibrate(fit,model);o=feature_rows(q[q.day.eq(held)],model,pad);t=feature_rows(te,model,pad);tr=feature_rows(q[q.day.eq(fit)],model,pad)
  of.append(o);tf.append(t);o.to_csv(A/f'features/{model}_held_{fit}.csv');t.to_csv(A/f'features/{model}_test_{fit}.csv')
  for group,rows in [('official_manual',manual[manual.day.eq(held)]),('reviewed_pool',pool[pool.day.eq(held)])]:
   f=feature_rows(rows,model,pad);f['frame_id']=rows.frame_id.to_numpy();f['day']=held;f['source']=group;detection.append(f)
  trainmask=q[q.day.eq(fit)].class_name.eq('nude_hands').to_numpy();X=tr[cols].to_numpy()[trainmask];y=q[q.day.eq(fit)].error.to_numpy()[trainmask]
  for name,clf in [('logreg',make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=500))),('catboost',CatBoostClassifier(iterations=150,depth=3,learning_rate=.03,l2_leaf_reg=8,random_seed=25,verbose=False,thread_count=2,allow_writing_files=False))]:
   clf.fit(X,y);joblib.dump(clf,A/f'models/{model}_{name}_{fit}.joblib');preds[name].append(pd.Series(clf.predict_proba(o[cols].to_numpy())[:,1],index=o.index));testpreds[name].append(clf.predict_proba(t[cols].to_numpy())[:,1])
 fo=pd.concat(of).loc[q.id];ft=(tf[0]+tf[1])/2
 det=pd.concat(detection);det.to_csv(A/f'results/{model}_manual_detection.csv')
 print('DETECTION',model,det.groupby('source').best_iou.agg(['count','mean']).to_dict(),flush=True)
 for method in ['geometry_rescue','geometry_both','logreg','catboost']:
  for weight in [.15,.35]:
   scores=[]
   for split,rows,base,f in [('oof',q,base_oof,fo),('test',te,current,ft)]:
    g=rows.class_name.eq('nude_hands').to_numpy()&guard(split,rows,'nude_hands')&f.reference_conf.ge(.4).to_numpy()&f.hand_count.le(1).to_numpy()
    if method.startswith('geometry'):
     p=np.clip((.75-f.best_iou.to_numpy())/.5,0,1)
     if method=='geometry_rescue':g &= f.best_iou.ge(.65).to_numpy();p=np.minimum(p,base)
    else:p=pd.concat(preds[method]).loc[q.id].to_numpy() if split=='oof' else np.mean(testpreds[method],axis=0)
    scores.append(np.clip(base+weight*g*(p-base),0,1))
   register(f'{model}_{method}_w{weight}',*scores,dict(type='new_pretrained_hand_evidence',model=model,method=method,weight=weight,padding_fit='opposite-day official manual',ranking_fit='opposite-day real query labels only; no test labels',gpu='Kaggle Tesla T4',guard='V22 semantic/duplicate/structural veto + independent count'))
(A/'results/pose_calibration.json').write_text(json.dumps(cal,indent=2))
