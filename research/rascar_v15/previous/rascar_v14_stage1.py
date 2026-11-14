
from pathlib import Path
import os,sys,json,hashlib,subprocess,traceback,time
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
E=Path('/kaggle/working/rascar_v14');O=Path('/kaggle/working/oof_validation_v11');R=E/'v8_replay/sam31_v8';Y=E/'yolo26_clean'
A=E/'stage1';A.mkdir(exist_ok=True);(A/'models').mkdir(exist_ok=True);(A/'oof').mkdir(exist_ok=True);(A/'candidates').mkdir(exist_ok=True)
sys.path[:0]=[str(O),str(Y)]
from frozen_scoring import score_refs_v4
from oof_validation import Validator,DayBank,combine,FIXED
for package in ['catboost','lightgbm']:
 try:__import__(package)
 except ImportError:subprocess.run([sys.executable,'-m','pip','install','-q',package],check=True)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
B=['x1','y1','x2','y2'];CL=['person','cup','nude_hands','tray'];DAYS=['2026-10-25','2026-10-26']
v=Validator(O);q=v.train[v.train.evaluated].copy().reset_index(drop=True);base=pd.read_csv(R/'selected_train_scores_v8.csv').set_index('id').loc[q.id].reset_index()
test=pd.read_csv(O/'test.csv');tb=pd.read_csv(R/'selected_test_scores_v8.csv').set_index('id').loc[test.id].reset_index()
refs=pd.read_csv(R/'fused_train_references.csv');trefs=pd.read_csv(R/'fused_test_references.csv');zoom=pd.read_csv(R/'zoom_train_boxes.csv');tzoom=pd.read_csv(R/'zoom_test_boxes.csv')
h=pd.read_csv(O/'human_boxes_train.csv');h.class_name=h.class_name.replace(v.policy['aliases']);h=h.merge(v.train[['frame_id','day']].drop_duplicates(),on='frame_id',validate='many_to_one')
h[B]=np.column_stack([(h.cx-h.w/2)*1280,(h.cy-h.h/2)*720,(h.cx+h.w/2)*1280,(h.cy+h.h/2)*720])
assert np.allclose(base.score,pd.read_csv(O/'baseline_previous_oof.csv').set_index('id').loc[q.id,'score'],atol=1e-12)
plan={'seed':17,'folds':[[a,b] for a,b in zip(DAYS,DAYS[::-1])],'metric':'pooled AP; separate day and class AP','baseline_lb':.90481,'preserve_all_trials':True,'public_lb_new_candidates':None,'historical_limit':'Two development days. All results exploratory; no unseen third-day estimate. Candidate selection uses these OOF results.','table_policy':'Tables are context only; query boxes are never edited. Corrected model references change error-ranking evidence.','metadata_training':'Raw frozen-model features; any retrieval features recomputed using training-day bank and leave-frame/context-out on training rows. No YOLO OOF features used to fit the stacker.'}
(A/'plan.json').write_text(json.dumps(plan,indent=2));q[['id','frame_id','class_name','day','error']].to_csv(A/'fold_ledger.csv',index=False)
records=[];predictions={};failures=[]
def metric(s, rows=q):
 return {'pooled_ap':float(average_precision_score(rows.error,s)),**{'day_'+d[-2:]:float(average_precision_score(rows.loc[g.index,'error'],np.asarray(s)[g.index])) for d,g in rows.groupby('day')},**{'ap_'+c:float(average_precision_score(rows.loc[g.index,'error'],np.asarray(s)[g.index])) for c,g in rows.groupby('class_name')}}
def register(name,s,details=None):
 s=np.asarray(s,float);assert s.shape==(len(q),) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
 record={'experiment':name,**metric(s),'details':json.dumps(details or {},sort_keys=True)};records.append(record);predictions[name]=s
 q[['id','frame_id','class_name','day','error']].assign(score=s).to_csv(A/'oof'/f'{name}.csv',index=False)
 pd.DataFrame(records).to_csv(A/'experiment_metrics.csv',index=False)
 print('EXPERIMENT',name,round(record['pooled_ap'],8),round(record['day_25'],8),round(record['day_26'],8),flush=True)
def candidate(name,s):
 s=np.asarray(s,float);assert len(s)==362 and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
 z=test[['id']].assign(score=s);assert z.id.is_unique and set(z.id)==set(pd.read_csv(O/'sample_submission.csv').id)
 z=pd.read_csv(O/'sample_submission.csv')[['id']].merge(z,on='id',validate='one_to_one');p=A/'candidates'/f'submission_{name}.csv';z.to_csv(p,index=False)
 return {'experiment':name,'filename':str(p.relative_to(E)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'public_lb':None}
register('v8_baseline',base.score)
# Class-specific blends reuse independent day-fold YOLO predictions.
for threshold in [.05,.1,.25]:
 pieces=[]
 for fit,held in zip(DAYS,DAYS[::-1]):
  yr=pd.read_csv(Y/f'fit_{fit}/heldout_references.csv');sub=q[q.day.eq(held)].drop(columns='error')
  pieces.append(score_refs_v4(yr[yr.confidence.ge(threshold)],sub)[['id','score']])
 ys=pd.concat(pieces).set_index('id').loc[q.id,'score'].to_numpy()
 pd.DataFrame({'id':q.id,'yolo_score':ys}).to_csv(A/f'yolo_conf_{threshold:g}.csv',index=False)
 for scope in ['cup_tray','cup_only','tray_only']:
  m=q.class_name.isin({'cup_tray':['cup','tray'],'cup_only':['cup'],'tray_only':['tray']}[scope]).to_numpy()
  for w in [.1,.2]:
   s=base.score.to_numpy().copy();s[m]=(1-w)*s[m]+w*ys[m]
   register(f'yolo_{scope}_w{w:g}_conf{threshold:g}',s,{'confidence':threshold,'weight':w,'classes':scope})
def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 inter=np.prod(np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])),axis=2)
 aa=np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=1);bb=np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=1)
 return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)
def table_model(fitting):
 tables=fitting[fitting.class_name.eq('table')];boxes=tables[B].to_numpy();centers=(boxes[:,:2]+boxes[:,2:])/2
 km=KMeans(n_clusters=6,n_init=10,random_state=17).fit(centers);ts=np.stack([np.median(boxes[km.labels_==i],axis=0) for i in range(6)])
 ts=ts[np.lexsort((ts[:,0],ts[:,1]))]
 people=fitting[fitting.class_name.eq('person')][B].to_numpy();pc=(people[:,:2]+people[:,2:])/2;tc=(ts[:,:2]+ts[:,2:])/2;tw=ts[:,2:]-ts[:,:2]
 assigned=np.argmin(np.sum(((pc[:,None]-tc[None])/np.maximum(tw[None],1))**2,axis=2),axis=1)
 return ts,people,assigned
models={d:table_model(h[h.day.eq(d)]) for d in DAYS};alltables=table_model(h)
for d,(ts,people,assigned) in models.items():
 (A/f'table_prior_fit_{d}.json').write_text(json.dumps({'tables':ts.tolist(),'person_boxes':people.tolist(),'person_table':assigned.tolist(),'fit_day':d},indent=2))
def transform_refs(raw,tm,mode,strength):
 out=raw.copy();ts,people,assigned=tm;changed=[]
 for ix,row in out[out.class_name.eq('person')].iterrows():
  b=row[B].to_numpy(float);w=max(b[2]-b[0],1);height=max(b[3]-b[1],1);cx=(b[0]+b[2])/2;best=None
  for j,t in enumerate(ts):
   tw=t[2]-t[0];th=t[3]-t[1];overlap=max(0,min(b[2],t[2])-max(b[0],t[0]))/min(w,tw)
   gapdown=min(abs(b[3]-t[1]),abs(b[3]-t[3]))/th;gapup=min(abs(b[1]-t[1]),abs(b[1]-t[3]))/th
   gap=min(gapdown,gapup)
   if overlap>=.2 and gap<.35 and height<1.8*th:
    cost=gap+.15*abs(cx-(t[0]+t[2])/2)/tw
    if best is None or cost<best[0]:best=(cost,j,gapdown,gapup)
  new=b.copy()
  if mode.startswith('edge_') and best is not None:
   _,j,gd,gu=best;th=ts[j,3]-ts[j,1];delta=strength*th
   direction=mode[5:]
   if direction=='down' or (direction=='nearest' and gd<=gu):new[3]+=delta
   elif direction=='up' or direction=='nearest':new[1]-=delta
   elif direction=='both':new[1]-=delta/2;new[3]+=delta/2
  elif mode=='seat_template':
   tc=(ts[:,:2]+ts[:,2:])/2;tw=ts[:,2:]-ts[:,:2];pc=(b[:2]+b[2:])/2;j=np.argmin(np.sum(((pc-tc)/tw)**2,axis=1));pool=people[assigned==j]
   if len(pool)>=2:
    # Top-center anchors identify a seat without consulting held-day GT.
    distances=np.sqrt((((pool[:,0]+pool[:,2])/2-cx)/tw[j,0])**2+((pool[:,1]-b[1])/tw[j,1])**2)
    idx=np.argsort(distances)[:3]
    if distances[idx[0]]<.65:
     target=np.median(pool[idx],axis=0)
     # Preserve current head and horizontal location; shrinkage estimates hidden body extent.
     desired_h=target[3]-target[1];new[3]=(1-strength)*b[3]+strength*(b[1]+desired_h)
  new[[0,2]]=np.clip(new[[0,2]],0,1280);new[[1,3]]=np.clip(new[[1,3]],0,720)
  if not np.allclose(new,b):out.loc[ix,B]=new;changed.append({'reference_id':row.reference_id,'frame_id':row.frame_id,'old':b.tolist(),'new':new.tolist()})
 return out,changed
manifest=[];geometries=[]
for mode in ['edge_down','edge_up','edge_both','edge_nearest','seat_template']:
 for amount in [.25,.5,.75]:
  name=f'person_{mode}_{amount:g}';s=base.score.to_numpy().copy();aud=[]
  for fit,held in zip(DAYS,DAYS[::-1]):
   pos=np.flatnonzero(q.day.eq(held));ids=set(q.iloc[pos].frame_id);raw=refs[refs.frame_id.isin(ids)]
   transformed,changes=transform_refs(raw,models[fit],mode,amount);aud += [dict(fit_day=fit,held_day=held,**c) for c in changes]
   pred=score_refs_v4(transformed,q.iloc[pos].drop(columns='error')).set_index('id').loc[q.iloc[pos].id,'score'].to_numpy()
   comp=v.components.set_index('id').loc[q.iloc[pos].id];m=q.iloc[pos].class_name.eq('person').to_numpy()
   s[pos[m]]=np.clip(s[pos[m]]+.65*(pred[m]-comp.fusion25_iou30.to_numpy()[m]),0,1)
  register(name,s,{'mode':mode,'strength':amount,'changed_references':len(aud)})
  (A/f'{name}_reference_changes.json').write_text(json.dumps(aud))
  tr,_=transform_refs(trefs,alltables,mode,amount);pred=score_refs_v4(tr,test).set_index('id').loc[test.id,'score'].to_numpy()
  old=score_refs_v4(trefs,test).set_index('id').loc[test.id,'score'].to_numpy();st=tb.score.to_numpy().copy();m=test.class_name.eq('person').to_numpy();st[m]=np.clip(st[m]+.65*(pred[m]-old[m]),0,1)
  manifest.append(candidate(name,st));geometries.append(name)
# Input-only features can be generated for train and test identically.
def features(queries,baseline,rr,zz):
 out=[];refgroups={(a,b):g for (a,b),g in rr.groupby(['frame_id','class_name'])};zg={k:g for k,g in zz.groupby('id')}
 for i,row in queries.reset_index(drop=True).iterrows():
  b=row[B].to_numpy(float);w=max(b[2]-b[0],1);hh=max(b[3]-b[1],1);r=refgroups.get((row.frame_id,row.class_name));rec={'cx':(b[0]+b[2])/2560,'cy':(b[1]+b[3])/1440,'w':w/1280,'h':hh/720,'log_area':np.log(w*hh),'log_aspect':np.log(w/hh),'confidence':float(row.confidence),'sam_score':float(baseline.iloc[i].sam_score),'wevis_score':float(baseline.iloc[i].wevis_score)}
  for c in CL:rec['class_'+c]=int(row.class_name==c)
  rec.update(ref_count=0,best_iou=0,second_iou=0,cover_query=0,cover_ref=0,log_width_ratio=0,log_height_ratio=0,dx=0,dy=0,top_offset=0,bottom_offset=0,left_offset=0,right_offset=0,contains_count=0,contained_count=0,ref_confidence=0)
  if r is not None and len(r):
   rb=r[B].to_numpy();ov=iou(b,rb)[0];order=np.argsort(ov)[::-1];j=order[0];best=rb[j];rw=max(best[2]-best[0],1);rh=max(best[3]-best[1],1);inter=np.prod(np.maximum(0,np.minimum(b[2:],best[2:])-np.maximum(b[:2],best[:2])))
   contains=((rb[:,0]<=b[0])&(rb[:,1]<=b[1])&(rb[:,2]>=b[2])&(rb[:,3]>=b[3])).sum();contained=((rb[:,0]>=b[0])&(rb[:,1]>=b[1])&(rb[:,2]<=b[2])&(rb[:,3]<=b[3])).sum()
   rec.update(ref_count=len(r),best_iou=ov[j],second_iou=ov[order[1]] if len(order)>1 else 0,cover_query=inter/(w*hh),cover_ref=inter/(rw*rh),log_width_ratio=np.log(w/rw),log_height_ratio=np.log(hh/rh),dx=((b[0]+b[2])-(best[0]+best[2]))/(2*rw),dy=((b[1]+b[3])-(best[1]+best[3]))/(2*rh),top_offset=(b[1]-best[1])/rh,bottom_offset=(b[3]-best[3])/rh,left_offset=(b[0]-best[0])/rw,right_offset=(b[2]-best[2])/rw,contains_count=contains,contained_count=contained,ref_confidence=float(r.iloc[j].confidence))
  zr=zg.get(row.id);rec['zoom_iou']=float(iou(b,zr[B]).max()) if zr is not None and len(zr) else 0;rec['zoom_count']=len(zr) if zr is not None else 0
  rec['no_reference']=int(rec['best_iou']==0);rec['sam_wevis_difference']=rec['sam_score']-rec['wevis_score'];out.append(rec)
 return pd.DataFrame(out).fillna(0)
X=features(q,base,refs,zoom);XT=features(test,tb,trefs,tzoom);assert X.columns.tolist()==XT.columns.tolist();assert np.isfinite(X).all().all()
pd.concat([q[['id']],X],axis=1).to_csv(A/'raw_frozen_train_features.csv',index=False);pd.concat([test[['id']],XT],axis=1).to_csv(A/'raw_frozen_test_features.csv',index=False)
constructors={
 'logreg_c005':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=3000)),
 'logreg_c02':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.2,max_iter=3000)),
 'catboost_d2':lambda:CatBoostClassifier(iterations=250,depth=2,learning_rate=.03,l2_leaf_reg=10,random_seed=17,verbose=False,allow_writing_files=False,thread_count=2),
 'catboost_d4':lambda:CatBoostClassifier(iterations=250,depth=4,learning_rate=.03,l2_leaf_reg=10,random_seed=17,verbose=False,allow_writing_files=False,thread_count=2),
 'lightgbm_l4':lambda:LGBMClassifier(n_estimators=150,num_leaves=4,max_depth=2,learning_rate=.03,min_child_samples=10,reg_lambda=10,random_state=17,verbosity=-1,n_jobs=2),
 'lightgbm_l7':lambda:LGBMClassifier(n_estimators=150,num_leaves=7,max_depth=3,learning_rate=.03,min_child_samples=10,reg_lambda=10,random_state=17,verbosity=-1,n_jobs=2),
 'extratrees':lambda:ExtraTreesClassifier(n_estimators=300,max_depth=5,min_samples_leaf=5,max_features=.8,random_state=17,n_jobs=2)}
for name,ctor in constructors.items():
 try:
  scores=np.zeros(len(q));testpred=[]
  for fit,held in zip(DAYS,DAYS[::-1]):
   fi=q.day.eq(fit).to_numpy();ho=q.day.eq(held).to_numpy();assert set(q.loc[fi,'frame_id']).isdisjoint(set(q.loc[ho,'frame_id']))
   model=ctor();model.fit(X.loc[fi],q.loc[fi,'error']);scores[ho]=model.predict_proba(X.loc[ho])[:,1];testpred.append(model.predict_proba(XT)[:,1]);joblib.dump(model,A/'models'/f'{name}_fit_{fit}.joblib')
   if hasattr(model,'feature_importances_'):pd.DataFrame({'feature':X.columns,'importance':model.feature_importances_}).to_csv(A/f'{name}_fit_{fit}_importance.csv',index=False)
  for weight in [.25,.5,1.0]:
   variant=f'{name}_blend{weight:g}';register(variant,(1-weight)*base.score.to_numpy()+weight*scores,{'model':name,'weight':weight,'features':'raw frozen geometry, no cross-fitted supervised base scores'})
   manifest.append(candidate(variant,(1-weight)*tb.score.to_numpy()+weight*np.mean(testpred,axis=0)))
 except Exception as exc:failures.append({'experiment':name,'error':repr(exc),'traceback':traceback.format_exc()});print('EXPERIMENT_FAILED',name,repr(exc),flush=True)
# A first supervised frozen-DINO probe; spatial synthetic-box head follows separately.
source=v.source.copy();labels=source.id.map(q.set_index('id').error);labels[source.kind.eq('gt')]=0;assert labels.notna().all()
qfeat=v.query_features(q[['id','frame_id','class_name','day']]);person=q.class_name.eq('person').to_numpy()
for c in [.01,.1,1.]:
 scores=base.score.to_numpy().copy();raw=np.zeros(len(q))
 for fit,held in zip(DAYS,DAYS[::-1]):
  ix=source.day.eq(fit)&source.class_name.eq('person');oh=q.day.eq(held)&q.class_name.eq('person')
  model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000));model.fit(v.features[ix],labels[ix]);raw[oh]=model.predict_proba(qfeat[oh])[:,1];joblib.dump(model,A/'models'/f'dino_person_probe_c{c}_fit_{fit}.joblib')
 for w in [.1,.25,.5]:
  s=base.score.to_numpy().copy();s[person]=(1-w)*s[person]+w*raw[person];register(f'dino_person_probe_c{c:g}_w{w:g}',s,{'C':c,'weight':w,'backbone':'frozen existing DINOv3','labels':'fit-day input errors and good GT only'})
(A/'candidate_manifest.json').write_text(json.dumps(manifest,indent=2));(A/'failures.json').write_text(json.dumps(failures,indent=2))
metrics=pd.DataFrame(records).sort_values('pooled_ap',ascending=False);print('STAGE1_TOP',metrics.drop(columns='details').head(15).to_json(orient='records'),flush=True)
lines=['# Rascar V14 stage 1 experiments','', 'All models evaluated 25→26 and 26→25. Public LB baseline 0.90481; new public scores unknown. All trials retained, including lower validation scores.','', '| Experiment | Pooled AP | Day 25 | Day 26 | Person | Cup | Hands | Tray |','|---|---:|---:|---:|---:|---:|---:|---:|']
for r in metrics.to_dict('records'):lines.append('| '+r['experiment']+' | '+' | '.join(f'{r[c]:.6f}' for c in ['pooled_ap','day_25','day_26','ap_person','ap_cup','ap_nude_hands','ap_tray'])+' |')
lines += ['', 'Tables are context only. Original query boxes are never changed. Table/reference corrections use training-day table geometry and human person templates. Candidate manifests list every generated test CSV; no competition submission has been made.','', 'Model-selection caveat: these are two development days and the historical baseline used both during development. Selecting the largest result among many experiments is optimistic.']
(A/'REPORT.md').write_text('\n'.join(lines));print('STAGE1_READY',len(records),'experiments',len(manifest),'test candidates',flush=True)
# Print source required for the next image-feature stage, not competition data.
nb=json.loads((O/'baseline.ipynb').read_text());print('BASELINE_SOURCE', '\n'.join(''.join(c.get('source',[])) for c in nb['cells'])[:16000],flush=True)
for path in [R/'research_2.py',R/'research_10.py']:
 print('RESEARCH_SOURCE',path.name,path.read_text()[:15000],flush=True)
