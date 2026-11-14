
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
E=Path('/kaggle/working/rascar_v14');O=Path('/kaggle/working/oof_validation_v11');R=E/'v8_replay/sam31_v8';S=E/'stage4';S.mkdir(exist_ok=True);sys.path.insert(0,str(E/'yolo26_clean'))
from frozen_scoring import score_refs_v4
B=['x1','y1','x2','y2'];DAYS=['2026-10-25','2026-10-26'];q=pd.read_csv(E/'stage1/oof/v8_baseline.csv');original=pd.read_csv(O/'train.csv');q=q.merge(original[['id']+B+['confidence']],on='id',validate='one_to_one');te=pd.read_csv(O/'test.csv');bt=pd.read_csv('/kaggle/working/submission.csv').set_index('id').loc[te.id,'score'].to_numpy();fr=pd.read_csv(O/'train_frames.csv');fr['day']=pd.to_datetime(fr.timestamp,utc=True,format='mixed').dt.strftime('%Y-%m-%d')
h=pd.read_csv(O/'human_boxes_train.csv');h.class_name=h.class_name.replace(json.loads((O/'scoring_policy.json').read_text())['aliases']);h=h.merge(fr[['frame_id','day']],on='frame_id',validate='many_to_one');h[B]=np.c_[(h.cx-h.w/2)*1280,(h.cy-h.h/2)*720,(h.cx+h.w/2)*1280,(h.cy+h.h/2)*720]
refs=pd.read_csv(R/'fused_train_references.csv');trefs=pd.read_csv(R/'fused_test_references.csv');raw=score_refs_v4(refs,q).set_index('id').loc[q.id,'score'].to_numpy();rawt=score_refs_v4(trefs,te).set_index('id').loc[te.id,'score'].to_numpy()
def fit_priors(train):
 ts=train[train.class_name.eq('table')];sets=[]
 for _,g in ts.groupby('frame_id'):
  if len(g)==6:sets.append(g.sort_values('cx')[B].to_numpy())
 if len(sets)<3:return {'valid':False,'reason':'fewer_than_three_six_table_frames','six_table_frames':len(sets)}
 tables=np.median(np.stack(sets),axis=0);people=train[train.class_name.eq('person')];seats=[]
 for i,t in enumerate(tables):
  tw,th=t[2:]-t[:2];cx=(people.x1+people.x2)/2;head=people.y1;ids=np.argmin(abs(cx.to_numpy()[:,None]-(tables[:,0]+tables[:,2])[None]/2),axis=1);g=people[(ids==i)&cx.between(t[0]-.3*tw,t[2]+.3*tw)&head.between(t[1]-1.5*th,t[3]+.2*th)]
  if len(g)<6:continue
  anchors=np.c_[((g.x1+g.x2)/2-t[0])/tw,(g.y1-t[1])/th];km=KMeans(2,random_state=17,n_init=10).fit(anchors)
  for k in range(2):
   gg=g[km.labels_==k]
   if gg.frame_id.nunique()<3:continue
   seats.append(dict(table=i,seat=k,anchor=km.cluster_centers_[k].tolist(),bbox=np.median(gg[B].to_numpy(),axis=0).tolist(),support=int(gg.frame_id.nunique())))
 return dict(valid=True,tables=tables.tolist(),seats=seats,six_table_frames=len(sets))
def modify(rr,prior,strength,mode):
 rr=rr.copy();changes=[]
 if not prior['valid']:return rr,changes
 tables=np.array(prior['tables'])
 for ix,r in rr[rr.class_name.eq('person')].iterrows():
  b=r[B].to_numpy(float);cx=(b[0]+b[2])/2;idx=int(np.argmin(abs(cx-(tables[:,0]+tables[:,2])/2)));t=tables[idx];tw,th=t[2:]-t[:2];rh=b[3]-b[1]
  # Aisle/standing boxes and distant anchors are outside this prior.
  if not (t[0]-.3*tw<=cx<=t[2]+.3*tw) or rh>2.5*th:continue
  candidates=[s for s in prior['seats'] if s['table']==idx]
  if not candidates:continue
  a=np.array([(cx-t[0])/tw,(b[1]-t[1])/th]);ds=[float(np.linalg.norm(a-np.array(s['anchor']))) for s in candidates];j=int(np.argmin(ds))
  if ds[j]>.55:continue
  target=np.array(candidates[j]['bbox']);nb=b.copy()
  if mode in ['down','both'] and abs(b[3]-t[3])<.5*th:nb[3]=b[3]+strength*max(0,min(target[3]-b[3],.6*rh))
  if mode in ['up','both'] and abs(b[1]-t[1])<.5*th:nb[1]=b[1]-strength*max(0,min(b[1]-target[1],.6*rh))
  nb[[1,3]]=np.clip(nb[[1,3]],0,720)
  if np.any(nb!=b):rr.loc[ix,B]=nb;changes.append(dict(reference_index=int(ix),frame_id=r.frame_id,table=idx,seat=candidates[j]['seat'],distance=ds[j],old=b.tolist(),new=nb.tolist()))
 return rr,changes
metrics=[]
priors={fit:fit_priors(h[h.day.eq(fit)]) for fit in DAYS};priors['all']=fit_priors(h);(S/'twelve_seat_priors.json').write_text(json.dumps(priors,indent=2))
for strength in [.25,.5,1.]:
 for mode in ['down','up','both']:
  pred=q.score.to_numpy().copy();all_changes=[]
  for fit,held in zip(DAYS,DAYS[::-1]):
   m=q.day.eq(held);sub=q[m];rr,changes=modify(refs[refs.frame_id.isin(sub.frame_id)],priors[fit],strength,mode);s=score_refs_v4(rr,sub).set_index('id').loc[sub.id,'score'].to_numpy();person=sub.class_name.eq('person').to_numpy();ind=sub.index[person];pred[ind]=np.clip(pred[ind]+.65*(s[person]-raw[ind]),0,1);all_changes.extend(changes)
  name=f'twelve_seats_{mode}_{strength}';row={'experiment':name,'pooled_ap':float(average_precision_score(q.error,pred)),'modified_references':len(all_changes)}
  for day,g in q.groupby('day'):row['day_'+day[-2:]]=float(average_precision_score(g.error,pred[g.index]))
  metrics.append(row);q.assign(score=pred).to_csv(S/f'{name}_oof.csv',index=False);(S/f'{name}_changes.json').write_text(json.dumps(all_changes,indent=2));rr,changes=modify(trefs,priors['all'],strength,mode);ss=score_refs_v4(rr,te).set_index('id').loc[te.id,'score'].to_numpy();tt=bt.copy();m=te.class_name.eq('person').to_numpy();tt[m]=np.clip(tt[m]+.65*(ss[m]-rawt[m]),0,1);te[['id']].assign(score=tt).to_csv(S/f'submission_{name}.csv',index=False);print('SEAT_RESULT',json.dumps(row),flush=True)
pd.DataFrame(metrics).to_csv(S/'seat_experiment_metrics.csv',index=False)
print('SEAT_SUPPORT',json.dumps({k:{'valid':v['valid'],'supported_seats':len(v.get('seats',[])),'six_table_frames':v['six_table_frames']} for k,v in priors.items()}),flush=True)
