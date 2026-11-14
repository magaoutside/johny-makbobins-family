from common import *
from scipy.special import expit
from scipy.optimize import linear_sum_assignment
sem=pd.read_csv(A/'semantic_oof.csv').set_index('id').loc[q.id].reset_index();semt=pd.read_csv(A/'semantic_test.csv').set_index('id').loc[te.id].reset_index()
local=pd.read_csv(A/'local_references.csv')
def semantic_signal(rows,features):
 p=np.array([features.iloc[i]['p_'+c] for i,c in enumerate(rows.class_name)])
 # Only increases error score; correct class alone does not establish correct geometry.
 competing=np.array([features.iloc[i][['p_'+x for x in CL+['background'] if x!=c]].max() for i,c in enumerate(rows.class_name)])
 return np.clip((competing-p-.1)/.7,0,1),p
sig,p=semantic_signal(q,sem);sigt,pt=semantic_signal(te,semt)
for scope in ['person','all']:
 for w in [.15,.3,.5]:
  m=q.class_name.eq('person').to_numpy() if scope=='person' else np.ones(len(q),bool);mt=te.class_name.eq('person').to_numpy() if scope=='person' else np.ones(len(te),bool)
  s=q.score.to_numpy().copy();t=current.copy();s[m]+=w*(1-s[m])*sig[m];t[mt]+=w*(1-t[mt])*sigt[mt]
  register(f'semantic_{scope}_{w:g}',s,t,dict(weight=w,scope=scope,semantic_only_increases_error=True))
for w in [.1,.25]:
 s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('person');mt=te.class_name.eq('person')
 s[m]=(1-w)*s[m]+w*sem.loc[m,'quality_person'];t[mt]=(1-w)*t[mt]+w*semt.loc[mt,'quality_person']
 register(f'person_quality_{w:g}',s,t)
def local_evidence(rows,held,threshold=.25):
 output=[];groups=[]
 for (fid,c),g in rows.groupby(['frame_id','class_name']):
  rr=local[(local.frame_id==fid)&(local.class_name==c)&(local.confidence>=threshold)&(local.held_day==held)]
  if rr.empty:
   for i in g.index:output.append(dict(idx=i,local_iou=0,local_score=1,local_confirm=False,reference_id='',local_duplicate=False,local_confidence=0,scale_support=0))
   continue
  # Merge repeated observations across crops, then match all queries jointly.
  rr=rr.sort_values('confidence',ascending=False).reset_index(drop=True);keep=[];support=[]
  while len(rr):
   row=rr.iloc[0];ov=iou(row[B],rr[B])[0];members=rr[ov>=.5];b=np.average(members[B],axis=0,weights=members.confidence)
   keep.append(dict(**dict(zip(B,b)),confidence=float(members.confidence.max()),reference_id=row.reference_id,scale_support=members.scale.nunique()));rr=rr[ov<.5].reset_index(drop=True)
  refs=pd.DataFrame(keep);ov=iou(g[B],refs[B]);reward=np.c_[np.where(ov>=.45,min(ov.shape)+1+ov,0),np.zeros((len(g),len(g)))];ii,jj=linear_sum_assignment(-reward);match={i:j for i,j in zip(ii,jj) if j<len(refs) and ov[i,j]>=.45}
  for j,ix in enumerate(g.index):
   best=int(ov[j].argmax());chosen=match.get(j,best);score=expit((.6-ov[j,chosen])/.08);duplicate=j not in match and ov[j,best]>=.6
   if duplicate:score=1.
   ref=refs.iloc[chosen];valid=j in match and ov[j,chosen]>=.6 and ref.confidence>=.35 and ref.scale_support>=2
   output.append(dict(idx=ix,local_iou=float(ov[j,chosen]),local_score=float(score),local_confirm=bool(valid),reference_id=ref.reference_id,local_duplicate=bool(duplicate),local_confidence=ref.confidence,scale_support=ref.scale_support))
   groups.append(dict(id=rows.loc[ix,'id'],frame_id=fid,class_name=c,assigned=j in match,duplicate=duplicate,reference_id=ref.reference_id,**dict(zip(['ref_'+k for k in B],ref[B]))))
 return pd.DataFrame(output).set_index('idx').reindex(rows.index),pd.DataFrame(groups)
parts=[];group_parts=[]
for day in DAYS:
 sub=q[q.day.eq(day)];ev,gg=local_evidence(sub,day);parts.append(ev);group_parts.append(gg)
ev=pd.concat(parts).reindex(q.index);evt,gt=local_evidence(te,'test')
q[['id']].join(ev).to_csv(A/'local_oof_evidence.csv',index=False);te[['id']].join(evt).to_csv(A/'local_test_evidence.csv',index=False)
pd.concat(group_parts).to_csv(A/'local_oof_assignments.csv',index=False);gt.to_csv(A/'local_test_assignments.csv',index=False)
for scope in ['hands','hands_cup']:
 for w in [.15,.3,.5]:
  s=q.score.to_numpy().copy();t=current.copy();allowed=['nude_hands'] if scope=='hands' else ['nude_hands','cup']
  m=q.class_name.isin(allowed)&ev.local_confirm&(p>=.45);mt=te.class_name.isin(allowed)&evt.local_confirm&(pt>=.45)
  # Rescue only a confident geometrically complete match, never just object presence.
  s[m]=(1-w)*s[m]+w*np.minimum(s[m],ev.loc[m,'local_score']);t[mt]=(1-w)*t[mt]+w*np.minimum(t[mt],evt.loc[mt,'local_score'])
  register(f'local_{scope}_{w:g}',s,t,dict(confirmed_train=int(m.sum()),confirmed_test=int(mt.sum()),weight=w))
# Position-conditioned full-body hypotheses use only the opposite day's manual people.
def person_extent(rows,semantics,fit_day=None):
 result=[]
 for r in rows.itertuples():
  if r.class_name!='person':result.append(0.);continue
  pool=h[h.class_name.eq('person')]
  if fit_day:pool=pool[pool.day.eq(fit_day)]
  b=np.array([getattr(r,c) for c in B]);bs=pool[B].to_numpy();w=b[2]-b[0];height=b[3]-b[1]
  dist=np.sqrt(((bs[:,0]+bs[:,2]-b[0]-b[2])/2/1280)**2+((bs[:,1]-b[1])/720)**2)
  near=np.argsort(dist)[:3];heights=bs[near,3]-bs[near,1]
  # Uncertain locations and inconsistent posture priors abstain.
  if dist[near[0]]>.055 or dist[near[-1]]>.12 or np.std(heights)/np.mean(heights)>.3:result.append(0.);continue
  target=float(np.median(heights));ratio=height/target
  semantic=float(semantics.loc[r.id,'p_person'])
  risk=np.clip((.75-ratio)/.35,0,1)*np.clip((semantic-.45)/.4,0,1)
  result.append(float(risk))
 return np.array(result)
extent=np.zeros(len(q))
for fit,held in zip(DAYS,DAYS[::-1]):
 mask=q.day.eq(held);extent[mask]=person_extent(q[mask],sem.set_index('id'),fit)
extentt=person_extent(te,semt.set_index('id'))
pd.DataFrame({'id':q.id,'extent_risk':extent}).to_csv(A/'extent_oof.csv',index=False);pd.DataFrame({'id':te.id,'extent_risk':extentt}).to_csv(A/'extent_test.csv',index=False)
for w in [.25,.5,.75]:register(f'person_extent_{w:g}',q.score+w*(1-q.score)*extent,current+w*(1-current)*extentt,dict(weight=w))
