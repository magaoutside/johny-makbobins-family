from common import *
import evaluate_components as ec
from duplicate_groups import grouped_duplicates
from scipy.optimize import linear_sum_assignment
def protect_two_hands(rows,ledger,refs,scores):
 if ledger.empty:return scores,ledger
 lookup=rows.set_index('id');scores=np.asarray(scores).copy()
 for group,g in ledger[ledger.class_name.eq('nude_hands')].groupby('group_ids'):
  ids=group.split('|');qb=lookup.loc[ids,B].to_numpy();fid=g.frame_id.iloc[0];rr=refs[(refs.frame_id==fid)&refs.class_name.eq('nude_hands')&(refs.confidence>=.35)]
  views=[]
  for (_,scale),view in rr.groupby(['query_id','scale']):
   b=view[B].to_numpy();b=b[iou(b,qb).max(1)>=.3]
   if len(b)>=2:views.append((scale,b))
  confirmed=False
  for scale,b in views:
   for scale2,c in views:
    if scale==scale2:continue
    ov=iou(b,c);ii,jj=linear_sum_assignment(-ov);good=ii[ov[ii,jj]>=.45]
    for x in good:
     for y in good:
      if x>=y:continue
      centers=(b[:,:2]+b[:,2:])/2
      if np.linalg.norm(centers[x]-centers[y])>=8:confirmed=True
  if confirmed:
   for ix,r in g.iterrows():
    if r.reason=='duplicate_loser':
     scores[rows.index[rows.id.eq(r.id)][0]]=r.score_before;ledger.loc[ix,'reason']='two_hands_supported_abstain';ledger.loc[ix,'score_after']=r.score_before
 return scores,ledger
local=pd.concat([pd.read_csv(A/'hand_specialist'/fit/'references.csv') for fit in DAYS+['all']],ignore_index=True)
local.to_csv(A/'hand_specialist_references.csv',index=False);ec.local=local
sem=pd.read_csv(A/'semantic_oof.csv').set_index('id').loc[q.id].reset_index();semt=pd.read_csv(A/'semantic_test.csv').set_index('id').loc[te.id].reset_index()
for threshold in [.15,.35]:
 parts=[];assign=[]
 for day in DAYS:
  ev,gg=ec.local_evidence(q[q.day.eq(day)],day,threshold);parts.append(ev);assign.append(gg)
 ev=pd.concat(parts).reindex(q.index);evt,gt=ec.local_evidence(te,'test',threshold)
 q[['id']].join(ev).to_csv(A/f'hand_specialist_oof_evidence_{threshold}.csv',index=False);te[['id']].join(evt).to_csv(A/f'hand_specialist_test_evidence_{threshold}.csv',index=False)
 pd.concat(assign).to_csv(A/f'hand_specialist_oof_assignments_{threshold}.csv',index=False);gt.to_csv(A/f'hand_specialist_test_assignments_{threshold}.csv',index=False)
 for w in [.2,.4,.6]:
  s=q.score.to_numpy().copy();t=current.copy();m=q.class_name.eq('nude_hands')&ev.local_confirm&(sem.p_nude_hands>=.45);mt=te.class_name.eq('nude_hands')&evt.local_confirm&(semt.p_nude_hands>=.45)
  s[m]=(1-w)*s[m]+w*np.minimum(s[m],ev.loc[m,'local_score']);t[mt]=(1-w)*t[mt]+w*np.minimum(t[mt],evt.loc[mt,'local_score'])
  register(f'hand_specialist_c{threshold}_w{w}',s,t,dict(confirmed_train=int(m.sum()),confirmed_test=int(mt.sum()),weight=w,confidence=threshold))
# Re-evaluate ambiguous hand groups with the specialist's independent object count.
ref=pd.read_csv(A/'v8/fused_train_references.csv');reft=pd.read_csv(A/'v8/fused_test_references.csv');parts=[]
for fit,held in zip(DAYS,DAYS[::-1]):
 mask=q.day.eq(held);refs=pd.concat([ref,local[(local.held_day==held)&(local.confidence>=.35)]],ignore_index=True)
 sub=q[mask].reset_index(drop=True);score,ledger=grouped_duplicates(sub,refs,q.loc[mask,'score'].to_numpy());score,ledger=protect_two_hands(sub,ledger,local[local.held_day.eq(held)],score);parts.append(pd.DataFrame({'id':q.loc[mask,'id'],'score':score}));ledger.to_csv(A/f'hand_group_oof_{held}.csv',index=False)
st,lt=grouped_duplicates(te,pd.concat([reft,local[(local.held_day=='test')&(local.confidence>=.35)]],ignore_index=True),current);st,lt=protect_two_hands(te,lt,local[local.held_day.eq('test')],st)
lt.to_csv(A/'hand_group_test.csv',index=False);s=pd.concat(parts).set_index('id').loc[q.id,'score'].to_numpy();register('duplicate_groups_with_specialist',s,st)
