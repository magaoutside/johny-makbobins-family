from common import *
from scipy.optimize import linear_sum_assignment
def grouped_duplicates(rows,refs,base,semantics=None):
 scores=np.asarray(base).copy();ledger=[]
 for (fid,c),g in rows.groupby(['frame_id','class_name']):
  if len(g)<2:continue
  boxes=g[B].to_numpy();ov=iou(boxes,boxes);area=np.prod(boxes[:,2:]-boxes[:,:2],1);inter=np.maximum(0,np.minimum(boxes[:,None,2:],boxes[None,:,2:])-np.maximum(boxes[:,None,:2],boxes[None,:,:2])).prod(2)
  contain=inter/np.maximum(np.minimum(area[:,None],area[None]),1e-9);edges=(ov>=.5)|((contain>=.85)&(np.maximum(area[:,None],area[None])/np.minimum(area[:,None],area[None])<=3))
  rr=refs[(refs.frame_id==fid)&(refs.class_name==c)].copy()
  if 'confidence' in rr:rr=rr[rr.confidence>=.25].sort_values('confidence',ascending=False)
  # Deduplicate repeated observations of the same physical reference.
  keep=[]
  for _,r in rr.iterrows():
   if not keep or iou(r[B],np.array(keep)).max()<.6:keep.append(r[B].to_numpy(float))
  if not keep:continue
  rb=np.array(keep);match_iou=iou(boxes,rb)
  visited=set()
  for start in range(len(g)):
   if start in visited:continue
   stack=[start];group=[]
   while stack:
    k=stack.pop()
    if k in visited:continue
    visited.add(k);group.append(k);stack.extend(j for j in range(len(g)) if edges[k,j] and j not in visited)
   if len(group)<2:continue
   z=match_iou[group];reward=np.c_[np.where(z>=.45,len(group)+1+z,0),np.zeros((len(group),len(group)))];ii,jj=linear_sum_assignment(-reward);matches={i:j for i,j in zip(ii,jj) if j<len(rb) and z[i,j]>=.45}
   for li,gi in enumerate(group):
    ref=int(z[li].argmax());best=float(z[li,ref]);winner=next((wi for wi,wj in matches.items() if wj==ref),None);reason='distinct_or_uncertain';chosen=None
    if li not in matches and winner is not None and best>=.3:
     wi=group[winner];margin=float(z[winner,ref]-best)
     if z[winner,ref]>=.6 and margin>=.08 and (ov[gi,wi]>=.5 or contain[gi,wi]>=.9):
      reason='duplicate_loser';scores[g.index[gi]]=max(scores[g.index[gi]],.95);chosen=rows.loc[g.index[wi],'id']
    ledger.append(dict(id=rows.loc[g.index[gi],'id'],frame_id=fid,class_name=c,group_ids='|'.join(g.iloc[group].id),best_iou=best,assigned=li in matches,reason=reason,winner_id=chosen,score_before=float(base[g.index[gi]]),score_after=float(scores[g.index[gi]])))
 return scores,pd.DataFrame(ledger)
if __name__=='__main__':
 refs=pd.read_csv(A/'v8/fused_train_references.csv');reft=pd.read_csv(A/'v8/fused_test_references.csv')
 s,ledger=grouped_duplicates(q,refs,q.score.to_numpy());t,ledgert=grouped_duplicates(te,reft,current)
 ledger.to_csv(A/'duplicate_groups_oof.csv',index=False);ledgert.to_csv(A/'duplicate_groups_test.csv',index=False)
 register('duplicate_groups',s,t,dict(rule='shared reference, unique assignment; IoU winner>=.6; margin>=.08',train_losers=int(ledger.reason.eq('duplicate_loser').sum()) if len(ledger) else 0,test_losers=int(ledgert.reason.eq('duplicate_loser').sum()) if len(ledgert) else 0))
