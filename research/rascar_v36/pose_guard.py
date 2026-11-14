"""Do not penalize a person using a reference which cuts off supported legs.
Use existing RTMW body predictions in both 0/180-degree views, matched by head.
This is a veto of new penalties, not an amodal box generator or pose truth.
"""
from common36 import *

def main():
 poses=pd.read_csv(ROOT/'rascar_v25/gpu/results/rtmw/keypoints.csv').drop_duplicates(['frame_id','turn','person']);heads=pd.read_csv(ROOT/'rascar_v16/wevis_heads_references.csv');out=[];results=[]
 def legs(row):
  hb=heads[heads.frame_id.eq(row.frame_id)][B].to_numpy(float);hc=(hb[:,:2]+hb[:,2:])/2;b=row[B].to_numpy(float);inside=((hc>=b[:2])&(hc<=b[2:])).all(1)
  if inside.sum()!=1:return []
  head=hc[inside][0];diag=np.linalg.norm(hb[inside][0,2:]-hb[inside][0,:2]);selected=[]
  for turn in [0,2]:
   candidates=[]
   for _,p in poses[poses.frame_id.eq(row.frame_id)&poses.turn.eq(turn)].iterrows():
    xy=np.array(json.loads(p.body_xy));co=np.array(json.loads(p.body_conf));dist=np.linalg.norm(np.median(xy[:5],axis=0)-head)
    if dist<=.75*diag:candidates.append((dist,xy,co))
   if not candidates:return []
   _,xy,co=min(candidates,key=lambda x:x[0]);selected.append((xy,co))
  xy0,c0=selected[0];xy1,c1=selected[1];tol=max(12,.1*np.linalg.norm(b[2:]-b[:2]));keep=[]
  for chain in [[13,15],[14,16]]:
   pts=(xy0[chain]+xy1[chain])/2
   if not ((c0[chain]>=.5*c0.max()).all() and (c1[chain]>=.5*c1.max()).all()):continue
   if not (np.linalg.norm(xy0[chain]-xy1[chain],axis=1)<=tol).all():continue
   if not ((pts>=b[:2])&(pts<=b[2:])).all():continue
   if np.linalg.norm(pts[1]-pts[0])<3:continue
   keep.extend(pts.tolist())
  return keep
 for old,kind,confidence,tag in [('person_strict_pool','dense_identity',.55,'strict_pose_pool'),('person_multiscale_pool','multiscale',.4,'multiscale_pose_pool')]:
  scores={}
  for split,rows in ROWS.items():
   path=A/(f'results/oof/{old}.csv' if split=='oof' else f'results/submissions/submission_{old}.csv');s=align(path,rows).score.to_numpy();refs={fit:align(A/f'features/{kind}_vote_c{confidence}_{split}_fit{fit[-2:]}.csv',rows[rows.day.ne(fit)] if split=='oof' else rows) for fit in DAYS}
   for i,(_,r) in enumerate(rows.iterrows()):
    if r.class_name!='person' or s[i]<=BASE[split][i]+1e-10:continue
    pts=np.array(legs(r)).reshape(-1,2);veto=False
    if len(pts):
     for fit,d in refs.items():
      if r.id not in d.index:continue
      ref=d.loc[r.id,['ref_'+v for v in B]].to_numpy(float);margin=max(6,.025*(r.y2-r.y1));outside=((pts<ref[:2]-margin)|(pts>ref[2:]+margin)).any(1)
      if outside.any():veto=True
    out.append(dict(id=r.id,frame_id=r.frame_id,split=split,experiment=tag,veto=veto,supported_lower_body_points=json.dumps(pts.tolist()),before_score=s[i],v31_score=BASE[split][i]))
    if veto:s[i]=BASE[split][i]
   scores[split]=s
  results.append(save(tag,scores,'New positive person penalties vetoed if RTMW 0/180 views agree on a knee-ankle chain inside the query but outside the proposed reference. Same head association; no score thresholds tuned.'))
 pd.DataFrame(out).to_csv(A/'audit/pose_guard_decisions.csv',index=False);pd.DataFrame(results).to_csv(A/'results/pose_guard_experiments.csv',index=False)

if __name__=='__main__':main()
