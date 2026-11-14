"""Official query labels, independent synthetic geometry, and presence-only pool.
Test images have no targets; no historical subjective test labels are loaded.
"""
from common import *
from scipy.optimize import linear_sum_assignment
from PIL import Image
import hashlib
q,te,h=inputs();rng=np.random.default_rng(4101);rows=[]
# Reproduce official maximum-cardinality, then max-IoU matching.
check=[]
for (fid,c),g in q.groupby(['frame_id','class_name']):
 gt=h[h.frame_id.eq(fid)&h.class_name.eq(c)][B].to_numpy();ov=iou(g[B],gt);err=np.ones(len(g),int)
 if ov.size:
  r,s=linear_sum_assignment(-((ov>=.6)*(max(ov.shape)+1)+ov*(ov>=.6)))
  err[r[ov[r,s]>=.6]]=0
 assert np.array_equal(err,g.error.to_numpy()),(fid,c)
 check.extend(g.id)
def add(id,fid,c,b,split,error=np.nan,quality=np.nan,presence=np.nan,weight=1.):
 b=np.asarray(b,float);b=np.clip(b,[0,0,0,0],[1280,720,1280,720]);wh=np.maximum(b[2:]-b[:2],4);cc=(b[:2]+b[2:])/2
 lo=np.maximum(0,np.floor(cc-wh*.95));hi=np.minimum([1280,720],np.ceil(cc+wh*.95))
 rows.append(dict(id=id,frame_id=fid,day=day(fid),class_name=c,split=split,error=error,quality=quality,presence=presence,weight=weight,**dict(zip(B,b)),**dict(zip(CB,np.r_[lo,hi]))))
def targets(fid,c,b):
 gt=h[h.frame_id.eq(fid)&h.class_name.eq(c)][B].to_numpy();ov=float(iou(b,gt).max()) if len(gt) else 0.
 return float(ov<.6),ov,float(ov>=.15)
for split,d in [('query',q),('test',te)]:
 for _,r in d.iterrows():
  t=targets(r.frame_id,r.class_name,r[B]) if split=='query' else [np.nan]*3
  add(r.id,r.frame_id,r.class_name,r[B],split,r.error if split=='query' else np.nan,t[1],np.nan,1.)
for j,r in h[h.class_name.isin(CL)].iterrows():
 b=r[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2
 variants=[b,np.r_[b[:2],b[2],b[1]+wh[1]*.55],np.r_[b[:2],b[2],b[1]+wh[1]*.8]]
 for s in [.8,1.3]:
  shift=rng.uniform(-.3,.3,2)*wh;variants.append(np.r_[cc+shift-wh*s/2,cc+shift+wh*s/2])
 for k,v in enumerate(variants):
  v=np.clip(v,[0,0,0,0],[1280,720,1280,720]);add(f'synth_{j}_{k}',r.frame_id,r.class_name,v,'synthetic',*targets(r.frame_id,r.class_name,v),weight=.6)
 # Wrong semantic class on the same object; labels come from full manual frame.
 c=CL[(CL.index(r.class_name)+1+(j%2))%4]
 add(f'wrong_{j}',r.frame_id,c,b,'synthetic',*targets(r.frame_id,c,b),weight=.35)
for j,fid in enumerate(h.frame_id.unique()):
 for k,c in enumerate(CL):
  wh=np.array([70,90]) if c!='person' else np.array([180,270]);cc=rng.uniform(wh/2,np.array([1280,720])-wh/2);b=np.r_[cc-wh/2,cc+wh/2]
  add(f'bg_{j}_{k}',fid,c,b,'synthetic',*targets(fid,c,b),weight=.35)
pool=pd.read_csv(ROOT/'rascar_v33/inputs/pseudo_records.csv')
for _,r in pool.iterrows():add(r.id,r.frame_id,r.class_name,r[B],'pool_presence',presence=1.,weight=.25)
hold=pd.read_csv(ROOT/'rascar_v37/inputs/blind_pool.csv')
assert not set(hold.frame_id)&set(pool.frame_id)
for _,r in hold.iterrows():add('hold_'+r.id,r.frame_id,r.class_name,r[B],'pool_holdout')
rec=pd.DataFrame(rows);assert rec.id.is_unique
rec.to_csv(A/'inputs/records.csv',index=False)
q.to_csv(A/'inputs/queries.csv',index=False);te.to_csv(A/'inputs/test.csv',index=False);h.to_csv(A/'inputs/human.csv',index=False)
manifest=dict(records=len(rec),split_counts=rec.split.value_counts().to_dict(),official_queries_reproduced=len(check),test_targets_null=bool(rec.loc[rec.split.eq('test'),['error','quality','presence']].isna().all().all()),pool_geometry_supervised=False,fit_days=DAYS,synthetic_note='Independent single-query max-IoU targets, not extra official competition labels; original queries use official one-to-one labels.',fine_tune='last transformer block + final normalization + query-conditioned head',sources=json.loads((A/'model_cache/sources.json').read_text()))
(A/'inputs/protocol.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
