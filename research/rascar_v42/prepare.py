"""Direct binary error targets. Never assigns class-presence as bbox correctness.
Official labels for real queries; auxiliary single-box labels for synthetic boxes.
Weak pool geometry is kept well away from the IoU=.6 boundary.
"""
from base42 import *
import cv2
q,te,h=inputs();rng=np.random.default_rng(4201);rows=[]
def add(id,fid,c,b,source,target=np.nan,weight=1.,origin='',confidence=np.nan):
 b=np.clip(np.asarray(b,float),[0,0,0,0],[1280,720,1280,720])
 if min(b[2:]-b[:2])<3:return
 wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;span=np.maximum(wh*2,64)
 lo=np.maximum(0,np.floor(cc-span/2));hi=np.minimum([1280,720],np.ceil(cc+span/2))
 rows.append(dict(id=id,frame_id=fid,day=pd.to_datetime(fid.split('_')[-2]).strftime('%Y-%m-%d'),class_name=c,source=source,target=target,weight=weight,origin_id=origin,confidence=confidence,**dict(zip(B,b)),**dict(zip(CB,np.r_[lo,hi]))))
for source,df in [('query',q),('test',te)]:
 for _,r in df.iterrows():add(r.id,r.frame_id,r.class_name,r[B],source,r.error if source=='query' else np.nan,confidence=r.confidence)
def variants(b):
 wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2
 out=[b]
 for scale in [.6,.82,1.18,1.55]:out.append(np.r_[cc-wh*scale/2,cc+wh*scale/2])
 for shift in [[.35,0],[-.35,0],[0,.35],[0,-.35]]:out.append(b+np.tile(wh*np.array(shift),2))
 out.extend([np.r_[b[:2],b[2],b[1]+wh[1]*.45],np.r_[b[:2],b[2],b[1]+wh[1]*.75],np.r_[b[0],b[1]+wh[1]*.4,b[2:]]])
 return out
for j,r in h[h.class_name.isin(CL)].iterrows():
 b=r[B].to_numpy(float);gt=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(r.class_name)][B].to_numpy()
 for k,v in enumerate(variants(b)):
  v=np.clip(v,[0,0,0,0],[1280,720,1280,720]);ov=float(iou(v,gt).max());add(f'human_{j}_{k}',r.frame_id,r.class_name,v,'synthetic',float(ov<.6),.45,origin=f'human_{j}')
 # Wrong-class targets can include background/nearby objects: consult all GT.
 c=CL[(CL.index(r.class_name)+1+(j%2))%4];g=h[h.frame_id.eq(r.frame_id)&h.class_name.eq(c)][B].to_numpy();ov=float(iou(b,g).max()) if len(g) else 0
 add(f'wrong_{j}',r.frame_id,c,b,'synthetic',float(ov<.6),.3,origin=f'human_{j}')
for j,fid in enumerate(h.frame_id.unique()):
 for c in CL:
  wh=np.array([160,240]) if c=='person' else np.array([55,65]);cc=rng.uniform(wh/2,[1280,720]-wh/2);b=np.r_[cc-wh/2,cc+wh/2];gt=h[h.frame_id.eq(fid)&h.class_name.eq(c)][B].to_numpy();ov=float(iou(b,gt).max()) if len(gt) else 0.
  add(f'bg_{j}_{c}',fid,c,b,'synthetic',float(ov<.6),.25,origin=f'background_{j}')
old=pd.read_csv(ROOT/'rascar_v33/inputs/pseudo_records.csv');old=old[old.label_use.eq('geometry_and_presence')].copy();old['pool_source']='old'
new=pd.read_csv(ROOT/'rascar_v41/pool_expansion/approved_geometry.csv');new['pool_source']='new'
pool=pd.concat([old,new[new.decision.eq('weak_geometry')]],ignore_index=True).drop_duplicates(['frame_id','class_name']+B)
held_frames=set(pd.read_csv(ROOT/'rascar_v37/inputs/blind_pool.csv').frame_id)
pool=pool[~pool.frame_id.isin(held_frames)]
for _,r in pool.iterrows():
 assert r.frame_id not in set(te.frame_id)|set(h.frame_id)
 b=r[B].to_numpy(float);wh=b[2:]-b[:2];uncertainty=np.tile(np.maximum(2,wh*.06),2)
 # Samples of plausible reference boundaries, used for conservative weak labels.
 refs=np.r_[b[None],b[None]+rng.uniform(-1,1,(32,4))*uncertainty]
 for k,v in enumerate(variants(b)):
  v=np.clip(v,[0,0,0,0],[1280,720,1280,720]);ovs=iou(v,refs)[0]
  if min(ovs)>.72:label=0.
  elif max(ovs)<.42:label=1.
  else:continue
  add(f'pool_{r.id}_{k}',r.frame_id,r.class_name,v,'pool',label,.15,origin=str(r.id))
neg=pd.read_csv(ROOT/'rascar_v37/inputs/promoted_pool_presence.csv');neg=neg[neg.semantic_target.eq(0)&~neg.frame_id.isin(held_frames)]
neg=pd.concat([neg,new[new.decision.eq('class_absent')]],ignore_index=True)
for _,r in neg.iterrows():
 assert r.frame_id not in set(te.frame_id)|set(h.frame_id)
 add(f'negative_{r.id}',r.frame_id,r.class_name,r[B],'pool',1.,.3,origin=str(r.id))
rec=pd.DataFrame(rows);assert rec.id.is_unique;assert rec[rec.source.eq('test')].target.isna().all();rec.to_csv(A/'inputs/records.csv',index=False)
# Raw detector neighbours are allowed inputs; no GT is used in these features.
stream=pd.concat([pd.read_csv(D/'train_stream_predictions.csv'),pd.read_csv(D/'test_stream_predictions.csv'),pd.read_csv(D/'machine_predictions.csv')],ignore_index=True)
stream=stream[stream.class_name.isin(CL)].drop_duplicates('detection_id');groups={(f,c):g for (f,c),g in stream.groupby(['frame_id','class_name'])};meta=[]
for _,r in rec.iterrows():
 b=r[B].to_numpy(float);wh=b[2:]-b[:2];cc=(b[:2]+b[2:])/2;cb=r[CB].to_numpy(float);rel=(b-np.tile(cb[:2],2))/np.tile(cb[2:]-cb[:2],2)
 g=groups.get((r.frame_id,r.class_name));nn=np.zeros(5)
 if g is not None:
  g=g[~g.detection_id.eq(r.id)&~g.detection_id.eq(str(r.origin_id).removeprefix('pseudo33_'))];ov=iou(b,g[B])[0]
  if len(ov):
   best=int(np.argmax(ov));other=g.iloc[best];nn=np.array([max(ov),sum(ov>.5)/5,float(other.confidence),np.log(max((other.x2-other.x1)*(other.y2-other.y1),1)/wh.prod()),float(other.confidence>r.confidence) if pd.notna(r.confidence) else 0])
 # Detector confidence is intentionally not used in the neural visual head.
 meta.append(np.r_[np.eye(4)[CL.index(r.class_name)],wh/[1280,720],cc/[1280,720],np.log(wh[0]/wh[1]),rel,nn])
np.save(A/'inputs/metadata.npy',np.asarray(meta,np.float32))
cv2.setNumThreads(2);cache={};images=np.lib.format.open_memmap(A/'features/crops_uint8.npy',mode='w+',dtype=np.uint8,shape=(len(rec),2,3,224,224))
for i,r in rec.iterrows():
 if r.frame_id not in cache:cache[r.frame_id]=cv2.resize(cv2.imread(str(D/f'{r.frame_id}.jpg')),(1280,720))
 for view,cols in enumerate([B,CB]):
  b=r[cols].to_numpy(float);lo=np.floor(b[:2]).astype(int);hi=np.ceil(b[2:]).astype(int);im=cache[r.frame_id][lo[1]:hi[1],lo[0]:hi[0],::-1]
  images[i,view]=cv2.resize(im,(224,224),interpolation=cv2.INTER_CUBIC).transpose(2,0,1)
 if i%800==0:print('CROPS',i,len(rec),flush=True)
images.flush()
manifest=dict(records=len(rec),sources=rec.source.value_counts().to_dict(),pool_original_boxes=len(pool),pool_negative_boxes=len(neg),pool_frames=int(rec.loc[rec.source.eq('pool'),'frame_id'].nunique()),test_targets_null=True,official_labels='error from train.csv',synthetic_labels='max same-class GT IoU < .6, single-query targets',weak_pool_labels='robust positive IoU>.72 or negative IoU<.42 under perturbed weak reference; reduced loss weight',manual_test_labels_used=False,metadata_dim=np.asarray(meta).shape[1],model_input='tight RGB crop + context RGB crop + bbox mask + class/geometry/raw-detector-neighbours; no V31 score or model reference IoUs')
(A/'inputs/protocol.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
