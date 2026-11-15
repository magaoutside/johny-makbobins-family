"""Selected geometry features, computed from independent detector outputs."""
from pathlib import Path
import numpy as np,pandas as pd
from .core import queries, aligned_csv, BBOX as B, CLASSES as CL


def iou(a,b):
 a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
 inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(2)
 return inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(1)[None]-inter,1e-9)

def reference_features(queries,baseline,rr,zz):
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

def wevis_features(rows, heads, hands, pose):
    out=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
        rec=dict(id=r.id,head_count=0.,head_near_outside=0.,head_iou=0.,head_query_cover=0.,hand_iou=0.,hand_area_ratio=0.,hand_query_coverage=0.,hand_instances=0.,head_cut_distance=0.)
        hh=heads[heads.frame_id.eq(r.frame_id)][B].to_numpy()
        if len(hh):
            center=(hh[:,:2]+hh[:,2:])/2;inside=((center>=b[:2])&(center<=b[2:])).all(1)
            rec['head_count']=float(inside.sum());rec['head_iou']=float(iou(b,hh).max())
            inter=np.maximum(0,np.minimum(hh[:,2:],b[2:])-np.maximum(hh[:,:2],b[:2])).prod(1)
            rec['head_query_cover']=float((inter/max(wh.prod(),1)).max())
            pp=pose[(pose.frame_id==r.frame_id)&(pose.confidence>=.3)]
            body_iou=float(iou(b,pp[B]).max()) if len(pp) else 0.
            # Heads must lie nearby above the box; abstain if a good full-body match exists.
            dx=np.abs(center[:,0]-(b[0]+b[2])/2)/wh[0];dy=(b[1]-center[:,1])/wh[1]
            near=(dx<.5)&(dy>.08)&(dy<.8)
            if not inside.any() and near.any() and .05<body_iou<.55:
                rec['head_near_outside']=1.;rec['head_cut_distance']=float(dy[near].min())
        rr=hands[hands.query_id.eq(r.id)][B].to_numpy()
        if len(rr):
            ov=iou(b,rr)[0];j=int(ov.argmax());rec['hand_iou']=float(ov[j])
            rec['hand_area_ratio']=float(wh.prod()/max(np.prod(rr[j,2:]-rr[j,:2]),1))
            inter=np.maximum(0,np.minimum(rr[:,2:],b[2:])-np.maximum(rr[:,:2],b[:2])).prod(1)
            rec['hand_query_coverage']=float((inter/max(wh.prod(),1)).max())
            cc=(rr[:,:2]+rr[:,2:])/2;rec['hand_instances']=float(((cc>=b[:2])&(cc<=b[2:])).all(1).sum())
        out.append(rec)
    return pd.DataFrame(out)

def pose_features(rows, poses):
    result=[]
    for r in rows.itertuples():
        b=np.array([getattr(r,c) for c in B]);wh=b[2:]-b[:2]
        rec=dict(id=r.id,pose_best_iou=0.,pose_head_outside=0.,pose_two_heads=0.,pose_body_outside=0.,pose_wrist_inside=0.,pose_forearm=0.,pose_confirmed_views=0.)
        pool=poses[(poses.frame_id==r.frame_id)&(poses.confidence>=.3)].copy()
        if pool.empty:result.append(rec);continue
        if r.class_name=='person':
            ov=iou(b,pool[B])[0];rec['pose_best_iou']=float(ov.max())
            rec['pose_confirmed_views']=int(pool.loc[ov>=.6,'view'].nunique())
            heads=[];outside=[];bodies=[]
            for j,rr in enumerate(pool.itertuples()):
                bb=np.array([getattr(rr,c) for c in B]);intersection=np.maximum(0,np.minimum(b[2:],bb[2:])-np.maximum(b[:2],bb[:2])).prod()
                qcover=intersection/np.prod(wh)
                k=np.array([[getattr(rr,f'kp{n}_x'),getattr(rr,f'kp{n}_y'),getattr(rr,f'kp{n}_conf')] for n in range(17)])
                valid=k[:5,2]>=.3
                if valid.sum()>=2:
                    head=np.average(k[:5,:2][valid],weights=k[:5,2][valid],axis=0)
                    inside=np.all(head>=b[:2]-4)&np.all(head<=b[2:]+4)
                    hc=float(k[:5,2][valid].mean())
                    if inside and ov[j]>=.1:heads.append((head,rr.view,hc,rr.confidence))
                    # Requires nearby, overlapping detected person, not a head elsewhere in scene.
                    if not inside and qcover>=.25 and ov[j]>=.1 and rr.confidence>=.45:
                        dist=np.linalg.norm((head-np.clip(head,b[:2],b[2:]))/wh)
                        if dist<=.8:outside.append((rr.view,hc))
                if ov[j]>=.3 and rr.confidence>=.4:
                    confident=k[5:,2]>=.55;out=(k[5:,0]<b[0]-8)|(k[5:,0]>b[2]+8)|(k[5:,1]<b[1]-8)|(k[5:,1]>b[3]+8)
                    bodies.append((rr.view,float(np.clip((confident&out).sum()/3,0,1))))
            # Require repetition from distinct views, preventing a single pose hallucination.
            if len(set(v for v,_ in outside))>=2:rec['pose_head_outside']=float(np.mean([p for _,p in outside]))
            byview={}
            for v,p in bodies:byview[v]=max(p,byview.get(v,0.))
            if len(byview)>=2:rec['pose_body_outside']=float(sorted(byview.values(),reverse=True)[1])
            clusters=[]
            for xy,view,hc,dc in sorted(heads,key=lambda z:z[2]*z[3],reverse=True):
                found=next((g for g in clusters if np.linalg.norm(xy-g['xy'])<max(10,min(wh)*.12)),None)
                if found is None:clusters.append(dict(xy=xy,views={view},score=hc*dc))
                else:found['views'].add(view)
            supported=[g for g in clusters if len(g['views'])>=2 and g['score']>=.15]
            rec['pose_two_heads']=float(len(supported)>=2)
        if r.class_name=='nude_hands':
            inside_views=set();forearm_views=set()
            for rr in pool.itertuples():
                if rr.confidence<.45:continue
                for e,w in [(7,9),(8,10)]:
                    wrist=np.array([getattr(rr,f'kp{w}_{c}') for c in ['x','y']]);elbow=np.array([getattr(rr,f'kp{e}_{c}') for c in ['x','y']])
                    if min(getattr(rr,f'kp{w}_conf'),getattr(rr,f'kp{e}_conf'))<.5:continue
                    if np.all(wrist>=b[:2]) and np.all(wrist<=b[2:]):inside_views.add(rr.view)
                    mid=(wrist+elbow)/2
                    if np.all(mid>=b[:2]) and np.all(mid<=b[2:]) and (np.any(wrist<b[:2]-5) or np.any(wrist>b[2:]+5)):forearm_views.add(rr.view)
            rec['pose_wrist_inside']=float(len(inside_views)>=2);rec['pose_forearm']=float(len(forearm_views)>=2)
        result.append(rec)
    return pd.DataFrame(result)

def build(data, detections, out):
    data,detections,out=Path(data),Path(detections),Path(out)
    rows=queries(data)
    raw=reference_features(rows,aligned_csv(detections/'reference_scores.csv',rows.id),
                           pd.read_csv(detections/'fused_references.csv'),
                           pd.read_csv(detections/'zoom_boxes.csv'))
    pose=pd.read_csv(detections/'poses.csv')
    wevis=wevis_features(rows,pd.read_csv(detections/'heads.csv'),pd.read_csv(detections/'hands.csv'),pose)
    pf=pose_features(rows,pose)
    features=pd.concat([raw,wevis.drop(columns='id'),pf.drop(columns='id')],axis=1)
    # Keep exactly the 22 numeric inputs consumed by the released classifiers.
    from .core import read_json, ROOT
    names=read_json(ROOT/'config.json')['features'][:22]
    result=features[names].copy()
    result.insert(0,'id',rows.id)
    result['class_name']=rows.class_name
    out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'static_features.csv',index=False)
    return result
