"""Cluster hand instances, cross-day calibrate box extents, one-to-one assignment."""
from common19 import *
from scipy.optimize import linear_sum_assignment
raw=json.loads((A/'raw_landmarks.json').read_text())+json.loads((A/'raw_landmarks_tight.json').read_text())
def cluster(kind):
    records=[]
    for fid in sorted({r['frame_id'] for r in raw if r['kind']==kind}):
        pool=[r for r in raw if r['kind']==kind and r['frame_id']==fid];groups=[]
        # Largest-support greedy clustering, recompute representative after each merge.
        while pool:
            boxes=np.array([r['bbox'] for r in pool]);ov=iou(boxes,boxes);seed=int((ov>=.4).sum(1).argmax());mask=ov[seed]>=.4
            group=[r for r,m in zip(pool,mask) if m];pool=[r for r,m in zip(pool,mask) if not m]
            xy=np.median([r['landmarks'] for r in group],axis=0);bb=np.r_[xy.min(0),xy.max(0)];overlap=iou(bb,[r['bbox'] for r in group])[0]
            if min(bb[2:]-bb[:2])<3:continue
            rec=dict(frame_id=fid,kind=kind,instance=len(groups),views=len(set((r['id'],r['scale'],r['turn']) for r in group)),turns=len(set(r['turn'] for r in group)),query_support=len(set(r['id'] for r in group)),coherence=float(np.median(overlap)),**dict(zip(B,bb)))
            for k in range(21):
                rec[f'kp{k}_x']=xy[k,0];rec[f'kp{k}_y']=xy[k,1]
            groups.append(rec)
        records.extend(groups)
    return pd.DataFrame(records)
refs={k:cluster(k) for k in ['query_oof','query_test','manual_train']}
for k,v in refs.items():v.to_csv(A/f'landmark_joint_refs_{k}.csv',index=False)
pairs=[]
for (fid,day),gt in h[h.class_name.eq('nude_hands')].groupby(['frame_id','day']):
    rr=refs['manual_train'];rr=rr[rr.frame_id.eq(fid)&rr.views.ge(3)&rr.turns.ge(2)&rr.coherence.ge(.6)]
    ov=iou(gt[B],rr[B])
    if not ov.size:continue
    ii,jj=linear_sum_assignment(-ov)
    for i,j in zip(ii,jj):
        if ov[i,j]<.3:continue
        b=gt.iloc[i][B].to_numpy(float);ref=rr.iloc[j][B].to_numpy(float);wh=ref[2:]-ref[:2];delta=((b[:2]+b[2:])-(ref[:2]+ref[2:]))/2/wh;scale=(b[2:]-b[:2])/wh
        if (scale<.6).any() or (scale>2.5).any() or (abs(delta)>.5).any():continue
        pairs.append(dict(frame_id=fid,day=day,dx=delta[0],dy=delta[1],log_sx=np.log(scale[0]),log_sy=np.log(scale[1]),raw_iou=ov[i,j]))
pairs=pd.DataFrame(pairs);pairs.to_csv(A/'landmark_joint_calibration_pairs.csv',index=False)

def features(rows,kind,fit):
    params=pairs[pairs.day.eq(fit)][['dx','dy','log_sx','log_sy']].median().to_numpy();out=[];rr=refs[kind]
    for fid,g in rows.groupby('frame_id',sort=False):
        allr=rr[rr.frame_id.eq(fid)&rr.views.ge(2)&rr.coherence.ge(.5)].reset_index(drop=True)
        rb=allr[B].to_numpy();wh=rb[:,2:]-rb[:,:2];cc=(rb[:,:2]+rb[:,2:])/2;center=cc+wh*params[:2];sz=wh*np.exp(params[2:]);cb=np.c_[center-sz/2,center+sz/2]
        assigned={}
        hands=g[g.class_name.eq('nude_hands')];ovs=iou(hands[B],cb)
        if ovs.size:
            ii,jj=linear_sum_assignment(-ovs)
            assigned={hands.iloc[i].id:int(j) for i,j in zip(ii,jj) if ovs[i,j]>=.1}
        for r in g.itertuples():
            rec=dict(id=r.id,found=False,assigned=False,iou=0.,raw_iou=0.,views=0,turns=0,coherence=0.,point_fraction=0.,tip_fraction=0.,palm_fraction=0.,wrist_inside=False,palm_center_inside=False,projection=0.,instances=0,ref_area_ratio=0.,ref_x1=0.,ref_y1=0.,ref_x2=0.,ref_y2=0.)
            if r.class_name!='nude_hands' or not len(allr):out.append(rec);continue
            b=np.array([getattr(r,c) for c in B]);ov=iou(b,cb)[0];j=int(ov.argmax());ref=allr.iloc[j];xy=np.array([[ref[f'kp{k}_x'],ref[f'kp{k}_y']] for k in range(21)]);inside=((xy>=b[:2])&(xy<=b[2:])).all(1)
            palm=xy[[0,5,9,13,17]].mean(0);axis=xy[9]-xy[0];querycenter=(b[:2]+b[2:])/2;length2=max(axis@axis,1)
            rec.update(found=True,assigned=assigned.get(r.id,-1)==j,iou=float(ov[j]),raw_iou=float(iou(b,rb[j])[0,0]),views=int(ref.views),turns=int(ref.turns),coherence=float(ref.coherence),point_fraction=float(inside.mean()),tip_fraction=float(inside[[4,8,12,16,20]].mean()),palm_fraction=float(inside[[0,5,9,13,17]].mean()),wrist_inside=bool(inside[0]),palm_center_inside=bool(((palm>=b[:2])&(palm<=b[2:])).all()),projection=float((querycenter-xy[0])@axis/length2),instances=int(((center>=b[:2])&(center<=b[2:])).all(1).sum()),ref_area_ratio=float(np.prod(cb[j,2:]-cb[j,:2])/np.prod(b[2:]-b[:2])),**dict(zip(['ref_'+c for c in B],cb[j])))
            out.append(rec)
    return pd.DataFrame(out).set_index('id').loc[rows.id].reset_index()
oof=[];test=[]
for fit,held in zip(DAYS,DAYS[::-1]):
    v=features(q[q.day.eq(held)],'query_oof',fit);t=features(te,'query_test',fit)
    v.to_csv(A/f'landmark_joint_held_features_fit_{fit}.csv',index=False);t.to_csv(A/f'landmark_joint_test_features_fit_{fit}.csv',index=False);oof.append(v);test.append(t)
oof=pd.concat(oof).set_index('id').loc[q.id].reset_index()
test=pd.concat(test).groupby('id',sort=False).mean().loc[te.id].reset_index()
oof.to_csv(A/'landmark_joint_features_oof.csv',index=False);test.to_csv(A/'landmark_joint_features_test.csv',index=False)
print('CALIBRATION',pairs.groupby('day').size().to_dict());print('QUERY COVERAGE',oof.found.sum(),test.found.sum())
