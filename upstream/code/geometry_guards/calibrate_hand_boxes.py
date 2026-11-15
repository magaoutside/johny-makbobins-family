from common16 import *
from scipy.optimize import linear_sum_assignment
refs=pd.read_csv(V16/'wevis_hands_references.csv')
cache={r.id: {'roi': np.r_[np.maximum(0, (np.array([r.x1,r.y1])+np.array([r.x2,r.y2]))/2-np.maximum((np.array([r.x2,r.y2])-np.array([r.x1,r.y1]))*4,160)/2), np.minimum([1280,720], (np.array([r.x1,r.y1])+np.array([r.x2,r.y2]))/2+np.maximum((np.array([r.x2,r.y2])-np.array([r.x1,r.y1]))*4,160)/2)].astype(int).tolist()} for r in pd.concat([q,te],ignore_index=True).itertuples() if r.class_name=='nude_hands'}
# Reject reference boxes truncated by the crop, then merge repeated observations.
eligible=[]
for r in refs.itertuples():
    b=np.array([getattr(r,c) for c in B]);roi=np.array(cache[r.query_id]['roi'])
    if min(b[0]-roi[0],b[1]-roi[1],roi[2]-b[2],roi[3]-b[3])>=2:eligible.append(r.Index)
refs=refs.loc[eligible].copy()
def dedup(rr):
    keep=[]
    for _,row in rr.iterrows():
        if not keep or iou(row[B],rr.loc[keep,B]).max()<.65:keep.append(row.name)
    return rr.loc[keep]
refs=pd.concat([dedup(g) for _,g in refs.groupby('frame_id')]).reset_index(drop=True)
pairs=[]
for (fid,day),g in h[h.class_name.eq('nude_hands')].groupby(['frame_id','day']):
    rr=refs[refs.frame_id.eq(fid)].reset_index(drop=True)
    if rr.empty:continue
    ov=iou(g[B],rr[B]);ii,jj=linear_sum_assignment(-ov)
    for i,j in zip(ii,jj):
        if ov[i,j]<.2:continue
        target=g.iloc[i][B].to_numpy(float);source=rr.iloc[j][B].to_numpy(float);sw=source[2:]-source[:2];tw=target[2:]-target[:2]
        offset=((target[:2]+target[2:])-(source[:2]+source[2:]))/2/sw;ratio=tw/sw
        if (ratio>.5).all() and (ratio<3).all() and (np.abs(offset)<.75).all():
            pairs.append(dict(frame_id=fid,day=day,dx=offset[0],dy=offset[1],sx=ratio[0],sy=ratio[1],initial_iou=float(ov[i,j]),**dict(zip(['ref_'+c for c in B],source)),**dict(zip(['gt_'+c for c in B],target))))
pairs=pd.DataFrame(pairs);pairs.to_csv(V16/'hand_calibration_pairs.csv',index=False)
def parameters(fit=None):
    p=pairs if fit is None else pairs[pairs.day.eq(fit)]
    assert len(p)>=5
    v=p[['dx','dy','sx','sy']].median().to_numpy();v[:2]=np.clip(v[:2],-.3,.3);v[2:]=np.clip(v[2:],.8,1.8)
    return v
def evidence(rows,fit=None):
    dx,dy,sx,sy=parameters(fit);out=[]
    for (fid,c),g in rows.groupby(['frame_id','class_name']):
        rr=refs[refs.frame_id.eq(fid)]
        if c!='nude_hands' or rr.empty:
            for ix in g.index:out.append(dict(idx=ix,calibrated_hand_iou=0.,assigned=False,calibrated_instances=0))
            continue
        bb=rr[B].to_numpy();wh=bb[:,2:]-bb[:,:2];cc=(bb[:,:2]+bb[:,2:])/2+wh*[dx,dy]
        rb=np.c_[cc-wh*[sx,sy]/2,cc+wh*[sx,sy]/2];ov=iou(g[B],rb)
        reward=np.c_[np.where(ov>=.4,min(ov.shape)+1+ov,0),np.zeros((len(g),len(g)))];ii,jj=linear_sum_assignment(-reward);matches={i:j for i,j in zip(ii,jj) if j<len(rb) and ov[i,j]>=.4}
        for j,ix in enumerate(g.index):
            k=matches.get(j,int(ov[j].argmax()));b=rows.loc[ix,B].to_numpy(float);inside=((cc>=b[:2])&(cc<=b[2:])).all(1)
            out.append(dict(idx=ix,calibrated_hand_iou=float(ov[j,k]),assigned=j in matches,calibrated_instances=int(inside.sum()),ref_x1=rb[k,0],ref_y1=rb[k,1],ref_x2=rb[k,2],ref_y2=rb[k,3]))
    return pd.DataFrame(out).set_index('idx').reindex(rows.index)
parts=[]
for fit,held in zip(DAYS,DAYS[::-1]):parts.append(evidence(q[q.day.eq(held)],fit))
f=pd.concat(parts).reindex(q.index);ft=evidence(te)
q[['id']].join(f).to_csv(V16/'calibrated_hand_oof.csv',index=False);te[['id']].join(ft).to_csv(V16/'calibrated_hand_test.csv',index=False)
param={fit:parameters(None if fit=='all' else fit).tolist() for fit in DAYS+['all']};(V16/'hand_calibration_parameters.json').write_text(json.dumps(param,indent=2));print('CALIBRATION',len(pairs),param)
for threshold in [.6,.7]:
    for w in [.2,.35]:
        outputs=[];counts=[]
        for rows,base,feat,test in [(q,q.score.to_numpy(),f,False),(te,current,ft,True)]:
            sem=pd.read_csv(V16/f'semantic_tight_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id,'p_nude_hands'].to_numpy()
            led=pd.read_csv(V15/'hand_group_test.csv') if test else pd.concat([pd.read_csv(V15/f'hand_group_oof_{d}.csv') for d in DAYS]);losers=led.loc[led.reason.eq('duplicate_loser'),'id']
            mask=rows.class_name.eq('nude_hands').to_numpy()&feat.assigned.to_numpy()&(feat.calibrated_hand_iou.to_numpy()>=threshold)&(feat.calibrated_instances.to_numpy()==1)&(sem>=.8)&~rows.id.isin(losers).to_numpy()
            target=np.clip((.85-feat.calibrated_hand_iou.to_numpy())/.5,0,1);s=base.copy();s[mask]=(1-w)*s[mask]+w*np.minimum(s[mask],target[mask]);outputs.append(s);counts.append(int(mask.sum()))
        register(f'hand_calibrated_iou{threshold}_w{w}',*outputs,details=dict(iou_threshold=threshold,weight=w,parameters=param,eligible_counts=counts,fit='opposite day manual GT only'))

# Contradiction from two separate local detectors; absence alone is not enough.
for w in [.35,.6]:
    outputs=[]
    for rows,base,test in [(q,q.score.to_numpy(),False),(te,current,True)]:
        wf=pd.read_csv(V16/f'wevis_features_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id]
        yf=pd.read_csv(V15/f'hand_specialist_{"test" if test else "oof"}_evidence_0.35.csv').set_index('id').loc[rows.id]
        sem=pd.read_csv(V16/f'semantic_tight_{"test" if test else "oof"}.csv').set_index('id').loc[rows.id,'p_nude_hands'].to_numpy()
        mask=rows.class_name.eq('nude_hands').to_numpy()&(wf.hand_iou.to_numpy()<.15)&(yf.local_iou.to_numpy()<.15)&(sem<.8)&(base<.75)
        outputs.append(base+w*(1-base)*mask)
    register(f'hand_contradiction_{w}',*outputs,details=dict(weight=w,wevis_iou_max=.15,yolo_iou_max=.15,semantic_hand_max=.8,base_max=.75))
